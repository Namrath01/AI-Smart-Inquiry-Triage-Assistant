"""Phase 2 tests: structured classification (+ retry/fallback), retrieval
integration, Top-K-derived priority (baseline + candidate), deterministic
routing, and the partial LangGraph workflow.

The chat model is mocked for all deterministic retry/failure/ordering tests
via dependency injection (classify_inquiry accepts a structured_classifier
callable) or monkeypatching -- only test_real_structured_classification_smoke
makes an actual call to the local qwen2.5:1.5b-instruct model.
"""

from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from src import main
from src.ingestion import get_routing_map
from src.taxonomy import get_category_names


class _FakeResult:
    def __init__(self, category):
        self.category = category


class FakeClassifier:
    """Injectable stand-in for the real structured-classifier callable.

    Behaviors is a list consumed one per call: a string becomes a successful
    result with that category, an Exception instance is raised.
    """

    def __init__(self, behaviors):
        self.behaviors = list(behaviors)
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        behavior = self.behaviors.pop(0)
        if isinstance(behavior, Exception):
            raise behavior
        return _FakeResult(behavior)


# ---------------------------------------------------------------------------
# Structured classification schema
# ---------------------------------------------------------------------------


def test_structured_schema_accepts_only_canonical_categories():
    schema = main._build_classification_schema()
    valid = schema(category="service")
    assert valid.category == "service"
    with pytest.raises(ValidationError):
        schema(category="not_a_real_category")


# ---------------------------------------------------------------------------
# Classification retry / fallback mechanics
# ---------------------------------------------------------------------------


def test_first_attempt_success_sets_llm_source():
    fake = FakeClassifier(["service"])
    category, source, failed = main.classify_inquiry("some query", structured_classifier=fake)
    assert category == "service"
    assert source == "llm"
    assert failed is False
    assert fake.calls == 1


def test_retry_success_sets_llm_retry_source():
    fake = FakeClassifier([ValueError("malformed"), "billing"])
    category, source, failed = main.classify_inquiry("some query", structured_classifier=fake)
    assert category == "billing"
    assert source == "llm_retry"
    assert failed is False
    assert fake.calls == 2


def test_double_failure_invokes_deterministic_fallback():
    fake = FakeClassifier([ValueError("bad"), ValueError("bad again")])
    category, source, failed = main.classify_inquiry(
        "I need an oil change appointment.", structured_classifier=fake
    )
    assert fake.calls == 2
    assert source in ("taxonomy_fallback", "safe_default")
    assert failed is True


def test_fallback_with_clear_evidence_sets_taxonomy_fallback_and_failed():
    fake = FakeClassifier([ValueError("bad"), ValueError("bad again")])
    category, source, failed = main.classify_inquiry(
        "I need an oil change appointment.", structured_classifier=fake
    )
    assert category == "service"
    assert source == "taxonomy_fallback"
    assert failed is True


def test_safe_default_path_returns_other():
    fake = FakeClassifier([ValueError("bad"), ValueError("bad again")])
    category, source, failed = main.classify_inquiry(
        "asdkjhasdlkjh qweoiuqwoeiu", structured_classifier=fake
    )
    assert category == "other"
    assert source == "safe_default"
    assert failed is True


def test_infrastructure_error_propagates_without_retry_or_fallback():
    fake = FakeClassifier([httpx.ConnectError("boom")])
    with pytest.raises(main.InfrastructureError):
        main.classify_inquiry("some query", structured_classifier=fake)
    assert fake.calls == 1


def test_infrastructure_error_on_retry_still_propagates():
    fake = FakeClassifier([ValueError("bad"), httpx.ConnectError("boom")])
    with pytest.raises(main.InfrastructureError):
        main.classify_inquiry("some query", structured_classifier=fake)
    assert fake.calls == 2


# ---------------------------------------------------------------------------
# Retrieval integration (real embedding backend, minimal calls)
# ---------------------------------------------------------------------------


def test_retrieval_is_not_filtered_by_predicted_category(monkeypatch):
    def fake_classify_inquiry(query, structured_classifier=None):
        return "billing", "llm", False

    monkeypatch.setattr(main, "classify_inquiry", fake_classify_inquiry)

    graph = main.build_partial_graph()
    result = graph.invoke(
        {"query": "My brakes are making a grinding noise.", "top_k": 10, "confidence_threshold": 0.5}
    )

    assert result["category"] == "billing"
    retrieved_categories = {r["category"] for r in result["retrieved_past_cases"]}
    # Retrieval must stay an independent semantic signal: even though the
    # (deliberately wrong, for this test) predicted category is "billing",
    # the actually-similar neighbors are "service" cases and must still
    # surface, proving no category filtering happened.
    assert "service" in retrieved_categories


# ---------------------------------------------------------------------------
# Priority: majority-vote baseline
# ---------------------------------------------------------------------------


def test_majority_vote_returns_expected_result():
    retrieved = [
        {"priority": "high", "similarity": 0.9},
        {"priority": "high", "similarity": 0.8},
        {"priority": "low", "similarity": 0.5},
    ]
    assert main.majority_vote_priority(retrieved) == "high"


def test_majority_vote_tie_uses_similarity_tiebreak():
    retrieved = [
        {"priority": "high", "similarity": 0.9},
        {"priority": "high", "similarity": 0.1},
        {"priority": "medium", "similarity": 0.6},
        {"priority": "medium", "similarity": 0.6},
    ]
    # count tied 2-2; similarity sums: high=1.0, medium=1.2 -> medium wins.
    assert main.majority_vote_priority(retrieved) == "medium"


def test_majority_vote_final_tie_uses_severity_order():
    retrieved = [
        {"priority": "high", "similarity": 0.5},
        {"priority": "medium", "similarity": 0.5},
    ]
    # count tied 1-1; similarity sums also tied 0.5-0.5 -> severity: high wins.
    assert main.majority_vote_priority(retrieved) == "high"


def test_majority_vote_rejects_empty_input():
    with pytest.raises(ValueError):
        main.majority_vote_priority([])


# ---------------------------------------------------------------------------
# Priority: similarity-weighted candidate
# ---------------------------------------------------------------------------


def test_similarity_weighted_returns_expected_result():
    retrieved = [
        {"priority": "low", "similarity": 0.9},
        {"priority": "high", "similarity": 0.3},
        {"priority": "high", "similarity": 0.3},
    ]
    # weighted sums: low=0.9, high=0.6 -> low wins despite fewer neighbors,
    # demonstrating this differs from a plain majority vote by design.
    assert main.similarity_weighted_priority(retrieved) == "low"


def test_similarity_weighted_tie_uses_severity_order():
    retrieved = [
        {"priority": "low", "similarity": 0.5},
        {"priority": "medium", "similarity": 0.5},
    ]
    assert main.similarity_weighted_priority(retrieved) == "medium"


def test_similarity_weighted_rejects_empty_input():
    with pytest.raises(ValueError):
        main.similarity_weighted_priority([])


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_routing_uses_data_derived_map():
    routing_map = get_routing_map()
    for category, queue in routing_map.items():
        assert main.route_category(category) == queue


def test_routing_raises_for_unknown_category():
    with pytest.raises(main.RoutingError):
        main.route_category("not_a_real_category")


# ---------------------------------------------------------------------------
# Partial LangGraph workflow
# ---------------------------------------------------------------------------


def test_partial_graph_executes_in_expected_order(monkeypatch):
    call_order = []

    def fake_classify_node(state):
        call_order.append("classify")
        return {"category": "service", "classification_source": "llm", "classification_failed": False}

    def fake_retrieve_node(state):
        call_order.append("retrieve")
        return {
            "retrieved_past_cases": [
                {
                    "case_id": "X",
                    "inquiry_text": "t",
                    "category": "service",
                    "priority": "high",
                    "routed_queue": "Service Scheduling Team",
                    "similarity": 0.9,
                }
            ]
        }

    def fake_priority_node(state):
        call_order.append("determine_priority")
        return {"priority": "high"}

    def fake_route_node(state):
        call_order.append("route")
        return {"routed_queue": "Service Scheduling Team"}

    monkeypatch.setattr(main, "classify_node", fake_classify_node)
    monkeypatch.setattr(main, "retrieve_node", fake_retrieve_node)
    monkeypatch.setattr(main, "determine_priority_node", fake_priority_node)
    monkeypatch.setattr(main, "route_node", fake_route_node)

    graph = main.build_partial_graph()
    result = graph.invoke({"query": "test", "top_k": 3, "confidence_threshold": 0.5})

    assert call_order == ["classify", "retrieve", "determine_priority", "route"]
    assert result["category"] == "service"
    assert result["routed_queue"] == "Service Scheduling Team"


# ---------------------------------------------------------------------------
# Real-model integration smoke test (the one deliberate real Ollama call)
# ---------------------------------------------------------------------------


def test_real_structured_classification_smoke():
    category, source, failed = main.classify_inquiry("My brakes are making a grinding noise.")
    assert category in get_category_names()
    assert source in ("llm", "llm_retry")
    assert failed is False


# Regression: failed model requests are not unusable model answers.
@pytest.mark.parametrize("error_type", [httpx.ReadError, httpx.RemoteProtocolError])
@pytest.mark.parametrize("on_retry", [False, True])
def test_transport_failure_stops_classification(monkeypatch, error_type, on_retry):
    error = error_type("transport failed")
    fake = FakeClassifier(([ValueError("malformed")] if on_retry else []) + [error])
    monkeypatch.setattr(
        main, "_deterministic_classification_fallback",
        lambda query: pytest.fail("transport failure entered taxonomy fallback"),
    )
    with pytest.raises(main.InfrastructureError) as caught:
        main.classify_inquiry("oil change", structured_classifier=fake)
    assert caught.value.__cause__ is error
    assert fake.calls == (2 if on_retry else 1)


@pytest.mark.parametrize("on_retry", [False, True])
def test_ollama_request_error_stops_classification(monkeypatch, on_retry):
    error = main.OllamaResponseError("request rejected", status_code=400)
    fake = FakeClassifier(([ValueError("malformed")] if on_retry else []) + [error])
    monkeypatch.setattr(
        main, "_deterministic_classification_fallback",
        lambda query: pytest.fail("request failure entered taxonomy fallback"),
    )
    with pytest.raises(main.OllamaResponseError) as caught:
        main.classify_inquiry("oil change", structured_classifier=fake)
    assert caught.value is error
    assert fake.calls == (2 if on_retry else 1)


@pytest.mark.parametrize("error", [
    httpx.ReadError("read failed"),
    httpx.RemoteProtocolError("server disconnected"),
    main.OllamaResponseError("request rejected", status_code=400),
])
def test_failed_classification_request_cannot_produce_triage(monkeypatch, error):
    fake = FakeClassifier([error])
    monkeypatch.setattr(main, "_default_structured_classifier", lambda: fake)
    monkeypatch.setattr(
        main, "retrieve_similar_cases",
        lambda *args: pytest.fail("pipeline continued after failed model request"),
    )
    expected = main.OllamaResponseError if isinstance(error, main.OllamaResponseError) else main.InfrastructureError
    with pytest.raises(expected):
        main.triage_inquiry("I need an oil change.", 5, 0.5)
    assert fake.calls == 1
