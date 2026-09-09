"""Phase 3 tests: confidence signals, confidence strategies, classification-
failure override, threshold escalation semantics, resolution-note generation
(+ retry/failure), the complete LangGraph, and the public triage_inquiry
contract.

The chat model is mocked everywhere except
test_real_end_to_end_smoke, which is the single deliberate real-Ollama call.
"""

from __future__ import annotations

import json

import httpx
import pytest

from src import main


class _FakeChatMessage:
    def __init__(self, content):
        self.content = content


class FakeChat:
    """Injectable stand-in for the resolution chat callable. A string
    behavior becomes a successful `.content`-bearing response; an Exception
    instance is raised.
    """

    def __init__(self, behaviors):
        self.behaviors = list(behaviors)
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        behavior = self.behaviors.pop(0)
        if isinstance(behavior, Exception):
            raise behavior
        return _FakeChatMessage(behavior)


SAMPLE_RETRIEVED = [
    {"case_id": "A", "inquiry_text": "t1", "category": "service", "priority": "high", "routed_queue": "Service Scheduling Team", "similarity": 0.9},
    {"case_id": "B", "inquiry_text": "t2", "category": "service", "priority": "high", "routed_queue": "Service Scheduling Team", "similarity": 0.6},
    {"case_id": "C", "inquiry_text": "t3", "category": "billing", "priority": "low", "routed_queue": "Billing & Payments Team", "similarity": 0.3},
]


# ---------------------------------------------------------------------------
# Confidence signals
# ---------------------------------------------------------------------------


def test_retrieval_strength_calculation():
    assert main.retrieval_strength(SAMPLE_RETRIEVED) == 0.9


def test_category_agreement_calculation():
    # matching (service) similarity sum = 0.9 + 0.6 = 1.5; total = 0.9+0.6+0.3=1.8
    result = main.category_agreement(SAMPLE_RETRIEVED, "service")
    assert result == pytest.approx(1.5 / 1.8)


def test_category_margin_calculation():
    # service evidence = 1.5, billing evidence (runner-up) = 0.3, total = 1.8
    # margin = max(0, 1.5 - 0.3) / 1.8 = 1.2 / 1.8
    result = main.category_margin(SAMPLE_RETRIEVED, "service")
    assert result == pytest.approx(1.2 / 1.8)


def test_confidence_signals_stay_in_unit_range():
    for predicted in ("service", "billing", "other"):
        rs = main.retrieval_strength(SAMPLE_RETRIEVED)
        ca = main.category_agreement(SAMPLE_RETRIEVED, predicted)
        cm = main.category_margin(SAMPLE_RETRIEVED, predicted)
        assert 0.0 <= rs <= 1.0
        assert 0.0 <= ca <= 1.0
        assert 0.0 <= cm <= 1.0


def test_no_predicted_category_support_gives_zero_margin():
    # "other" has no retrieved neighbors at all.
    assert main.category_margin(SAMPLE_RETRIEVED, "other") == 0.0
    assert main.category_agreement(SAMPLE_RETRIEVED, "other") == 0.0


def test_zero_total_similarity_handled_safely():
    zero_sim = [
        {"case_id": "A", "inquiry_text": "t1", "category": "service", "priority": "high", "routed_queue": "x", "similarity": 0.0},
        {"case_id": "B", "inquiry_text": "t2", "category": "billing", "priority": "low", "routed_queue": "y", "similarity": 0.0},
    ]
    assert main.category_agreement(zero_sim, "service") == 0.0
    assert main.category_margin(zero_sim, "service") == 0.0


def test_confidence_signal_functions_reject_empty_input():
    with pytest.raises(ValueError):
        main.retrieval_strength([])
    with pytest.raises(ValueError):
        main.category_agreement([], "service")
    with pytest.raises(ValueError):
        main.category_margin([], "service")


# ---------------------------------------------------------------------------
# Confidence strategies
# ---------------------------------------------------------------------------


def test_retrieval_strength_confidence_baseline():
    assert main.compute_confidence(0.8, 0.5, 0.2, strategy="retrieval_strength") == 0.8


def test_evidence_mean_confidence_candidate():
    result = main.compute_confidence(0.8, 0.5, 0.2, strategy="evidence_mean")
    assert result == pytest.approx((0.8 + 0.5 + 0.2) / 3.0)


def test_compute_confidence_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        main.compute_confidence(0.8, 0.5, 0.2, strategy="not_a_real_strategy")


# ---------------------------------------------------------------------------
# Classification-failure override + escalation semantics
# ---------------------------------------------------------------------------


def test_classification_failed_forces_zero_confidence():
    confidence, escalated = main.compute_confidence_and_escalation(
        classification_failed=True,
        retrieval_strength_value=0.99,
        category_agreement_value=0.99,
        category_margin_value=0.99,
        confidence_threshold=0.1,
    )
    assert confidence == 0.0


def test_classification_failed_forces_escalated_true():
    confidence, escalated = main.compute_confidence_and_escalation(
        classification_failed=True,
        retrieval_strength_value=0.99,
        category_agreement_value=0.99,
        category_margin_value=0.99,
        confidence_threshold=0.1,
    )
    assert escalated is True


def test_confidence_below_threshold_escalates():
    confidence, escalated = main.compute_confidence_and_escalation(
        classification_failed=False,
        retrieval_strength_value=0.4,
        category_agreement_value=0.4,
        category_margin_value=0.4,
        confidence_threshold=0.5,
    )
    assert confidence == 0.4
    assert escalated is True


def test_confidence_equal_to_threshold_does_not_escalate():
    confidence, escalated = main.compute_confidence_and_escalation(
        classification_failed=False,
        retrieval_strength_value=0.5,
        category_agreement_value=0.5,
        category_margin_value=0.5,
        confidence_threshold=0.5,
    )
    assert confidence == 0.5
    assert escalated is False


def test_confidence_above_threshold_does_not_escalate():
    confidence, escalated = main.compute_confidence_and_escalation(
        classification_failed=False,
        retrieval_strength_value=0.9,
        category_agreement_value=0.9,
        category_margin_value=0.9,
        confidence_threshold=0.5,
    )
    assert escalated is False


@pytest.mark.parametrize("bad_threshold", [-0.01, 1.01, 2.0, -1.0])
def test_invalid_threshold_rejected(bad_threshold):
    with pytest.raises(ValueError):
        main.compute_confidence_and_escalation(
            classification_failed=False,
            retrieval_strength_value=0.5,
            category_agreement_value=0.5,
            category_margin_value=0.5,
            confidence_threshold=bad_threshold,
        )


# ---------------------------------------------------------------------------
# Resolution note generation
# ---------------------------------------------------------------------------


def test_resolution_note_is_non_empty():
    fake = FakeChat(["  Contact the customer to confirm brake symptoms and schedule service.  \n"])
    note = main.generate_resolution_note(
        "My brakes are grinding.", "service", "high", SAMPLE_RETRIEVED, chat=fake
    )
    assert note.strip() != ""
    assert note == "Contact the customer to confirm brake symptoms and schedule service."


def test_resolution_note_retries_on_unusable_output():
    fake = FakeChat(["   ", "Escalate to the service scheduling team for review."])
    note = main.generate_resolution_note(
        "My brakes are grinding.", "service", "high", SAMPLE_RETRIEVED, chat=fake
    )
    assert note == "Escalate to the service scheduling team for review."
    assert fake.calls == 2


def test_resolution_note_raises_after_second_unusable_output():
    fake = FakeChat(["", "   "])
    with pytest.raises(main.ResolutionGenerationError):
        main.generate_resolution_note(
            "My brakes are grinding.", "service", "high", SAMPLE_RETRIEVED, chat=fake
        )
    assert fake.calls == 2


def test_infrastructure_error_during_resolution_propagates():
    fake = FakeChat([httpx.ConnectError("boom")])
    with pytest.raises(main.InfrastructureError):
        main.generate_resolution_note(
            "My brakes are grinding.", "service", "high", SAMPLE_RETRIEVED, chat=fake
        )
    assert fake.calls == 1


# ---------------------------------------------------------------------------
# 1-2 sentence contract enforcement
# ---------------------------------------------------------------------------


def test_count_sentences_helper():
    assert main._count_sentences("") == 0
    assert main._count_sentences("One sentence") == 1
    assert main._count_sentences("One sentence.") == 1
    assert main._count_sentences("One. Two.") == 2
    assert main._count_sentences("One. Two. Three.") == 3


def test_resolution_note_with_one_sentence_is_accepted():
    fake = FakeChat(["Please contact the customer for more details."])
    note = main.generate_resolution_note(
        "My brakes are grinding.", "service", "high", SAMPLE_RETRIEVED, chat=fake
    )
    assert note == "Please contact the customer for more details."
    assert fake.calls == 1


def test_resolution_note_with_two_sentences_is_accepted():
    fake = FakeChat(["Please schedule a service visit. This will address the brake issue."])
    note = main.generate_resolution_note(
        "My brakes are grinding.", "service", "high", SAMPLE_RETRIEVED, chat=fake
    )
    assert note == "Please schedule a service visit. This will address the brake issue."
    assert fake.calls == 1


def test_resolution_note_with_more_than_two_sentences_triggers_retry():
    too_long = "First sentence here. Second sentence here. Third sentence here."
    fake = FakeChat([too_long, "A single acceptable sentence."])
    note = main.generate_resolution_note(
        "My brakes are grinding.", "service", "high", SAMPLE_RETRIEVED, chat=fake
    )
    assert note == "A single acceptable sentence."
    assert fake.calls == 2


def test_resolution_note_with_more_than_two_sentences_on_both_attempts_raises():
    too_long = "First sentence here. Second sentence here. Third sentence here."
    fake = FakeChat([too_long, too_long])
    with pytest.raises(main.ResolutionGenerationError):
        main.generate_resolution_note(
            "My brakes are grinding.", "service", "high", SAMPLE_RETRIEVED, chat=fake
        )
    assert fake.calls == 2


# ---------------------------------------------------------------------------
# Prompt inspection (grounding / anti-fabrication)
# ---------------------------------------------------------------------------


def test_resolution_prompt_contains_query_category_priority_and_case_context():
    user_prompt = main._resolution_user_prompt(
        "My brakes are grinding.",
        "service",
        "high",
        [
            {
                "case_id": "C1",
                "inquiry_text": "Similar brake issue reported last week.",
                "category": "service",
                "priority": "high",
                "routed_queue": "Service Scheduling Team",
                "similarity": 0.9,
            }
        ],
    )
    assert "My brakes are grinding." in user_prompt
    assert "service" in user_prompt
    assert "high" in user_prompt
    assert "Similar brake issue reported last week." in user_prompt


def test_resolution_prompt_treats_retrieved_cases_as_pattern_not_facts():
    """(A) Prompt-inspection: historical-case-specific facts must not be
    transferable into the current inquiry -- retrieved cases are framed only
    as a handling-pattern example, never as a fact source.
    """
    lowered = main._resolution_system_prompt().lower()
    assert "examples of how similar inquiries were handled" in lowered
    assert "do not copy or infer" in lowered
    assert "case-specific facts, events, entitlements, statuses, promises" in lowered
    assert (
        "only treat information explicitly stated in the current inquiry as "
        "current-case facts" in lowered
    )


def test_resolution_prompt_prohibits_unsupported_business_state_assertions():
    """(B) Prompt-inspection: unknown business/company/account/event state
    must not be asserted -- generalized categories, not special-cased words.
    """
    lowered = main._resolution_system_prompt().lower()
    for banned_topic in [
        "company or product plans",
        "whether a launch or event exists",
        "availability of anything",
        "prices",
        "account, payment, refund, or warranty status",
        "order or delivery status",
        "vehicle-specific facts not supplied by the customer",
        "approval, eligibility, or entitlement",
        "promises or commitments",
        "already been performed",
    ]:
        assert banned_topic in lowered

    # unsupported UI/procedural detail fabrication guidance (preserved from
    # the earlier grounding correction)
    assert "ui button" in lowered or "interface controls" in lowered
    assert "menu path" in lowered or "procedural steps" in lowered


def test_resolution_prompt_instructs_verification_when_action_depends_on_unknown_info():
    """(C) Prompt-inspection: when the correct action depends on unknown
    information, the model must be told to recommend verification/review/
    routing rather than asserting the action should or will occur.
    """
    lowered = main._resolution_system_prompt().lower()
    assert "recommend verification, review, or routing" in lowered
    assert "rather than asserting that the action should or will occur" in lowered


# ---------------------------------------------------------------------------
# Resolution generated even for escalated (classification_failed) cases
# ---------------------------------------------------------------------------


def test_resolution_generated_even_when_classification_failed(monkeypatch):
    def fake_classify_node(state):
        return {
            "category": "other",
            "classification_source": "safe_default",
            "classification_failed": True,
        }

    monkeypatch.setattr(main, "classify_node", fake_classify_node)
    monkeypatch.setattr(main, "generate_resolution_note", lambda *a, **k: "Draft note for human review.")

    graph = main.build_complete_graph()
    result = graph.invoke({"query": "some ambiguous inquiry", "top_k": 3, "confidence_threshold": 0.5})

    assert result["resolution_notes"] == "Draft note for human review."
    assert result["classification_failed"] is True
    assert result["confidence"] == 0.0
    assert result["escalated"] is True


# ---------------------------------------------------------------------------
# Complete LangGraph node order
# ---------------------------------------------------------------------------


def test_complete_graph_executes_in_expected_order(monkeypatch):
    call_order = []

    def fake_classify(state):
        call_order.append("classify")
        return {"category": "service", "classification_source": "llm", "classification_failed": False}

    def fake_retrieve(state):
        call_order.append("retrieve")
        return {"retrieved_past_cases": SAMPLE_RETRIEVED}

    def fake_priority(state):
        call_order.append("determine_priority")
        return {"priority": "high"}

    def fake_route(state):
        call_order.append("route")
        return {"routed_queue": "Service Scheduling Team"}

    def fake_resolution(state):
        call_order.append("generate_resolution")
        return {"resolution_notes": "Note."}

    def fake_finalize(state):
        call_order.append("finalize_confidence_and_escalation")
        return {
            "retrieval_strength": 0.9,
            "category_agreement": 1.0,
            "category_margin": 1.0,
            "confidence": 0.9,
            "escalated": False,
        }

    monkeypatch.setattr(main, "classify_node", fake_classify)
    monkeypatch.setattr(main, "retrieve_node", fake_retrieve)
    monkeypatch.setattr(main, "determine_priority_node", fake_priority)
    monkeypatch.setattr(main, "route_node", fake_route)
    monkeypatch.setattr(main, "generate_resolution_node", fake_resolution)
    monkeypatch.setattr(main, "finalize_confidence_and_escalation_node", fake_finalize)

    graph = main.build_complete_graph()
    result = graph.invoke({"query": "test", "top_k": 3, "confidence_threshold": 0.5})

    assert call_order == [
        "classify",
        "retrieve",
        "determine_priority",
        "route",
        "generate_resolution",
        "finalize_confidence_and_escalation",
    ]
    assert result["confidence"] == 0.9
    assert result["escalated"] is False


# ---------------------------------------------------------------------------
# Public triage_inquiry contract
# ---------------------------------------------------------------------------


def _mock_llm_layer(monkeypatch, category="service", note="Suggested next step: contact the customer."):
    def fake_classify_node(state):
        return {"category": category, "classification_source": "llm", "classification_failed": False}

    monkeypatch.setattr(main, "classify_node", fake_classify_node)
    monkeypatch.setattr(main, "generate_resolution_note", lambda *a, **k: note)


def test_triage_inquiry_returns_exact_public_keys(monkeypatch):
    _mock_llm_layer(monkeypatch)
    result = main.triage_inquiry("My brakes are making a grinding noise.", top_k=3, confidence_threshold=0.5)

    expected_keys = {
        "query",
        "category",
        "priority",
        "routed_queue",
        "confidence",
        "resolution_notes",
        "retrieved_past_cases",
        "escalated",
    }
    assert set(result.keys()) == expected_keys


def test_triage_inquiry_result_is_json_serializable(monkeypatch):
    _mock_llm_layer(monkeypatch, category="billing")
    result = main.triage_inquiry("I was charged twice for my payment.", top_k=3, confidence_threshold=0.5)
    serialized = json.dumps(result)
    assert isinstance(serialized, str)


def test_triage_inquiry_retrieved_cases_preserve_required_fields(monkeypatch):
    _mock_llm_layer(monkeypatch, category="configurator")
    result = main.triage_inquiry("The online configurator will not save my build.", top_k=3, confidence_threshold=0.5)
    required = {"inquiry_text", "category", "priority", "similarity"}
    for case in result["retrieved_past_cases"]:
        assert required <= set(case.keys())


@pytest.mark.parametrize("bad_query", ["", "   "])
def test_triage_inquiry_rejects_invalid_query(bad_query):
    with pytest.raises(ValueError):
        main.triage_inquiry(bad_query, top_k=3, confidence_threshold=0.5)


@pytest.mark.parametrize("bad_top_k", [0, -1, 2.5])
def test_triage_inquiry_rejects_invalid_top_k(bad_top_k):
    with pytest.raises(ValueError):
        main.triage_inquiry("some query", top_k=bad_top_k, confidence_threshold=0.5)


@pytest.mark.parametrize("bad_threshold", [-0.1, 1.1])
def test_triage_inquiry_rejects_invalid_threshold(bad_threshold):
    with pytest.raises(ValueError):
        main.triage_inquiry("some query", top_k=3, confidence_threshold=bad_threshold)


# ---------------------------------------------------------------------------
# Real end-to-end integration smoke test (the one deliberate real-Ollama run)
# ---------------------------------------------------------------------------


def test_real_end_to_end_smoke():
    result = main.triage_inquiry("My brakes are making a grinding noise.", top_k=5, confidence_threshold=0.5)

    assert result["query"] == "My brakes are making a grinding noise."
    assert result["category"] in main.get_routing_map()
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["resolution_notes"], str)
    assert result["resolution_notes"].strip() != ""
    assert isinstance(result["escalated"], bool)
    assert len(result["retrieved_past_cases"]) == 5


@pytest.mark.parametrize("error_type", [httpx.ReadError, httpx.RemoteProtocolError])
@pytest.mark.parametrize("on_retry", [False, True])
def test_transport_failure_stops_resolution_generation(error_type, on_retry):
    error = error_type("transport failed")
    fake = FakeChat(([""] if on_retry else []) + [error])
    with pytest.raises(main.InfrastructureError) as caught:
        main.generate_resolution_note("Brake noise", "service", "high", SAMPLE_RETRIEVED, chat=fake)
    assert caught.value.__cause__ is error
    assert fake.calls == (2 if on_retry else 1)


@pytest.mark.parametrize("on_retry", [False, True])
def test_ollama_request_error_stops_resolution_generation(on_retry):
    error = main.OllamaResponseError("request rejected", status_code=400)
    fake = FakeChat(([""] if on_retry else []) + [error])
    with pytest.raises(main.OllamaResponseError) as caught:
        main.generate_resolution_note("Brake noise", "service", "high", SAMPLE_RETRIEVED, chat=fake)
    assert caught.value is error
    assert fake.calls == (2 if on_retry else 1)
