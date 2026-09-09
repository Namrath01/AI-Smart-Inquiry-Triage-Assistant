"""Tests for the leakage-safe offline evaluation harness (evaluation/evaluate.py).

Metric/selection-logic tests use synthetic fixtures and touch neither Ollama
nor Chroma. A small number of tests build a tiny real Chroma index (a
handful of rows) to prove the leakage-prevention mechanism and K-limiting
actually work against the real integration, not just in theory.
"""

from __future__ import annotations

import json
import tempfile

import pandas as pd
import pytest

from evaluation import evaluate as ev
from src.ingestion import load_past_cases


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------


def test_train_test_folds_do_not_overlap_by_case_id():
    df = load_past_cases()
    folds = ev.make_folds(df)
    for train_pos, test_pos in folds:
        train_ids = set(df.iloc[train_pos]["case_id"])
        test_ids = set(df.iloc[test_pos]["case_id"])
        assert train_ids & test_ids == set()


def test_each_case_appears_in_test_exactly_once_across_folds():
    df = load_past_cases()
    folds = ev.make_folds(df)
    all_case_ids = list(df["case_id"])

    test_id_counts: dict[str, int] = {}
    for _, test_pos in folds:
        for case_id in df.iloc[test_pos]["case_id"]:
            test_id_counts[case_id] = test_id_counts.get(case_id, 0) + 1

    assert set(test_id_counts.keys()) == set(all_case_ids)
    assert all(count == 1 for count in test_id_counts.values())


def test_stratification_feasibility_check_passes_for_real_dataset():
    df = load_past_cases()
    counts = ev.verify_stratification_feasible(df, 5)
    assert counts.min() >= 5


def test_stratification_feasibility_check_rejects_insufficient_category():
    tiny_df = pd.DataFrame(
        {
            "case_id": ["A", "B", "C"],
            "inquiry_text": ["x", "y", "z"],
            "category": ["service", "service", "billing"],
            "priority": ["low", "low", "low"],
            "routed_queue": ["q1", "q1", "q2"],
        }
    )
    with pytest.raises(ValueError):
        ev.verify_stratification_feasible(tiny_df, 5)


# ---------------------------------------------------------------------------
# Real-index leakage prevention + K limiting (small synthetic fixture)
# ---------------------------------------------------------------------------


_TINY_TRAIN_ROWS = [
    {"case_id": "T1", "inquiry_text": "My brakes are grinding loudly.", "category": "service", "priority": "high", "routed_queue": "Service Scheduling Team"},
    {"case_id": "T2", "inquiry_text": "I need to book an oil change.", "category": "service", "priority": "low", "routed_queue": "Service Scheduling Team"},
    {"case_id": "T3", "inquiry_text": "I was charged twice on my invoice.", "category": "billing", "priority": "high", "routed_queue": "Billing & Payments Team"},
    {"case_id": "T4", "inquiry_text": "The configurator crashed while saving.", "category": "configurator", "priority": "medium", "routed_queue": "Digital Product / Configurator Support"},
]
_HELD_OUT_CASE_ID = "HELD-OUT-1"  # deliberately never added to the tiny index


@pytest.fixture(scope="module")
def tiny_vectorstore():
    train_df = pd.DataFrame(_TINY_TRAIN_ROWS)
    temp_dir = tempfile.mkdtemp(prefix="triage_eval_test_")
    vs = ev.build_fold_vectorstore(train_df, temp_dir)
    try:
        yield vs
    finally:
        ev.close_vectorstore(vs)
        ev.rmtree_with_retry(temp_dir)


def test_temporary_index_excludes_held_out_case_id(tiny_vectorstore):
    metadatas = tiny_vectorstore.get()["metadatas"]
    indexed_case_ids = {m["case_id"] for m in metadatas}
    assert _HELD_OUT_CASE_ID not in indexed_case_ids
    assert indexed_case_ids == {r["case_id"] for r in _TINY_TRAIN_ROWS}


def test_retrieval_never_returns_the_held_out_case(tiny_vectorstore):
    results = ev.retrieve_from_vectorstore(tiny_vectorstore, "My brakes are making noise.", k=4)
    retrieved_ids = {r["case_id"] for r in results}
    assert _HELD_OUT_CASE_ID not in retrieved_ids


def test_k3_returns_at_most_3(tiny_vectorstore):
    results = ev.retrieve_from_vectorstore(tiny_vectorstore, "brake issue", k=3)
    assert len(results) <= 3


def test_k5_returns_at_most_5(tiny_vectorstore):
    # only 4 rows are indexed, so this also exercises the "fewer than K
    # available" clamp, same as production retrieve_similar_cases.
    results = ev.retrieve_from_vectorstore(tiny_vectorstore, "brake issue", k=5)
    assert len(results) <= 5
    assert len(results) == 4


# ---------------------------------------------------------------------------
# Classification caching (K-independent, reused across K)
# ---------------------------------------------------------------------------


def test_classification_is_reused_across_k_values(monkeypatch):
    calls = []

    def fake_classify_inquiry(query, structured_classifier=None):
        calls.append(query)
        return "service", "llm", False

    monkeypatch.setattr(ev, "classify_inquiry", fake_classify_inquiry)

    cache: dict = {}
    result1 = ev.classify_with_cache("My brakes are grinding.", cache)
    result2 = ev.classify_with_cache("My brakes are grinding.", cache)

    assert result1 == result2 == ("service", "llm", False)
    assert len(calls) == 1  # second call served from cache, not re-invoked


def test_classification_cache_key_includes_model_and_version():
    key = ev.classification_cache_key("  My brakes   are grinding.  ")
    normalized_text, model, version = key
    assert normalized_text == "My brakes are grinding."
    assert model == ev.CHAT_MODEL_NAME
    assert version == ev.CLASSIFIER_VERSION


# ---------------------------------------------------------------------------
# Metric calculations on a synthetic fixture (hand-computable)
# ---------------------------------------------------------------------------


def _synthetic_record(case_id, true_category, true_priority, true_queue,
                       predicted_category, source, failed, predicted_queue,
                       k5_majority, k5_weighted, k5_conf_rs, k5_conf_em):
    return {
        "case_id": case_id,
        "fold": 0,
        "true_category": true_category,
        "true_priority": true_priority,
        "true_routed_queue": true_queue,
        "predicted_category": predicted_category,
        "classification_source": source,
        "classification_failed": failed,
        "predicted_routed_queue": predicted_queue,
        "k": {
            5: {
                "retrieved_case_ids": [],
                "majority_priority": k5_majority,
                "weighted_priority": k5_weighted,
                "retrieval_strength": k5_conf_rs,
                "category_agreement": 1.0,
                "category_margin": 1.0,
                "confidence_retrieval_strength": k5_conf_rs,
                "confidence_evidence_mean": k5_conf_em,
            }
        },
    }


SYNTHETIC_RECORDS = [
    _synthetic_record("S1", "service", "high", "Q_service", "service", "llm", False, "Q_service", "high", "high", 0.90, 0.90),
    _synthetic_record("S2", "service", "low", "Q_service", "billing", "llm", False, "Q_billing", "low", "medium", 0.60, 0.55),
    _synthetic_record("S3", "billing", "high", "Q_billing", "billing", "llm_retry", False, "Q_billing", "high", "high", 0.85, 0.80),
    _synthetic_record("S4", "other", "low", "Q_other", "other", "taxonomy_fallback", True, "Q_other", "low", "low", 0.20, 0.20),
]


def test_classification_metrics_on_synthetic_fixture():
    result = ev.compute_classification_metrics(SYNTHETIC_RECORDS)
    assert result["overall"]["correct"] == 3
    assert result["overall"]["total"] == 4
    assert result["overall"]["accuracy"] == pytest.approx(0.75)
    assert result["per_category"]["service"]["correct"] == 1
    assert result["per_category"]["service"]["total"] == 2
    assert result["per_category"]["billing"]["accuracy"] == pytest.approx(1.0)
    assert result["confusion_matrix"]["service"]["billing"] == 1
    assert result["source_counts"]["llm"] == 2
    assert result["source_counts"]["llm_retry"] == 1
    assert result["source_counts"]["taxonomy_fallback"] == 1
    assert result["source_counts"]["safe_default"] == 0


def test_priority_metrics_on_synthetic_fixture():
    result = ev.compute_priority_metrics(SYNTHETIC_RECORDS, k=5)
    # majority correct: S1(high==high) T, S2(low==low) T, S3(high==high) T, S4(low==low) T -> 4/4
    assert result["majority"]["correct"] == 4
    assert result["majority"]["accuracy"] == pytest.approx(1.0)
    # weighted correct: S1 T, S2(medium!=low) F, S3 T, S4 T -> 3/4
    assert result["weighted"]["correct"] == 3
    assert result["weighted"]["accuracy"] == pytest.approx(0.75)
    # disagreement only on S2
    assert result["disagreement"]["count"] == 1
    assert result["disagreement"]["rate"] == pytest.approx(0.25)
    assert result["disagreement"]["majority_accuracy_on_disagreements"]["accuracy"] == pytest.approx(1.0)
    assert result["disagreement"]["weighted_accuracy_on_disagreements"]["accuracy"] == pytest.approx(0.0)
    assert result["better_candidate"] == "majority"


def test_routing_metrics_on_synthetic_fixture():
    result = ev.compute_routing_metrics(SYNTHETIC_RECORDS)
    # routed correctly whenever predicted_category == true_category (routing is
    # deterministic from category in this fixture's Q_<category> convention)
    assert result["correct"] == 3
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["category_accuracy_for_comparison"] == pytest.approx(0.75)
    assert result["matches_category_accuracy"] is True


def test_select_priority_strategy_prefers_majority_when_not_materially_better():
    priority_metrics = ev.compute_priority_metrics(SYNTHETIC_RECORDS, k=5)
    # majority (1.0) beats weighted (0.75) here by more than MATERIAL_DIFF,
    # so majority must be selected.
    assert ev.select_priority_strategy(priority_metrics) == "majority"


# ---------------------------------------------------------------------------
# Threshold table: zero-auto-triage -> N/A, classification_failed -> escalated
# ---------------------------------------------------------------------------


def test_zero_auto_triage_threshold_produces_na_reliability():
    low_confidence_records = [
        _synthetic_record("Z1", "service", "high", "Q", "service", "llm", False, "Q", "high", "high", 0.10, 0.10),
        _synthetic_record("Z2", "billing", "low", "Q", "billing", "llm", False, "Q", "low", "low", 0.15, 0.15),
    ]
    table = ev.compute_threshold_table(
        low_confidence_records, k=5, confidence_field="confidence_retrieval_strength", priority_field="majority_priority"
    )
    # every threshold in THRESHOLDS is >= 0.40, well above both confidences
    for entry in table.values():
        assert entry["n_auto"] == 0
        assert entry["reliability"] is None


def test_classification_failed_always_counted_as_escalated():
    record = _synthetic_record(
        "F1", "other", "low", "Q_other", "other", "safe_default", True, "Q_other",
        "low", "low", k5_conf_rs=1.0, k5_conf_em=1.0,  # deliberately maximal confidence
    )
    table = ev.compute_threshold_table(
        [record], k=5, confidence_field="confidence_retrieval_strength", priority_field="majority_priority"
    )
    for entry in table.values():
        # confidence is 1.0 (>= every threshold) but classification_failed=True
        # must still force escalation, never auto-triage.
        assert entry["n_auto"] == 0
        assert entry["n_escalated"] == 1
        assert entry["reliability"] is None


def test_select_confidence_strategy_and_default_threshold_on_synthetic_table():
    # retrieval_strength: reliable and broad coverage at every threshold.
    rs_table = {
        f"{t:.2f}": {"threshold": t, "reliability": 0.95, "n_auto": 10, "n_escalated": 0, "coverage": 1.0, "escalation_rate": 0.0}
        for t in ev.THRESHOLDS
    }
    # evidence_mean: no material improvement.
    em_table = {
        f"{t:.2f}": {"threshold": t, "reliability": 0.93, "n_auto": 10, "n_escalated": 0, "coverage": 1.0, "escalation_rate": 0.0}
        for t in ev.THRESHOLDS
    }
    assert ev.select_confidence_strategy(rs_table, em_table) == "retrieval_strength"
    assert ev.select_default_threshold(rs_table) == pytest.approx(0.40)  # broadest coverage meeting the bar


# ---------------------------------------------------------------------------
# Production .chroma/ store is untouched by evaluation
# ---------------------------------------------------------------------------


@pytest.fixture
def production_fingerprint_path(tmp_path, monkeypatch):
    from src import ingestion

    monkeypatch.setattr(ev, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ingestion, "CHROMA_DIR", tmp_path / ".chroma")
    monkeypatch.setattr(ingestion, "load_past_cases", lambda: pd.DataFrame(_TINY_TRAIN_ROWS))
    store = ingestion.get_or_build_vectorstore()
    try:
        yield ingestion.CHROMA_DIR / ingestion.FINGERPRINT_FILENAME
    finally:
        ev.close_vectorstore(store)


def test_production_chroma_store_not_modified_by_evaluation(production_fingerprint_path):
    assert production_fingerprint_path.exists()

    before = production_fingerprint_path.read_text(encoding="utf-8")
    before_json = json.loads(before)

    # Build a fold index after snapshotting the independently prepared
    # production-store fingerprint, so no earlier test is a prerequisite.
    train_df = pd.DataFrame(_TINY_TRAIN_ROWS)
    temp_dir = tempfile.mkdtemp(prefix="triage_eval_test_isolation_")
    vs = None
    try:
        vs = ev.build_fold_vectorstore(train_df, temp_dir)
    finally:
        ev.close_vectorstore(vs)
        ev.rmtree_with_retry(temp_dir)

    after = production_fingerprint_path.read_text(encoding="utf-8")
    after_json = json.loads(after)
    assert before_json == after_json
    assert before == after
