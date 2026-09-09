"""Leakage-safe offline evaluation of the triage pipeline.

Stratified 5-fold cross-validation over data/past_cases.csv. For each fold,
a temporary Chroma index is built from ONLY the training rows, so a held-out
case can never retrieve itself (or any other held-out case in that fold).
Classification is computed once per held-out inquiry and reused across the
tested Top-K values (K is irrelevant to classification). Resolution notes
are never generated here -- they are irrelevant to the metrics computed.

Run with (from the repository root, with the venv active):

    python -m evaluation.evaluate

Writes evaluation/results.json (metrics) and evaluation/predictions.csv
(one row per held-out case). Does not touch the production .chroma/ store --
each fold's index lives in its own temporary directory, deleted after that
fold's cases are evaluated.

This script is evaluation-only tooling. It does not change, and must not be
imported by, the production pipeline in src/.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from sklearn.model_selection import StratifiedKFold

from src.ingestion import EMBEDDING_MODEL_NAME, _build_documents, load_past_cases
from src.main import (
    CHAT_MODEL_NAME,
    category_agreement,
    category_margin,
    classify_inquiry,
    compute_confidence,
    majority_vote_priority,
    route_category,
    retrieval_strength,
    similarity_weighted_priority,
)
from src.taxonomy import get_category_names

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "evaluation" / "results.json"
PREDICTIONS_CSV_PATH = REPO_ROOT / "evaluation" / "predictions.csv"

N_SPLITS = 5
RANDOM_SEED = 42
K_VALUES = (3, 5)
REFERENCE_K = 5  # used for confidence/threshold analysis and strategy selection
THRESHOLDS = (0.40, 0.50, 0.60, 0.70, 0.80)

# Bump this if the classification prompt/schema in src/main.py changes, so a
# stale cache entry is never reused across an incompatible classifier version.
CLASSIFIER_VERSION = "v1"

# Convention, not a statistical test: how many percentage points count as a
# "material" difference for strategy-selection purposes in this evaluation.
MATERIAL_DIFF = 0.03
RELIABILITY_BAR = 0.90


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def verify_stratification_feasible(df: pd.DataFrame, n_splits: int) -> pd.Series:
    counts = df["category"].value_counts()
    if counts.min() < n_splits:
        raise ValueError(
            f"category {counts.idxmin()!r} has only {counts.min()} rows, "
            f"insufficient for {n_splits}-fold stratification"
        )
    return counts


def make_folds(df: pd.DataFrame, n_splits: int = N_SPLITS, seed: int = RANDOM_SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(df.index.values, df["category"].values))


# ---------------------------------------------------------------------------
# Classification cache (K-independent)
# ---------------------------------------------------------------------------


def classification_cache_key(query: str) -> tuple[str, str, str]:
    normalized = " ".join(query.split())
    return (normalized, CHAT_MODEL_NAME, CLASSIFIER_VERSION)


def classify_with_cache(query: str, cache: dict) -> tuple[str, str, bool]:
    key = classification_cache_key(query)
    if key in cache:
        return cache[key]
    result = classify_inquiry(query)
    cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Per-fold temporary Chroma index (never the production .chroma/ store)
# ---------------------------------------------------------------------------


def build_fold_vectorstore(train_df: pd.DataFrame, persist_dir: str) -> Chroma:
    documents = _build_documents(train_df)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    return Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="eval_fold",
        collection_metadata={"hnsw:space": "cosine"},
    )


def close_vectorstore(vectorstore: Chroma | None) -> None:
    """Best-effort release of the underlying chromadb client's SQLite file
    handle. Without this, shutil.rmtree on the temp fold directory silently
    fails on Windows (file still in use) even with ignore_errors=True,
    leaving orphaned temp directories behind after every run.
    """
    if vectorstore is None:
        return
    try:
        vectorstore._client.close()
    except Exception:
        pass


def rmtree_with_retry(path: str, attempts: int = 5, delay_seconds: float = 0.5) -> None:
    """shutil.rmtree with a short retry-with-delay loop. close_vectorstore()
    releases the SQLite handle in the common case, but Windows can hold the
    file mapping open briefly longer (e.g. a background telemetry thread);
    a few short retries clears the remaining race without resorting to
    anything heavier.
    """
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except OSError:
            if attempt == attempts - 1:
                return  # give up silently; orphaned temp dirs are a disk-space
                # nuisance only, never a correctness issue (production .chroma/
                # is never touched by this path)
            time.sleep(delay_seconds)


def retrieve_from_vectorstore(vectorstore: Chroma, query: str, k: int) -> list[dict]:
    """Same distance -> similarity conversion as src/ingestion.py's
    retrieve_similar_cases (cosine distance, clamped to [0,1])."""
    raw_results = vectorstore.similarity_search_with_score(query, k=k)
    results = []
    for doc, distance in raw_results:
        similarity = max(0.0, min(1.0, 1.0 - distance))
        results.append(
            {
                "case_id": doc.metadata["case_id"],
                "inquiry_text": doc.page_content,
                "category": doc.metadata["category"],
                "priority": doc.metadata["priority"],
                "routed_queue": doc.metadata["routed_queue"],
                "similarity": similarity,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------


def evaluate_case(row, fold_idx: int, vectorstore: Chroma, classify_cache: dict) -> dict[str, Any]:
    query = row["inquiry_text"]
    category, source, failed = classify_with_cache(query, classify_cache)

    predicted_routed_queue = route_category(category)

    max_k = max(K_VALUES)
    retrieved_max = retrieve_from_vectorstore(vectorstore, query, k=max_k)

    # Mandatory leakage guard: the held-out case must never retrieve itself
    # (or, since the fold index excludes ALL held-out rows, any other
    # held-out case either) from a training-only index.
    retrieved_ids = {r["case_id"] for r in retrieved_max}
    if row["case_id"] in retrieved_ids:
        raise AssertionError(
            f"leakage detected: held-out case {row['case_id']} was retrieved "
            f"from its own fold's training-only index"
        )

    per_k: dict[int, dict] = {}
    for k in K_VALUES:
        subset = retrieved_max[:k]
        rs = retrieval_strength(subset)
        ca = category_agreement(subset, category)
        cm = category_margin(subset, category)
        per_k[k] = {
            "retrieved_case_ids": [r["case_id"] for r in subset],
            "majority_priority": majority_vote_priority(subset),
            "weighted_priority": similarity_weighted_priority(subset),
            "retrieval_strength": rs,
            "category_agreement": ca,
            "category_margin": cm,
            "confidence_retrieval_strength": compute_confidence(
                rs, ca, cm, strategy="retrieval_strength"
            ),
            "confidence_evidence_mean": compute_confidence(rs, ca, cm, strategy="evidence_mean"),
        }

    return {
        "case_id": row["case_id"],
        "fold": fold_idx,
        "true_category": row["category"],
        "true_priority": row["priority"],
        "true_routed_queue": row["routed_queue"],
        "predicted_category": category,
        "classification_source": source,
        "classification_failed": failed,
        "predicted_routed_queue": predicted_routed_queue,
        "k": per_k,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_evaluation(verbose: bool = True) -> list[dict]:
    df = load_past_cases()
    verify_stratification_feasible(df, N_SPLITS)
    folds = make_folds(df)

    classify_cache: dict = {}
    records: list[dict] = []

    for fold_idx, (train_pos, test_pos) in enumerate(folds):
        train_df = df.iloc[train_pos].reset_index(drop=True)
        test_df = df.iloc[test_pos].reset_index(drop=True)

        assert not (set(train_df["case_id"]) & set(test_df["case_id"])), (
            f"fold {fold_idx}: train/test case_id overlap detected"
        )

        temp_dir = tempfile.mkdtemp(prefix=f"triage_eval_fold{fold_idx}_")
        vectorstore = None
        try:
            vectorstore = build_fold_vectorstore(train_df, temp_dir)
            for i, row in test_df.iterrows():
                record = evaluate_case(row, fold_idx, vectorstore, classify_cache)
                records.append(record)
                if verbose:
                    print(
                        f"[fold {fold_idx}] {record['case_id']} "
                        f"true={record['true_category']} pred={record['predicted_category']} "
                        f"({record['classification_source']})"
                    )
        finally:
            close_vectorstore(vectorstore)
            rmtree_with_retry(temp_dir)

    return records


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_classification_metrics(records: list[dict]) -> dict:
    correct = sum(1 for r in records if r["predicted_category"] == r["true_category"])
    total = len(records)

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        confusion[r["true_category"]][r["predicted_category"]] += 1

    per_category = {}
    for cat in get_category_names():
        cat_records = [r for r in records if r["true_category"] == cat]
        n = len(cat_records)
        c = sum(1 for r in cat_records if r["predicted_category"] == cat)
        per_category[cat] = {"correct": c, "total": n, "accuracy": (c / n if n else None)}

    source_counts = dict(Counter(r["classification_source"] for r in records))
    for source in ("llm", "llm_retry", "taxonomy_fallback", "safe_default"):
        source_counts.setdefault(source, 0)

    return {
        "overall": {"correct": correct, "total": total, "accuracy": correct / total},
        "per_category": per_category,
        "confusion_matrix": {t: dict(preds) for t, preds in confusion.items()},
        "source_counts": source_counts,
    }


def compute_priority_metrics(records: list[dict], k: int) -> dict:
    rows = [
        {
            "true": r["true_priority"],
            "majority": r["k"][k]["majority_priority"],
            "weighted": r["k"][k]["weighted_priority"],
        }
        for r in records
    ]
    total = len(rows)

    def accuracy(method: str) -> dict:
        correct = sum(1 for row in rows if row[method] == row["true"])
        return {"correct": correct, "total": total, "accuracy": correct / total if total else None}

    def by_true_priority(method: str) -> dict:
        out = {}
        for p in ("low", "medium", "high"):
            subset = [row for row in rows if row["true"] == p]
            n = len(subset)
            c = sum(1 for row in subset if row[method] == p)
            out[p] = {"correct": c, "total": n, "accuracy": (c / n if n else None)}
        return out

    def accuracy_on_subset(method: str, subset: list[dict]) -> dict:
        n = len(subset)
        if n == 0:
            return {"correct": 0, "total": 0, "accuracy": None}
        c = sum(1 for row in subset if row[method] == row["true"])
        return {"correct": c, "total": n, "accuracy": c / n}

    majority_metrics = accuracy("majority")
    weighted_metrics = accuracy("weighted")
    majority_metrics["by_true_priority"] = by_true_priority("majority")
    weighted_metrics["by_true_priority"] = by_true_priority("weighted")

    disagreements = [row for row in rows if row["majority"] != row["weighted"]]
    better = "weighted" if (weighted_metrics["accuracy"] or 0) > (majority_metrics["accuracy"] or 0) else "majority"

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        confusion[row["true"]][row[better]] += 1

    return {
        "majority": majority_metrics,
        "weighted": weighted_metrics,
        "disagreement": {
            "count": len(disagreements),
            "total": total,
            "rate": len(disagreements) / total if total else None,
            "majority_accuracy_on_disagreements": accuracy_on_subset("majority", disagreements),
            "weighted_accuracy_on_disagreements": accuracy_on_subset("weighted", disagreements),
        },
        "better_candidate": better,
        "confusion_matrix_better_candidate": {t: dict(p) for t, p in confusion.items()},
    }


def compute_routing_metrics(records: list[dict]) -> dict:
    total = len(records)
    routing_correct = sum(1 for r in records if r["predicted_routed_queue"] == r["true_routed_queue"])
    category_correct = sum(1 for r in records if r["predicted_category"] == r["true_category"])
    return {
        "correct": routing_correct,
        "total": total,
        "accuracy": routing_correct / total if total else None,
        "category_accuracy_for_comparison": category_correct / total if total else None,
        "matches_category_accuracy": routing_correct == category_correct,
    }


def select_priority_strategy(priority_k_metrics: dict) -> str:
    """Prefer materially better held-out performance, paying particular
    attention to high-priority accuracy; otherwise prefer the simpler
    majority-vote baseline. MATERIAL_DIFF is a chosen convention (see module
    header), not a statistical significance threshold.
    """
    maj_acc = priority_k_metrics["majority"]["accuracy"] or 0.0
    wei_acc = priority_k_metrics["weighted"]["accuracy"] or 0.0
    maj_high = priority_k_metrics["majority"]["by_true_priority"]["high"]["accuracy"] or 0.0
    wei_high = priority_k_metrics["weighted"]["by_true_priority"]["high"]["accuracy"] or 0.0

    overall_diff = wei_acc - maj_acc
    high_diff = wei_high - maj_high

    if overall_diff > MATERIAL_DIFF:
        return "weighted"
    if abs(overall_diff) <= MATERIAL_DIFF and high_diff > MATERIAL_DIFF:
        return "weighted"
    return "majority"


def compute_threshold_table(
    records: list[dict], k: int, confidence_field: str, priority_field: str
) -> dict:
    total = len(records)
    table = {}
    for t in THRESHOLDS:
        auto, escalated = [], []
        for r in records:
            if r["classification_failed"]:
                escalated.append(r)
                continue
            conf = r["k"][k][confidence_field]
            (auto if conf >= t else escalated).append(r)

        n_auto = len(auto)
        n_escalated = len(escalated)

        if n_auto == 0:
            reliability = None
            reliability_correct = 0
        else:
            reliability_correct = sum(
                1
                for r in auto
                if r["predicted_category"] == r["true_category"]
                and r["k"][k][priority_field] == r["true_priority"]
                and r["predicted_routed_queue"] == r["true_routed_queue"]
            )
            reliability = reliability_correct / n_auto

        table[f"{t:.2f}"] = {
            "threshold": t,
            "coverage": n_auto / total if total else None,
            "escalation_rate": n_escalated / total if total else None,
            "n_auto": n_auto,
            "n_escalated": n_escalated,
            "reliability": reliability,
            "reliability_correct": reliability_correct,
        }
    return table


def select_confidence_strategy(rs_table: dict, em_table: dict) -> str:
    """Compare reliability at each threshold where both strategies actually
    auto-triage at least one case; select evidence_mean only if it shows a
    materially better average reliability. Keeps the simpler baseline
    otherwise.
    """
    diffs = []
    for key in rs_table:
        rs_rel = rs_table[key]["reliability"]
        em_rel = em_table[key]["reliability"]
        if rs_rel is None or em_rel is None:
            continue
        diffs.append(em_rel - rs_rel)
    if not diffs:
        return "retrieval_strength"
    avg_diff = sum(diffs) / len(diffs)
    return "evidence_mean" if avg_diff > MATERIAL_DIFF else "retrieval_strength"


def select_default_threshold(table: dict) -> float:
    """Among thresholds achieving RELIABILITY_BAR reliability, pick the one
    with the broadest coverage (lowest threshold). If none reach the bar,
    fall back to the threshold with the highest observed reliability,
    breaking ties toward broader coverage. If no threshold ever auto-triages
    a single case, fall back to the original 0.5 default.
    """
    qualifying = [
        v for v in table.values() if v["reliability"] is not None and v["reliability"] >= RELIABILITY_BAR
    ]
    if qualifying:
        return min(qualifying, key=lambda v: v["threshold"])["threshold"]

    scored = [v for v in table.values() if v["reliability"] is not None]
    if not scored:
        return 0.50
    best = max(scored, key=lambda v: (v["reliability"], -v["threshold"]))
    return best["threshold"]


# ---------------------------------------------------------------------------
# Top-level report assembly + persistence
# ---------------------------------------------------------------------------


def build_report(records: list[dict]) -> dict:
    classification = compute_classification_metrics(records)
    priority_by_k = {k: compute_priority_metrics(records, k) for k in K_VALUES}
    routing = compute_routing_metrics(records)

    selected_priority = select_priority_strategy(priority_by_k[REFERENCE_K])
    priority_field = "majority_priority" if selected_priority == "majority" else "weighted_priority"
    priority_key_in_row = "majority" if selected_priority == "majority" else "weighted"

    rs_table = compute_threshold_table(
        records, REFERENCE_K, "confidence_retrieval_strength", f"{priority_key_in_row}_priority"
    )
    em_table = compute_threshold_table(
        records, REFERENCE_K, "confidence_evidence_mean", f"{priority_key_in_row}_priority"
    )
    selected_confidence = select_confidence_strategy(rs_table, em_table)
    selected_table = rs_table if selected_confidence == "retrieval_strength" else em_table
    default_threshold = select_default_threshold(selected_table)

    return {
        "meta": {
            "n_cases": len(records),
            "n_folds": N_SPLITS,
            "seed": RANDOM_SEED,
            "k_values": list(K_VALUES),
            "reference_k": REFERENCE_K,
            "thresholds": list(THRESHOLDS),
            "chat_model": CHAT_MODEL_NAME,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "classifier_version": CLASSIFIER_VERSION,
        },
        "classification": classification,
        "priority": {str(k): priority_by_k[k] for k in K_VALUES},
        "routing": routing,
        "confidence": {
            f"k{REFERENCE_K}": {
                "retrieval_strength": rs_table,
                "evidence_mean": em_table,
            }
        },
        "selection": {
            "priority_strategy": selected_priority,
            "priority_reference_k": REFERENCE_K,
            "confidence_strategy": selected_confidence,
            "default_threshold": default_threshold,
        },
    }


def save_report(report: dict, records: list[dict]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    rows = []
    for r in records:
        row = {
            "case_id": r["case_id"],
            "fold": r["fold"],
            "true_category": r["true_category"],
            "true_priority": r["true_priority"],
            "true_routed_queue": r["true_routed_queue"],
            "predicted_category": r["predicted_category"],
            "classification_source": r["classification_source"],
            "classification_failed": r["classification_failed"],
            "predicted_routed_queue": r["predicted_routed_queue"],
        }
        for k in K_VALUES:
            kd = r["k"][k]
            row[f"k{k}_majority_priority"] = kd["majority_priority"]
            row[f"k{k}_weighted_priority"] = kd["weighted_priority"]
            row[f"k{k}_retrieval_strength"] = kd["retrieval_strength"]
            row[f"k{k}_category_agreement"] = kd["category_agreement"]
            row[f"k{k}_category_margin"] = kd["category_margin"]
            row[f"k{k}_confidence_retrieval_strength"] = kd["confidence_retrieval_strength"]
            row[f"k{k}_confidence_evidence_mean"] = kd["confidence_evidence_mean"]
        rows.append(row)

    pd.DataFrame(rows).to_csv(PREDICTIONS_CSV_PATH, index=False)


def main() -> None:
    records = run_evaluation(verbose=True)
    report = build_report(records)
    save_report(report, records)
    print(f"\nWrote {RESULTS_PATH}")
    print(f"Wrote {PREDICTIONS_CSV_PATH}")


if __name__ == "__main__":
    main()
