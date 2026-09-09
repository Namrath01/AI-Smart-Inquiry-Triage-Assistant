"""Data foundation: past_cases.csv loading/validation, deterministic routing,
and the Chroma-backed historical-case retrieval used by the (later) triage
pipeline.

Scope note: this module only builds the deterministic data layer. It does not
call any chat model and does not implement classification, priority,
confidence, or resolution notes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from src.taxonomy import get_category_names

REPO_ROOT = Path(__file__).resolve().parent.parent
PAST_CASES_PATH = REPO_ROOT / "data" / "past_cases.csv"
CHROMA_DIR = REPO_ROOT / ".chroma"
COLLECTION_NAME = "past_cases"
EMBEDDING_MODEL_NAME = "nomic-embed-text"
VALID_PRIORITIES = ("low", "medium", "high")
REQUIRED_COLUMNS = ("case_id", "inquiry_text", "category", "priority", "routed_queue")

# Bump this if the logic that turns a CSV row into a Chroma document/metadata
# changes, so a stale collection built under the old logic is not silently reused.
INGESTION_SCHEMA_VERSION = "v1"

FINGERPRINT_FILENAME = "corpus_fingerprint.json"


class DataValidationError(ValueError):
    """Raised when data/past_cases.csv is missing, malformed, or invalid.

    This is a genuine data/infrastructure failure, not a triage-uncertainty
    signal -- callers must let it surface as an application error rather than
    converting it into an escalated triage result.
    """


def load_past_cases() -> pd.DataFrame:
    """Load and validate data/past_cases.csv. Returns a fresh, validated copy."""
    if not PAST_CASES_PATH.exists():
        raise DataValidationError(f"past_cases.csv not found at {PAST_CASES_PATH}")

    df = pd.read_csv(PAST_CASES_PATH)

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise DataValidationError(f"past_cases.csv is missing column(s): {missing_columns}")

    if len(df) == 0:
        raise DataValidationError("past_cases.csv contains no rows")

    for col in REQUIRED_COLUMNS:
        if df[col].isna().any():
            raise DataValidationError(f"past_cases.csv has missing values in column '{col}'")

    if df["case_id"].duplicated().any():
        dupes = df.loc[df["case_id"].duplicated(), "case_id"].tolist()
        raise DataValidationError(f"past_cases.csv has duplicate case_id values: {dupes}")

    if (df["inquiry_text"].astype(str).str.strip() == "").any():
        raise DataValidationError("past_cases.csv has empty inquiry_text value(s)")

    valid_categories = set(get_category_names())
    unknown_categories = set(df["category"].unique()) - valid_categories
    if unknown_categories:
        raise DataValidationError(
            f"past_cases.csv contains categories not in taxonomy.json: {unknown_categories}"
        )

    invalid_priorities = set(df["priority"].unique()) - set(VALID_PRIORITIES)
    if invalid_priorities:
        raise DataValidationError(
            f"past_cases.csv contains invalid priority value(s): {invalid_priorities} "
            f"(must be one of {VALID_PRIORITIES})"
        )

    return df.copy()


def get_routing_map() -> dict[str, str]:
    """Derive category -> routed_queue directly from past_cases.csv.

    Raises DataValidationError if any category maps to more than one queue --
    later deterministic routing depends on this being a strict 1:1 mapping.
    """
    df = load_past_cases()
    grouped = df.groupby("category")["routed_queue"].unique()

    conflicts = {cat: list(queues) for cat, queues in grouped.items() if len(queues) > 1}
    if conflicts:
        raise DataValidationError(
            f"category -> routed_queue mapping is not 1:1, conflicts: {conflicts}"
        )

    return {cat: queues[0] for cat, queues in grouped.items()}


def _corpus_fingerprint(df: pd.DataFrame) -> str:
    """Deterministic hash over the validated rows + embedding config.

    Any change to the underlying CSV content, or to the embedding model /
    ingestion schema version, changes the fingerprint and forces a rebuild --
    this is what prevents stale embeddings, not row-count alone.
    """
    rows = df[list(REQUIRED_COLUMNS)].sort_values("case_id").to_dict(orient="records")
    payload = {
        "rows": rows,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "schema_version": INGESTION_SCHEMA_VERSION,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _fingerprint_path() -> Path:
    return CHROMA_DIR / FINGERPRINT_FILENAME


def _read_stored_fingerprint() -> dict[str, Any] | None:
    path = _fingerprint_path()
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_fingerprint(fingerprint: str, row_count: int) -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    with _fingerprint_path().open("w", encoding="utf-8") as f:
        json.dump(
            {
                "fingerprint": fingerprint,
                "row_count": row_count,
                "embedding_model": EMBEDDING_MODEL_NAME,
                "schema_version": INGESTION_SCHEMA_VERSION,
                "collection_name": COLLECTION_NAME,
            },
            f,
            indent=2,
        )


def _build_documents(df: pd.DataFrame) -> list[Document]:
    documents = []
    for row in df.itertuples(index=False):
        documents.append(
            Document(
                page_content=row.inquiry_text,
                metadata={
                    "case_id": row.case_id,
                    "category": row.category,
                    "priority": row.priority,
                    "routed_queue": row.routed_queue,
                },
            )
        )
    return documents


def get_or_build_vectorstore(force_rebuild: bool = False) -> Chroma:
    """Return a Chroma vector store over past_cases.csv, rebuilding only when
    the corpus fingerprint (data + embedding config) no longer matches what
    is currently persisted on disk.
    """
    df = load_past_cases()
    fingerprint = _corpus_fingerprint(df)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

    stored = None if force_rebuild else _read_stored_fingerprint()
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    if stored is not None and stored.get("fingerprint") == fingerprint:
        if vectorstore._collection.count() == len(df):
            return vectorstore
        # Persisted collection doesn't actually match the fingerprint's row
        # count (e.g. partial/corrupted write) -- fall through and rebuild.

    # Recreate only this collection. Deleting the persistence directory can
    # fail on Windows while Chroma holds its SQLite database open.
    vectorstore.delete_collection()

    documents = _build_documents(df)
    vectorstore = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )
    _write_fingerprint(fingerprint, len(df))
    return vectorstore


def retrieve_similar_cases(query: str, top_k: int) -> list[dict[str, Any]]:
    """Retrieve the top_k most similar historical cases for `query`.

    Similarity convention: with the collection's distance metric explicitly
    set to cosine (hnsw:space=cosine), Chroma's raw "distance" equals
    (1 - cosine_similarity) -- verified empirically against hand-computed
    cosine similarity on nomic-embed-text vectors. similarity is therefore
    computed as (1 - distance), clamped to [0, 1]. Clipping absorbs
    floating-point noise and maps negative cosine similarities to zero.
    1.0 indicates identical vector direction; 0.0 includes orthogonal or
    negatively aligned vectors.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    vectorstore = get_or_build_vectorstore()
    corpus_size = vectorstore._collection.count()
    k = min(top_k, corpus_size)
    if k == 0:
        return []

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
