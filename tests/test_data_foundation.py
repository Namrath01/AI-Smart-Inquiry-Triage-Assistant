"""Phase 1 tests: taxonomy, routing, and Chroma retrieval data foundation.

Deterministic logic (taxonomy validation, fallback scoring, routing-map
derivation, fingerprinting, input validation) is tested without touching the
embedding backend. A small number of tests exercise real retrieval through
the actual Ollama embedding model + Chroma vector store (per instructions,
this integration path is not mocked away) -- the underlying vector store is
only built once per test run thanks to the corpus-fingerprint cache in
src/ingestion.py, so this keeps embedding calls to: one batch ingestion of
the 300-row corpus, plus one query embedding per retrieval test.
"""

from __future__ import annotations

import pytest

from src import ingestion, taxonomy


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


def test_taxonomy_loads_successfully():
    categories = taxonomy.load_taxonomy()
    assert isinstance(categories, tuple)
    assert len(categories) > 0


def test_taxonomy_categories_match_dataset_categories():
    taxonomy_categories = set(taxonomy.get_category_names())
    dataset_categories = set(ingestion.load_past_cases()["category"].unique())
    assert taxonomy_categories == dataset_categories


def test_taxonomy_required_fields_present():
    for category in taxonomy.load_taxonomy():
        assert category["name"].strip()
        assert category["description"].strip()
        assert isinstance(category["keywords"], list)
        assert len(category["keywords"]) > 0
        assert all(isinstance(k, str) and k.strip() for k in category["keywords"])


# ---------------------------------------------------------------------------
# Deterministic fallback scoring
# ---------------------------------------------------------------------------


def test_fallback_evidence_returns_signals_for_every_category():
    evidence = taxonomy.compute_fallback_evidence(
        "The brake fluid warning appeared and the pedal feels soft."
    )
    assert set(evidence.keys()) == set(taxonomy.get_category_names())
    for signals in evidence.values():
        assert 0.0 <= signals["keyword_score"] <= 1.0
        assert 0.0 <= signals["description_overlap_score"] <= 1.0
        assert isinstance(signals["name_match"], bool)
        assert isinstance(signals["keyword_matches"], list)


def test_fallback_evidence_signals_are_observable_separately():
    evidence = taxonomy.compute_fallback_evidence("I need an oil change appointment.")
    service_evidence = evidence["service"]
    assert "keyword_matches" in service_evidence
    assert "keyword_score" in service_evidence
    assert "name_match" in service_evidence
    assert "description_overlap_score" in service_evidence
    assert "combined_score" not in service_evidence
    # "oil change" is a literal keyword for the service category.
    assert "oil change" in service_evidence["keyword_matches"]


def test_fallback_evidence_keyword_score_reflects_match_count():
    evidence = taxonomy.compute_fallback_evidence("I need an oil change appointment.")
    service = taxonomy.get_category("service")
    expected_score = len(service_matches := [
        kw for kw in service["keywords"] if kw.lower() in "i need an oil change appointment."
    ]) / len(service["keywords"])
    assert evidence["service"]["keyword_score"] == pytest.approx(expected_score)
    assert service_matches == evidence["service"]["keyword_matches"]


def test_fallback_evidence_rejects_empty_query():
    with pytest.raises(ValueError):
        taxonomy.compute_fallback_evidence("   ")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_routing_map_contains_every_category():
    routing_map = ingestion.get_routing_map()
    assert set(routing_map.keys()) == set(taxonomy.get_category_names())


def test_every_category_maps_to_exactly_one_queue():
    df = ingestion.load_past_cases()
    grouped = df.groupby("category")["routed_queue"].unique()
    for category, queues in grouped.items():
        assert len(queues) == 1, f"{category} maps to multiple queues: {queues}"


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------


def test_dataset_priorities_are_valid():
    df = ingestion.load_past_cases()
    assert set(df["priority"].unique()) <= set(ingestion.VALID_PRIORITIES)


# ---------------------------------------------------------------------------
# Corpus fingerprint / idempotency
# ---------------------------------------------------------------------------


def test_corpus_fingerprint_is_deterministic():
    df = ingestion.load_past_cases()
    fp1 = ingestion._corpus_fingerprint(df)
    fp2 = ingestion._corpus_fingerprint(df)
    assert fp1 == fp2


def test_corpus_fingerprint_changes_when_data_changes():
    df = ingestion.load_past_cases()
    fp_original = ingestion._corpus_fingerprint(df)

    mutated = df.copy()
    mutated.loc[mutated.index[0], "inquiry_text"] = "a completely different inquiry text"
    fp_mutated = ingestion._corpus_fingerprint(mutated)

    assert fp_original != fp_mutated


# ---------------------------------------------------------------------------
# Retrieval input validation (no embedding calls -- rejected before that point)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_query", ["", "   ", None, 123])
def test_empty_or_invalid_query_is_rejected(bad_query):
    with pytest.raises(ValueError):
        ingestion.retrieve_similar_cases(bad_query, top_k=3)


@pytest.mark.parametrize("bad_top_k", [0, -1, 2.5, "3", True])
def test_invalid_top_k_is_rejected(bad_top_k):
    with pytest.raises(ValueError):
        ingestion.retrieve_similar_cases("brake noise", top_k=bad_top_k)


# ---------------------------------------------------------------------------
# Real retrieval (integration-level; uses the actual Ollama + Chroma backend)
# ---------------------------------------------------------------------------


def test_retrieval_respects_top_k():
    results = ingestion.retrieve_similar_cases("My brakes are making a grinding noise.", top_k=3)
    assert len(results) == 3


def test_retrieval_clamps_top_k_to_corpus_size():
    corpus_size = len(ingestion.load_past_cases())
    results = ingestion.retrieve_similar_cases("My brakes are making a grinding noise.", top_k=corpus_size + 50)
    assert len(results) == corpus_size


def test_retrieved_result_contains_required_metadata():
    results = ingestion.retrieve_similar_cases("I was charged twice for my payment.", top_k=2)
    required_keys = {"case_id", "inquiry_text", "category", "priority", "routed_queue", "similarity"}
    for result in results:
        assert required_keys <= set(result.keys())


def test_similarity_values_are_within_documented_bounds():
    results = ingestion.retrieve_similar_cases("The online configurator will not save my build.", top_k=5)
    for result in results:
        assert 0.0 <= result["similarity"] <= 1.0


def test_near_identical_query_yields_top_match_with_high_similarity():
    # Uses the exact text of a known historical case (CASE-0003) so the
    # top-1 result should be that same case with similarity close to 1.0.
    known_case_text = "The brake fluid warning appeared and the pedal feels soft."
    results = ingestion.retrieve_similar_cases(known_case_text, top_k=1)
    assert results[0]["case_id"] == "CASE-0003"
    assert results[0]["similarity"] > 0.99


# Real Chroma lifecycle with tiny deterministic embeddings: isolates Windows
# file/collection behavior from model-service availability and generation.
@pytest.fixture
def lifecycle_store(tmp_path, monkeypatch):
    import chromadb
    from langchain_core.embeddings import Embeddings

    class LifecycleEmbeddings(Embeddings):
        def __init__(self):
            self.batches = 0
            self.fail = False

        def embed_documents(self, texts):
            self.batches += 1
            if self.fail:
                raise ConnectionError("embedding service unavailable")
            return [[float(len(text)), float(text.count("e")), 1.0] for text in texts]

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    frame = ingestion.load_past_cases().head(2).copy()
    embeddings = LifecycleEmbeddings()
    path = tmp_path / "chroma"
    monkeypatch.setattr(ingestion, "CHROMA_DIR", path)
    monkeypatch.setattr(ingestion, "load_past_cases", lambda: frame.copy())
    monkeypatch.setattr(ingestion, "OllamaEmbeddings", lambda **kwargs: embeddings)
    client = chromadb.PersistentClient(path=str(path))
    try:
        yield frame, embeddings
    finally:
        client.close()


def test_open_store_rebuild_replaces_changed_corpus(lifecycle_store):
    frame, embeddings = lifecycle_store
    opened = ingestion.get_or_build_vectorstore()
    old_text = frame.iloc[0]["inquiry_text"]
    frame.loc[frame.index[0], "inquiry_text"] = "Replacement inquiry for collection rebuild."
    frame.loc[frame.index[0], "priority"] = "high"

    rebuilt = ingestion.get_or_build_vectorstore()
    contents = rebuilt.get()
    assert old_text not in contents["documents"]
    assert frame.iloc[0]["inquiry_text"] in contents["documents"]
    metadata = {row["case_id"]: row for row in contents["metadatas"]}
    assert metadata[frame.iloc[0]["case_id"]]["priority"] == "high"
    assert rebuilt._collection.count() == len(frame)
    assert rebuilt._collection.metadata["hnsw:space"] == "cosine"
    assert ingestion._read_stored_fingerprint()["fingerprint"] == ingestion._corpus_fingerprint(frame)
    assert embeddings.batches == 2
    assert opened is not None  # keep the original handle alive throughout rebuilding


def test_unchanged_corpus_reuses_collection(lifecycle_store):
    frame, embeddings = lifecycle_store
    first = ingestion.get_or_build_vectorstore()
    ids = first.get()["ids"]
    fingerprint = ingestion._fingerprint_path().read_bytes()
    embeddings.fail = True  # reuse must not attempt ingestion

    reused = ingestion.get_or_build_vectorstore()
    assert reused.get()["ids"] == ids
    assert reused._collection.count() == len(frame)
    assert ingestion._fingerprint_path().read_bytes() == fingerprint
    assert embeddings.batches == 1


@pytest.mark.parametrize("existing_store", [False, True])
def test_failed_ingestion_does_not_complete_fingerprint(lifecycle_store, existing_store):
    frame, embeddings = lifecycle_store
    if existing_store:
        opened = ingestion.get_or_build_vectorstore()
        previous = ingestion._fingerprint_path().read_bytes()
        frame.loc[frame.index[0], "inquiry_text"] = "Changed inquiry awaiting successful ingestion."
    else:
        previous = None
    embeddings.fail = True

    with pytest.raises(ConnectionError, match="embedding service unavailable"):
        ingestion.get_or_build_vectorstore()
    if previous is None:
        assert not ingestion._fingerprint_path().exists()
    else:
        assert ingestion._fingerprint_path().read_bytes() == previous
        assert ingestion._read_stored_fingerprint()["fingerprint"] != ingestion._corpus_fingerprint(frame)

    embeddings.fail = False
    recovered = ingestion.get_or_build_vectorstore()
    assert recovered._collection.count() == len(frame)
    assert set(recovered.get()["documents"]) == set(frame["inquiry_text"])
    assert ingestion._read_stored_fingerprint()["fingerprint"] == ingestion._corpus_fingerprint(frame)
