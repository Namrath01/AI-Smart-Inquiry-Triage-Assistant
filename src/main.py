"""Complete triage pipeline: classification + Top-K-derived priority +
deterministic routing + resolution-note generation + confidence/escalation
finalization, wired as a LangGraph workflow, exposed via `triage_inquiry`.

Two compiled graphs are kept: `build_partial_graph()` (Phase 2: classify ->
retrieve -> determine_priority -> route) is retained as-is for the existing
Phase 2 tests, and `build_complete_graph()` (Phase 3) extends it with
generate_resolution -> finalize_confidence_and_escalation. `triage_inquiry`
uses the complete graph.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Callable, Literal, Optional, TypedDict

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from ollama import ResponseError as OllamaResponseError
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph

from src.ingestion import get_routing_map, retrieve_similar_cases
from src.taxonomy import (
    compute_fallback_evidence,
    format_taxonomy_for_prompt,
    get_category_names,
)

CHAT_MODEL_NAME = "qwen2.5:1.5b-instruct"

# Exceptions that always mean the transport/backend itself is broken (never
# treated as an ordinary classification/generation failure -- no retry, no
# fallback; they propagate as genuine system errors).
_TRANSPORT_INFRASTRUCTURE_EXCEPTIONS = (
    httpx.TransportError,
    ConnectionError,
)

# Ollama-reported failures that genuinely mean inference cannot be performed:
# the required model is missing/not pulled (404), or a server-side failure
# (5xx). Any other ResponseError (e.g. a 400 from a malformed request) is
# NOT treated as infrastructure -- it is not blindly assumed to mean the
# backend is down. It propagates as a request error, without output retries
# or taxonomy fallback.
_INFRASTRUCTURE_OLLAMA_STATUS_CODES = {404}


def _is_infrastructure_error(exc: Exception) -> bool:
    if isinstance(exc, _TRANSPORT_INFRASTRUCTURE_EXCEPTIONS):
        return True
    if isinstance(exc, OllamaResponseError):
        return (
            exc.status_code in _INFRASTRUCTURE_OLLAMA_STATUS_CODES
            or exc.status_code >= 500
        )
    return False


class InfrastructureError(RuntimeError):
    """Raised for recognized chat backend or transport failures.

    This is distinct from a classification failure: it means the model
    could not be reached or invoked at all (server down, model missing,
    transport error), not that it returned unusable output. Embedding errors
    propagate separately and are also displayed as errors by the UI.
    """


class RoutingError(RuntimeError):
    """Raised when a predicted category has no entry in the data-derived
    routing map -- a data/system inconsistency, not a triage-uncertainty
    signal.
    """


class ResolutionGenerationError(RuntimeError):
    """Raised when the chat model produced no usable resolution note after
    one retry. This is a genuine generation failure -- callers must not
    silently fabricate a generic note to paper over it.
    """


# ---------------------------------------------------------------------------
# Structured classification schema
# ---------------------------------------------------------------------------


def _build_classification_schema() -> type[BaseModel]:
    """Build the structured-output schema from the taxonomy's canonical
    category names. Never hand-duplicates the 8 labels -- Literal is built
    directly from taxonomy.py's get_category_names().
    """
    categories = get_category_names()

    class ClassificationResult(BaseModel):
        category: Literal[categories] = Field(  # type: ignore[valid-type]
            description=(
                "The single best-fit canonical category for this customer "
                "inquiry, chosen strictly from the provided category list."
            )
        )

    return ClassificationResult


def _classification_system_prompt() -> str:
    return (
        "You are a strict customer-inquiry classifier for an automotive "
        "company. Classify the inquiry into exactly one of the following "
        "canonical categories. Use each category's description and "
        "keywords to decide; do not invent a category outside this list.\n\n"
        + format_taxonomy_for_prompt()
    )


def _default_structured_classifier() -> Callable[[list], Any]:
    """Returns a callable(messages) -> ClassificationResult, backed by the
    real local Ollama chat model. Kept as a separate factory so tests can
    substitute a fake classifier without touching the network.
    """
    schema = _build_classification_schema()
    llm = ChatOllama(model=CHAT_MODEL_NAME, temperature=0)
    structured = llm.with_structured_output(schema)
    return structured.invoke


# ---------------------------------------------------------------------------
# Classification: structured attempt -> retry once -> deterministic fallback
# ---------------------------------------------------------------------------


def _invoke_classifier(classifier: Callable[[list], Any], query: str) -> str:
    """Single structured-classification attempt. Returns the category string.

    Raises InfrastructureError for backend/transport failures (never caught
    for retry/fallback purposes). Ollama request errors also propagate without
    output retry/fallback. Unusable model output is handled by the caller.
    """
    messages = [
        SystemMessage(content=_classification_system_prompt()),
        HumanMessage(content=query),
    ]
    try:
        result = classifier(messages)
    except Exception as e:
        if _is_infrastructure_error(e):
            raise InfrastructureError(
                f"chat backend unavailable during classification: {e}"
            ) from e
        raise

    category = getattr(result, "category", None)
    if category is None:
        raise ValueError(f"structured classifier returned no usable category: {result!r}")
    return category


def _deterministic_classification_fallback(query: str) -> tuple[str, str]:
    """Conservative, interpretable fallback when both LLM attempts fail.

    Decision rule (deliberately simple, using only literal evidence -- no
    unweighted mean or other unapproved combination of signals):

    1. For each category, count "literal hits" = number of matched keyword
       phrases (substring, case-insensitive) plus 1 if the category name
       itself appears in the query. The fuzzy description-overlap signal is
       intentionally NOT used here -- it is too easily ambiguous/noisy to
       serve as sole evidence for a confident category pick.
    2. If no category has any literal hits, there is no meaningful evidence
       -> return ("other", "safe_default").
    3. If exactly one category has the maximum hit count -> that category is
       the fallback pick -> return (category, "taxonomy_fallback").
    4. If multiple categories tie for the maximum hit count -> the evidence
       is ambiguous, not a confident signal -> return ("other", "safe_default").

    Callers must still set classification_failed = True regardless of which
    branch is taken here -- reaching this function at all means both LLM
    attempts already failed.
    """
    evidence = compute_fallback_evidence(query)

    literal_hits = {
        name: len(data["keyword_matches"]) + (1 if data["name_match"] else 0)
        for name, data in evidence.items()
    }
    max_hits = max(literal_hits.values())
    if max_hits == 0:
        return "other", "safe_default"

    winners = [name for name, hits in literal_hits.items() if hits == max_hits]
    if len(winners) == 1:
        return winners[0], "taxonomy_fallback"

    return "other", "safe_default"


def classify_inquiry(
    query: str, structured_classifier: Optional[Callable[[list], Any]] = None
) -> tuple[str, str, bool]:
    """Returns (category, classification_source, classification_failed).

    classification_source is one of: "llm", "llm_retry", "taxonomy_fallback",
    "safe_default". classification_failed is True only when both LLM
    attempts failed and the deterministic fallback was used, regardless of
    whether that fallback found a plausible category.

    InfrastructureError and Ollama request errors propagate uncaught -- they
    are not classification failures and must not trigger retry or fallback.
    """
    classifier = structured_classifier or _default_structured_classifier()

    try:
        category = _invoke_classifier(classifier, query)
        return category, "llm", False
    except (InfrastructureError, OllamaResponseError):
        raise
    except Exception:
        pass  # malformed/unusable output on first attempt -- retry once

    try:
        category = _invoke_classifier(classifier, query)
        return category, "llm_retry", False
    except (InfrastructureError, OllamaResponseError):
        raise
    except Exception:
        pass  # malformed/unusable output on retry too -- deterministic fallback

    category, source = _deterministic_classification_fallback(query)
    return category, source, True


# ---------------------------------------------------------------------------
# Priority: majority-vote baseline + similarity-weighted candidate
# ---------------------------------------------------------------------------

# Used only to break exact ties, in both priority methods below. This is a
# deterministic, conservative convention (favor the more severe priority
# when evidence does not clearly distinguish) -- not a claim that it is
# empirically the best tie-break, which has not been evaluated.
_SEVERITY_ORDER = {"high": 3, "medium": 2, "low": 1}


def majority_vote_priority(retrieved_past_cases: list[dict]) -> str:
    """Baseline: plain majority vote over retrieved neighbors' priorities.

    Tie-breaking (deterministic, in order):
    1. Among priorities tied for the highest vote count, prefer the one with
       the higher summed retrieval similarity.
    2. If still tied, prefer by severity: high > medium > low.
    """
    if not retrieved_past_cases:
        raise ValueError("cannot determine priority from empty retrieval results")

    counts = Counter(r["priority"] for r in retrieved_past_cases)
    max_count = max(counts.values())
    tied = [p for p, c in counts.items() if c == max_count]
    if len(tied) == 1:
        return tied[0]

    similarity_sums = {
        p: sum(r["similarity"] for r in retrieved_past_cases if r["priority"] == p)
        for p in tied
    }
    max_similarity = max(similarity_sums.values())
    still_tied = [p for p in tied if similarity_sums[p] == max_similarity]
    if len(still_tied) == 1:
        return still_tied[0]

    return max(still_tied, key=lambda p: _SEVERITY_ORDER[p])


def similarity_weighted_priority(retrieved_past_cases: list[dict]) -> str:
    """Candidate: score(priority) = sum(similarity of neighbors with that
    priority). Highest total wins.

    Tie-breaking: if weighted totals are exactly tied, prefer by severity:
    high > medium > low.

    Kept separately callable/testable for evaluation reproducibility. The
    leakage-safe evaluation found no prediction differences, so the graph
    retains majority_vote_priority.
    """
    if not retrieved_past_cases:
        raise ValueError("cannot determine priority from empty retrieval results")

    weighted_sums: dict[str, float] = defaultdict(float)
    for r in retrieved_past_cases:
        weighted_sums[r["priority"]] += r["similarity"]

    max_sum = max(weighted_sums.values())
    tied = [p for p, s in weighted_sums.items() if s == max_sum]
    if len(tied) == 1:
        return tied[0]

    return max(tied, key=lambda p: _SEVERITY_ORDER[p])


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_category(category: str) -> str:
    """Deterministic category -> routed_queue using the Phase 1 data-derived
    routing map. Raises RoutingError (a system/data error, not a triage
    signal) if the category is unexpectedly absent from that map.
    """
    routing_map = get_routing_map()
    if category not in routing_map:
        raise RoutingError(
            f"predicted category {category!r} has no entry in the "
            f"data-derived routing map: {sorted(routing_map.keys())}"
        )
    return routing_map[category]


# ---------------------------------------------------------------------------
# Resolution note generation: chat attempt -> retry once -> hard error
# ---------------------------------------------------------------------------


def _resolution_system_prompt() -> str:
    return (
        "You are drafting a short internal resolution note for a customer-"
        "service agent handling an automotive customer inquiry. Write "
        "exactly 1-2 concise sentences that are operationally useful -- you "
        "may suggest an appropriate next step.\n\n"
        "Retrieved historical cases are examples of how similar inquiries "
        "were handled -- useful for the likely handling pattern or general "
        "type of response, never as a source of facts. Do not copy or infer "
        "case-specific facts, events, entitlements, statuses, promises, "
        "available actions, or requested remedies from those cases into the "
        "current inquiry. Only treat information explicitly stated in the "
        "current inquiry as current-case facts. When the correct action "
        "depends on information that is not known, recommend verification, "
        "review, or routing rather than asserting that the action should or "
        "will occur. A retrieved case worded very similarly to the current "
        "inquiry is still only a pattern match, not confirmation -- do not "
        "let similar wording promote that case's specific details into the "
        "current inquiry's facts.\n\n"
        "Do not assert or assume any of the following unless the current "
        "inquiry itself explicitly states it:\n"
        "- company or product plans\n"
        "- whether a launch or event exists, or its status\n"
        "- availability of anything (appointments, access, credentials, "
        "stock, dates)\n"
        "- prices\n"
        "- account, payment, refund, or warranty status\n"
        "- order or delivery status\n"
        "- vehicle-specific facts not supplied by the customer\n"
        "- approval, eligibility, or entitlement to anything\n"
        "- promises or commitments\n"
        "- that an action has already been performed or completed\n"
        "- specific UI button names, interface controls, links, menu paths, "
        "or procedural steps, unless the current inquiry itself states "
        "them\n\n"
        "Respond with only the resolution note text -- no preamble, no "
        "labels, no markdown."
    )


def _format_retrieved_cases_for_prompt(retrieved_past_cases: list[dict]) -> str:
    if not retrieved_past_cases:
        return "(no similar past cases were retrieved)"
    return "\n".join(
        f"- [{r['category']}/{r['priority']}] {r['inquiry_text']}"
        for r in retrieved_past_cases
    )


def _resolution_user_prompt(
    query: str, category: str, priority: str, retrieved_past_cases: list[dict]
) -> str:
    return (
        f"Current customer inquiry: {query}\n"
        f"Predicted category: {category}\n"
        f"Determined priority: {priority}\n\n"
        "Similar past cases (context only -- NOT facts about this "
        "customer):\n"
        f"{_format_retrieved_cases_for_prompt(retrieved_past_cases)}\n\n"
        "Write the 1-2 sentence resolution note now."
    )


def _default_resolution_chat() -> Callable[[list], Any]:
    """Returns a callable(messages) -> chat response, backed by the real
    local Ollama chat model. Kept separate so tests can inject a fake.
    """
    llm = ChatOllama(model=CHAT_MODEL_NAME, temperature=0.2)
    return llm.invoke


def _normalize_resolution_note(raw_content: str) -> str:
    """Collapses all whitespace/newlines into single spaces rather than
    relying on brittle character counts as a proxy for "1-2 lines" -- the
    sentence limit is checked separately by _invoke_resolution_chat after
    normalization; no truncation is performed here.
    """
    return " ".join(raw_content.split()).strip()


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _count_sentences(note: str) -> int:
    """Small, deterministic sentence count for short English notes -- not a
    general-purpose sentence splitter (no NLP library), just enough to catch
    a note that clearly runs past the intended 1-2 sentence scope. Splits on
    '.', '!', or '?' followed by whitespace; a trailing sentence with no
    terminal punctuation still counts as one.
    """
    note = note.strip()
    if not note:
        return 0
    parts = _SENTENCE_SPLIT_RE.split(note)
    return len([p for p in parts if p.strip()])


def _invoke_resolution_chat(chat: Callable[[list], Any], messages: list) -> str:
    try:
        response = chat(messages)
    except Exception as e:
        if _is_infrastructure_error(e):
            raise InfrastructureError(
                f"chat backend unavailable during resolution generation: {e}"
            ) from e
        raise

    content = getattr(response, "content", response)
    if not isinstance(content, str):
        raise ValueError(f"resolution chat returned non-text content: {content!r}")

    note = _normalize_resolution_note(content)
    if not note:
        raise ValueError("resolution chat returned an empty note")
    if _count_sentences(note) > 2:
        raise ValueError(f"resolution note exceeds the 1-2 sentence contract: {note!r}")
    return note


def generate_resolution_note(
    query: str,
    category: str,
    priority: str,
    retrieved_past_cases: list[dict],
    chat: Optional[Callable[[list], Any]] = None,
) -> str:
    """Generates a short, grounded resolution note. Always attempted,
    including for cases that will end up escalated -- escalation means
    "verify this triage result", not "produce no resolution guidance".

    Retries once on unusable output; raises ResolutionGenerationError if the
    retry also fails. InfrastructureError and Ollama request errors propagate
    uncaught in both cases.
    """
    chat = chat or _default_resolution_chat()
    messages = [
        SystemMessage(content=_resolution_system_prompt()),
        HumanMessage(
            content=_resolution_user_prompt(query, category, priority, retrieved_past_cases)
        ),
    ]

    try:
        return _invoke_resolution_chat(chat, messages)
    except (InfrastructureError, OllamaResponseError):
        raise
    except Exception:
        pass  # unusable output on first attempt -- retry once

    try:
        return _invoke_resolution_chat(chat, messages)
    except (InfrastructureError, OllamaResponseError):
        raise
    except Exception as e:
        raise ResolutionGenerationError(
            f"resolution note generation failed after retry: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Confidence signals (kept independently observable -- no formula here)
# ---------------------------------------------------------------------------

_SMALL_NONZERO_GUARD = 1e-9


def retrieval_strength(retrieved_past_cases: list[dict]) -> float:
    """Strongest retrieved similarity. Range [0.0, 1.0]."""
    if not retrieved_past_cases:
        raise ValueError("cannot compute retrieval_strength from empty retrieval results")
    return max(r["similarity"] for r in retrieved_past_cases)


def category_agreement(retrieved_past_cases: list[dict], predicted_category: str) -> float:
    """Similarity-weighted agreement between retrieved categories and the
    predicted category: sum(similarity for matching neighbors) / sum(all
    similarities). A highly similar neighbor contributes more evidence than
    a weak one. 0.0 if total similarity is zero. Range [0.0, 1.0].
    """
    if not retrieved_past_cases:
        raise ValueError("cannot compute category_agreement from empty retrieval results")

    total_similarity = sum(r["similarity"] for r in retrieved_past_cases)
    if total_similarity <= 0.0:
        return 0.0

    matching_similarity = sum(
        r["similarity"] for r in retrieved_past_cases if r["category"] == predicted_category
    )
    return matching_similarity / total_similarity


def category_margin(retrieved_past_cases: list[dict], predicted_category: str) -> float:
    """Normalized similarity-weighted evidence margin between the predicted
    category and its strongest competitor among retrieved neighbors:

    margin = max(0, predicted_evidence - runner_up_evidence) / total_similarity

    clamped to [0.0, 1.0]. 0.0 if the predicted category has no retrieved
    support at all (predicted_evidence == 0), which this formula already
    guarantees since runner_up_evidence >= 0.
    """
    if not retrieved_past_cases:
        raise ValueError("cannot compute category_margin from empty retrieval results")

    evidence_by_category: dict[str, float] = defaultdict(float)
    for r in retrieved_past_cases:
        evidence_by_category[r["category"]] += r["similarity"]

    predicted_evidence = evidence_by_category.get(predicted_category, 0.0)
    if predicted_evidence <= 0.0:
        return 0.0

    other_totals = [v for cat, v in evidence_by_category.items() if cat != predicted_category]
    runner_up_evidence = max(other_totals) if other_totals else 0.0

    total_similarity = sum(evidence_by_category.values())
    margin = max(0.0, predicted_evidence - runner_up_evidence) / max(
        total_similarity, _SMALL_NONZERO_GUARD
    )
    return max(0.0, min(1.0, margin))


# ---------------------------------------------------------------------------
# Confidence strategies (both implemented; the graph uses "evidence_mean",
# selected by the Phase 4 leakage-safe evaluation -- see
# evaluation/results.json) + classification-failure override + threshold-
# based escalation
# ---------------------------------------------------------------------------


def compute_confidence(
    retrieval_strength_value: float,
    category_agreement_value: float,
    category_margin_value: float,
    strategy: str = "retrieval_strength",
) -> float:
    """strategy="retrieval_strength" (baseline): confidence =
    retrieval_strength -- the simplest method explicitly permitted by the
    case study.

    strategy="evidence_mean" (selected -- see evaluation/results.json):
    unweighted mean of all three signals. No arbitrary/learned weights, no
    claim this is optimal in general -- selected specifically because the
    Phase 4 evaluation showed it gives a materially more usable
    coverage/reliability trade-off than retrieval_strength on this dataset.
    """
    if strategy == "retrieval_strength":
        return retrieval_strength_value
    if strategy == "evidence_mean":
        return (
            retrieval_strength_value + category_agreement_value + category_margin_value
        ) / 3.0
    raise ValueError(f"unknown confidence strategy: {strategy!r}")


def compute_confidence_and_escalation(
    classification_failed: bool,
    retrieval_strength_value: float,
    category_agreement_value: float,
    category_margin_value: float,
    confidence_threshold: float,
    strategy: str = "retrieval_strength",
) -> tuple[float, bool]:
    """Final confidence + escalation decision.

    Mandatory override: classification_failed=True forces
    confidence=0.0/escalated=True regardless of retrieval evidence --
    nothing below this check can overwrite that outcome.

    Otherwise: confidence = compute_confidence(..., strategy), and
    escalated = confidence < confidence_threshold (strict less-than -- a
    confidence exactly equal to the threshold does NOT escalate).

    confidence_threshold must be in [0.0, 1.0]; an out-of-range value raises
    ValueError rather than being silently clamped.
    """
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError(
            f"confidence_threshold must be in [0.0, 1.0], got {confidence_threshold!r}"
        )

    if classification_failed:
        return 0.0, True

    confidence = compute_confidence(
        retrieval_strength_value,
        category_agreement_value,
        category_margin_value,
        strategy=strategy,
    )
    escalated = confidence < confidence_threshold
    return confidence, escalated


# ---------------------------------------------------------------------------
# LangGraph state + partial workflow
# ---------------------------------------------------------------------------


class TriageState(TypedDict, total=False):
    query: str
    top_k: int
    confidence_threshold: float

    category: str
    classification_source: str
    classification_failed: bool

    retrieved_past_cases: list[dict]

    priority: str
    routed_queue: str

    resolution_notes: str

    retrieval_strength: float
    category_agreement: float
    category_margin: float

    confidence: float
    escalated: bool


def classify_node(state: TriageState) -> dict:
    category, source, failed = classify_inquiry(state["query"])
    return {
        "category": category,
        "classification_source": source,
        "classification_failed": failed,
    }


def retrieve_node(state: TriageState) -> dict:
    results = retrieve_similar_cases(state["query"], state["top_k"])
    return {"retrieved_past_cases": results}


def determine_priority_node(state: TriageState) -> dict:
    priority = majority_vote_priority(state["retrieved_past_cases"])
    return {"priority": priority}


def route_node(state: TriageState) -> dict:
    return {"routed_queue": route_category(state["category"])}


def build_partial_graph():
    """Builds the Phase 2 partial workflow:
    START -> classify -> retrieve -> determine_priority -> route -> END

    Retained as-is for the existing Phase 2 tests. build_complete_graph()
    below is the real Phase 3 workflow that triage_inquiry uses.
    """
    graph = StateGraph(TriageState)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("determine_priority", determine_priority_node)
    graph.add_node("route", route_node)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "determine_priority")
    graph.add_edge("determine_priority", "route")
    graph.add_edge("route", END)

    return graph.compile()


def generate_resolution_node(state: TriageState) -> dict:
    note = generate_resolution_note(
        state["query"],
        state["category"],
        state["priority"],
        state["retrieved_past_cases"],
    )
    return {"resolution_notes": note}


def finalize_confidence_and_escalation_node(state: TriageState) -> dict:
    """Confidence strategy: "evidence_mean" -- selected over the
    "retrieval_strength" baseline by the Phase 4 leakage-safe evaluation
    (evaluation/results.json, evaluation/evaluate.py). See the evaluation
    report for the measured coverage/reliability trade-off that justified
    this choice; retrieval_strength alone showed reliability declining as
    confidence increased on this dataset, which evidence_mean did not.
    """
    retrieved = state["retrieved_past_cases"]
    category = state["category"]

    rs = retrieval_strength(retrieved)
    ca = category_agreement(retrieved, category)
    cm = category_margin(retrieved, category)

    confidence, escalated = compute_confidence_and_escalation(
        classification_failed=state["classification_failed"],
        retrieval_strength_value=rs,
        category_agreement_value=ca,
        category_margin_value=cm,
        confidence_threshold=state["confidence_threshold"],
        strategy="evidence_mean",
    )

    return {
        "retrieval_strength": rs,
        "category_agreement": ca,
        "category_margin": cm,
        "confidence": confidence,
        "escalated": escalated,
    }


def build_complete_graph():
    """Builds the complete Phase 3 workflow:
    START -> classify -> retrieve -> determine_priority -> route
          -> generate_resolution -> finalize_confidence_and_escalation -> END

    The five required case-study stages (classify, retrieve,
    determine_priority, route, generate_resolution) stay in that exact
    order; confidence/escalation finalization is one additional final node.
    """
    graph = StateGraph(TriageState)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("determine_priority", determine_priority_node)
    graph.add_node("route", route_node)
    graph.add_node("generate_resolution", generate_resolution_node)
    graph.add_node(
        "finalize_confidence_and_escalation", finalize_confidence_and_escalation_node
    )

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "determine_priority")
    graph.add_edge("determine_priority", "route")
    graph.add_edge("route", "generate_resolution")
    graph.add_edge("generate_resolution", "finalize_confidence_and_escalation")
    graph.add_edge("finalize_confidence_and_escalation", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public backend contract
# ---------------------------------------------------------------------------


def triage_inquiry(query: str, top_k: int, confidence_threshold: float) -> dict:
    """Runs the complete triage pipeline and returns exactly the fields the
    starter Streamlit UI expects. Internal fields (classification_source,
    classification_failed, retrieval_strength, category_agreement,
    category_margin) are computed and available on the graph's final state
    for tests/debugging, but are not part of this public return value.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if (
        not isinstance(confidence_threshold, (int, float))
        or isinstance(confidence_threshold, bool)
        or not (0.0 <= confidence_threshold <= 1.0)
    ):
        raise ValueError(
            f"confidence_threshold must be a number in [0.0, 1.0], got {confidence_threshold!r}"
        )

    graph = build_complete_graph()
    final_state = graph.invoke(
        {
            "query": query,
            "top_k": top_k,
            "confidence_threshold": float(confidence_threshold),
        }
    )

    return {
        "query": query,
        "category": final_state["category"],
        "priority": final_state["priority"],
        "routed_queue": final_state["routed_queue"],
        "confidence": final_state["confidence"],
        "resolution_notes": final_state["resolution_notes"],
        "retrieved_past_cases": final_state["retrieved_past_cases"],
        "escalated": final_state["escalated"],
    }
