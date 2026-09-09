"""
Automotive Inquiry Triage Assistant — Streamlit chat interface.

Frontend only. The triage logic (LLM, embeddings, vector store, LangGraph
workflow) lives in src/main.py and is wired in below via
`src.main.triage_inquiry`.
"""

import sys
from pathlib import Path

import streamlit as st

# Make `src` importable regardless of the working directory Streamlit was
# launched from (e.g. `streamlit run app/app.py` from the repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import notifications  # noqa: E402
from src.ingestion import DataValidationError  # noqa: E402
from src.main import InfrastructureError, ResolutionGenerationError, RoutingError  # noqa: E402
from src.main import triage_inquiry as _triage_inquiry  # noqa: E402

# --- Page setup + theme -----------------------------------------------------

st.set_page_config(page_title="Automotive Inquiry Triage Assistant", page_icon="🧭")

# Minimal, restrained theme that follows Streamlit's active color scheme.
# Targets Streamlit's
# data-testid selectors rather than generated/
# hashed CSS class names, and never hides a control or status element --
# only backgrounds, borders, spacing, and type color are touched.
st.markdown(
    """
    <style>
      :root {
        --paper: light-dark(#F5F3EE, #14171C);
        --surface: light-dark(#FFFFFF, #1C2027);
        --ink: light-dark(#1B2027, #EDEBE4);
        --ink-soft: light-dark(#5B5F63, #A7ACB6);
        --steel: light-dark(#3E6E91, #7FB0D1);
        --steel-tint: light-dark(#E4EEF3, #20313D);
        --line: light-dark(#DCD8CE, #333944);
        --green: light-dark(#4C7A4A, #8FBB8A);  --green-tint: light-dark(#E4EEE1, #22301F);
        --amber: light-dark(#B77F2C, #E0A855);  --amber-tint: light-dark(#F5E9D6, #3A2E1A);
        --red: light-dark(#A64438, #E2897C);    --red-tint: light-dark(#F4E1DD, #3A2320);
      }

      .stApp { background: var(--paper); }

      [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--line);
      }
      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3 { color: var(--ink); }

      h1, h2, h3 { color: var(--ink); }

      [data-testid="stChatMessage"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
      }

      [data-testid="stMetricValue"] { color: var(--ink); font-size: 1rem; }
      .stApp [data-testid="stMetricValue"] * {
        white-space: normal;
        overflow-wrap: anywhere;
        overflow: visible;
        text-overflow: clip;
      }
      [data-testid="stMetricLabel"] { color: var(--ink-soft); }

      .eyebrow-label {
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--steel);
        font-weight: 600;
        margin-bottom: 0.15rem;
      }

      .priority-pill {
        display: inline-block;
        padding: 0.15rem 0.65rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ------------------------------------------------------------------

st.markdown('<div class="eyebrow-label">AI-Powered Customer Operations</div>', unsafe_allow_html=True)
st.title("Automotive Inquiry Triage Assistant")
st.caption("Classify, prioritize, route, and escalate customer inquiries using historical case evidence.")

# --- Sidebar controls ---------------------------------------------------------

with st.sidebar:
    st.header("Triage Settings")
    top_k = st.slider("Top-K past cases", min_value=1, max_value=10, value=5)
    confidence_threshold = st.slider(
        "Confidence threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )


# --- Backend hook -----------------------------------------------------

def triage_inquiry(query: str, top_k: int, confidence_threshold: float) -> dict:
    """Delegates to the LangGraph triage pipeline in src/main.py."""
    return _triage_inquiry(query, top_k, confidence_threshold)


# --- Result rendering -----------------------------------------------------

_PRIORITY_STYLES = {
    "low": ("var(--green)", "var(--green-tint)"),
    "medium": ("var(--amber)", "var(--amber-tint)"),
    "high": ("var(--red)", "var(--red-tint)"),
}


def _priority_badge_html(priority) -> str:
    color, tint = _PRIORITY_STYLES.get(str(priority).lower(), ("var(--ink-soft)", "var(--paper)"))
    label = str(priority).upper() if priority else "N/A"
    return (
        f'<span class="priority-pill" style="background:{tint};color:{color};">{label}</span>'
    )


def render_result(result: dict) -> None:
    """Render a triage result. Every required field from the fixed output
    contract is still shown; only the presentation is refined.
    """
    st.markdown(f"**Query:** {result.get('query', '')}")

    summary_cols = st.columns(4)
    summary_cols[0].metric("Category", result.get("category", ""))
    with summary_cols[1]:
        st.markdown('<div style="font-size:.8rem;color:var(--ink-soft);">Priority</div>', unsafe_allow_html=True)
        st.markdown(_priority_badge_html(result.get("priority", "")), unsafe_allow_html=True)
    summary_cols[2].metric("Routed Queue", result.get("routed_queue", ""))
    confidence = result.get("confidence", "")
    confidence_display = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else str(confidence)
    summary_cols[3].metric("Confidence", confidence_display)

    with st.container(border=True):
        st.markdown("**Resolution Notes**")
        st.write(result.get("resolution_notes", ""))

    past_cases = result.get("retrieved_past_cases", [])
    with st.container(border=True):
        st.markdown("**Retrieved Past Cases**")
        if past_cases:
            for case in past_cases:
                if isinstance(case, dict):
                    similarity = case.get("similarity")
                    similarity_str = f"{similarity:.2f}" if isinstance(similarity, (int, float)) else "n/a"
                    st.markdown(
                        f"- [{case.get('category', '?')}/{case.get('priority', '?')}] "
                        f"(similarity {similarity_str}) {case.get('inquiry_text', '')}"
                    )
                else:
                    st.markdown(f"- {case}")
        else:
            st.markdown("_none_")

    if result.get("escalated"):
        st.warning("Human review required due to insufficient triage evidence or classification fallback.")


# --- Chat / session view --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"query": str, "result": dict|None}

# Replay history for this session.
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["query"])
    with st.chat_message("assistant"):
        if turn["result"] is not None:
            render_result(turn["result"])
        else:
            st.error(turn.get("error", "No result."))

# New inquiry input.
query = st.chat_input("Enter a customer inquiry...")
if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Triaging..."):
                result = triage_inquiry(query, top_k, confidence_threshold)
            render_result(result)
            # Notify n8n (if configured) for this newly generated result only --
            # never for history replay/reruns, and never able to affect the
            # triage result already rendered above.
            notifications.send_triage_notification(result)
            st.session_state.history.append({"query": query, "result": result})
        except InfrastructureError as e:
            # The chat/embedding backend itself is unavailable (server down,
            # model missing, transport error) -- a genuine application
            # failure, never disguised as a triage result or an escalation.
            msg = f"Backend infrastructure is unavailable: {e}"
            st.error(msg)
            st.session_state.history.append({"query": query, "result": None, "error": msg})
        except (RoutingError, ResolutionGenerationError, DataValidationError) as e:
            msg = f"Application error during triage: {e}"
            st.error(msg)
            st.session_state.history.append({"query": query, "result": None, "error": msg})
        except ValueError as e:
            msg = f"Invalid input: {e}"
            st.error(msg)
            st.session_state.history.append({"query": query, "result": None, "error": msg})
        except Exception as e:  # noqa: BLE001 -- surface as a clear app error, not a crash
            msg = f"Unexpected error: {e}"
            st.error(msg)
            st.session_state.history.append({"query": query, "result": None, "error": msg})
