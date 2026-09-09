"""Optional n8n webhook notification for finalized triage results.

This module is entirely outside the AI reasoning pipeline: it only ever
receives an already-finalized `triage_inquiry()` result and forwards it, by
HTTP POST, to an operator-configured n8n webhook. It never influences
classification, retrieval, priority, routing, confidence, or escalation --
those are decided exclusively by src/main.py before this module ever sees a
result. After rendering the result, Streamlit makes a synchronous POST
with a 3-second HTTP timeout. Delivery failures do not alter or fail the
completed triage result; the AI pipeline runs exactly once.

Disabled by default. Enabled only by setting TRIAGE_N8N_WEBHOOK_URL to a
non-empty value. A webhook/network failure here must never surface as a
triage failure -- it is always caught and logged, never raised, and there is
no automatic retry.

Downstream business rules (who gets notified, and how) belong in the n8n
workflow (see n8n/), not here -- this module's only job is to forward the
selected fields from the finalized result without changing their values.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

WEBHOOK_URL_ENV_VAR = "TRIAGE_N8N_WEBHOOK_URL"

_REQUEST_TIMEOUT_SECONDS = 3.0

# The existing triage_inquiry() field names, preserved verbatim -- no
# invented schema. retrieved_past_cases is intentionally excluded: it is not
# needed for the escalation/priority branching rules in the n8n workflow and
# keeping the payload minimal keeps this module's one job (forwarding the
# decision, not the evidence behind it) clear.
_PAYLOAD_FIELDS = (
    "query",
    "category",
    "priority",
    "routed_queue",
    "confidence",
    "escalated",
    "resolution_notes",
)


def is_enabled() -> bool:
    """True only when TRIAGE_N8N_WEBHOOK_URL is set to a non-empty value."""
    return bool(os.environ.get(WEBHOOK_URL_ENV_VAR, "").strip())


def build_payload(result: dict) -> dict:
    """Extracts exactly the existing public triage_inquiry() fields relevant
    to downstream notification routing, unchanged and unrenamed.
    """
    return {field: result[field] for field in _PAYLOAD_FIELDS}


def send_triage_notification(result: dict) -> None:
    """Best-effort synchronous notification of one newly finalized
    triage result. No-ops silently if TRIAGE_N8N_WEBHOOK_URL is unset.

    Expects the complete public triage result schema. A missing URL disables
    delivery. Network failures, timeouts, and non-2xx responses are caught
    and logged without altering or failing the completed triage result. No
    retry is attempted; the caller is expected to call this at most once per
    newly generated result.
    """
    webhook_url = os.environ.get(WEBHOOK_URL_ENV_VAR, "").strip()
    if not webhook_url:
        return

    payload = build_payload(result)
    try:
        response = httpx.post(webhook_url, json=payload, timeout=_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except Exception:
        logger.warning("n8n webhook notification failed", exc_info=True)
