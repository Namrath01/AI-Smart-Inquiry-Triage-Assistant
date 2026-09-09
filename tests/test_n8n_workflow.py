"""Static/configuration tests for the committed n8n workflow export
(n8n/triage-notifications.workflow.json).

These do not require n8n to be installed or running -- they only parse the
JSON and check its node expressions and branch wiring. This guards
specifically against the payload-path regression found during manual
end-to-end verification: n8n's Webhook node wraps the POST JSON body under a
top-level "body" key, so every expression reading a triage-result field must
be $json["body"]["<field>"], never the bare $json["<field>"].

Deliberately not tested here (per instructions, to avoid brittleness): node
IDs, canvas positions, or n8n version metadata.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

WORKFLOW_PATH = (
    Path(__file__).resolve().parent.parent / "n8n" / "triage-notifications.workflow.json"
)

# The fields src/notifications.py actually sends in the webhook payload.
PAYLOAD_FIELDS = (
    "query",
    "category",
    "priority",
    "routed_queue",
    "confidence",
    "escalated",
    "resolution_notes",
)


def _load_workflow() -> dict:
    with WORKFLOW_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _node(data: dict, name: str) -> dict:
    return next(n for n in data["nodes"] if n["name"] == name)


def _iter_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_strings(value)


def test_workflow_json_is_valid():
    data = _load_workflow()
    assert data["nodes"]
    assert data["connections"]


def test_escalated_condition_reads_from_webhook_body():
    node = _node(_load_workflow(), "Escalated?")
    value1 = node["parameters"]["conditions"]["boolean"][0]["value1"]
    assert value1 == '={{$json["body"]["escalated"]}}'


def test_high_priority_condition_reads_from_webhook_body():
    node = _node(_load_workflow(), "High Priority?")
    value1 = node["parameters"]["conditions"]["string"][0]["value1"]
    assert value1 == '={{$json["body"]["priority"]}}'


def test_no_bare_root_level_payload_field_references_remain():
    """Regression guard: scans every string in the workflow (not just the
    two IF-node conditions) for a $json["<field>"] reference that is not
    nested under "body" -- catches the same bug class anywhere in the file,
    including the Gmail notification message templates.
    """
    data = _load_workflow()
    all_text = "\n".join(_iter_strings(data))

    # Remove every correctly body-nested reference first.
    cleaned = re.sub(r'\$json\["body"\]\["\w+"\]', "", all_text)

    for field in PAYLOAD_FIELDS:
        bare_reference = f'$json["{field}"]'
        assert bare_reference not in cleaned, (
            f"found a root-level (bare) reference to {field!r} -- the Webhook "
            f'node wraps the POST payload under "body", so this must read '
            f'$json["body"]["{field}"]'
        )


def test_escalation_is_checked_before_priority():
    """Structural guard for the required branch precedence: the 'Escalated?'
    true-branch goes straight to the escalation notification, and only the
    false-branch feeds into 'High Priority?' -- so an escalated,
    high-priority result can never reach both notification paths.
    """
    data = _load_workflow()
    branches = data["connections"]["Escalated?"]["main"]
    true_targets = {edge["node"] for edge in branches[0]}
    false_targets = {edge["node"] for edge in branches[1]}

    assert true_targets == {"Escalation Email (Gmail - configure manually)"}
    assert false_targets == {"High Priority?"}


# ---------------------------------------------------------------------------
# Gmail notification nodes
# ---------------------------------------------------------------------------

_GMAIL_NODE_NAMES = (
    "Escalation Email (Gmail - configure manually)",
    "Urgent High-Priority Email (Gmail - configure manually)",
)


def test_notification_branches_use_gmail_nodes_not_placeholders():
    data = _load_workflow()
    for name in _GMAIL_NODE_NAMES:
        node = _node(data, name)
        assert node["type"] == "n8n-nodes-base.gmail"
    # no leftover Set-node placeholders from the earlier draft
    node_types = {n["type"] for n in data["nodes"]}
    assert "n8n-nodes-base.set" not in node_types


def test_gmail_nodes_have_correct_subjects():
    data = _load_workflow()
    escalation = _node(data, "Escalation Email (Gmail - configure manually)")
    urgent = _node(data, "Urgent High-Priority Email (Gmail - configure manually)")

    assert (
        escalation["parameters"]["subject"] == "Human Review Required - Smart Inquiry Triage"
    )
    assert (
        urgent["parameters"]["subject"] == "Urgent High-Priority Inquiry - Smart Inquiry Triage"
    )


def test_gmail_node_bodies_include_required_fields():
    data = _load_workflow()
    required_body_refs = (
        '$json["body"]["query"]',
        '$json["body"]["category"]',
        '$json["body"]["priority"]',
        '$json["body"]["routed_queue"]',
        '$json["body"]["confidence"]',
        '$json["body"]["resolution_notes"]',
    )
    for name in _GMAIL_NODE_NAMES:
        message = _node(data, name)["parameters"]["message"]
        for ref in required_body_refs:
            assert ref in message, f"{name} message is missing {ref}"

    escalation_message = _node(data, _GMAIL_NODE_NAMES[0])["parameters"]["message"]
    urgent_message = _node(data, _GMAIL_NODE_NAMES[1])["parameters"]["message"]
    assert "escalated for manual review" in escalation_message.lower()
    assert "insufficient triage evidence or classification fallback" in escalation_message.lower()
    assert "high-priority" in urgent_message.lower()
    assert "not escalated" in urgent_message.lower()


def test_gmail_message_bodies_match_exact_wording():
    """Locks in the exact internal-operational-notification wording agreed
    for both emails, so a future edit that drifts from it is caught here.
    """
    data = _load_workflow()
    escalation_message = _node(data, _GMAIL_NODE_NAMES[0])["parameters"]["message"]
    urgent_message = _node(data, _GMAIL_NODE_NAMES[1])["parameters"]["message"]

    assert escalation_message == (
        '=A customer inquiry has been escalated for manual review.\n\n'
        'Customer Inquiry: {{$json["body"]["query"]}}\n'
        'Predicted Category: {{$json["body"]["category"]}}\n'
        'Priority: {{$json["body"]["priority"]}}\n'
        'Assigned Queue: {{$json["body"]["routed_queue"]}}\n'
        'Confidence: {{$json["body"]["confidence"]}}\n\n'
        'AI-generated handling note:\n'
        '{{$json["body"]["resolution_notes"]}}\n\n'
        'Human review required due to insufficient triage evidence or classification fallback. Please review the '
        'classification, routing, and suggested handling before taking action.'
    )
    assert urgent_message == (
        '=A high-priority customer inquiry has been auto-triaged and requires prompt '
        'operational attention.\n\n'
        'Customer Inquiry: {{$json["body"]["query"]}}\n'
        'Predicted Category: {{$json["body"]["category"]}}\n'
        'Priority: {{$json["body"]["priority"]}}\n'
        'Assigned Queue: {{$json["body"]["routed_queue"]}}\n'
        'Confidence: {{$json["body"]["confidence"]}}\n\n'
        'AI-generated handling note:\n'
        '{{$json["body"]["resolution_notes"]}}\n\n'
        'This item was not escalated for manual classification review, but it is marked high '
        'priority and should be handled promptly by the assigned operational queue.'
    )


def test_no_credentials_or_recipient_address_stored_in_workflow():
    """No credentials block, no populated recipient, and no hardcoded email
    address anywhere in the committed workflow file -- both are left for
    manual configuration in the n8n UI.
    """
    data = _load_workflow()
    for name in _GMAIL_NODE_NAMES:
        node = _node(data, name)
        assert "credentials" not in node, f"{name} must not ship with credentials configured"
        assert node["parameters"]["sendTo"] == "", f"{name} must ship with an empty recipient"

    all_text = "\n".join(_iter_strings(data))
    # No plausible email address literal anywhere in the file.
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", all_text)
