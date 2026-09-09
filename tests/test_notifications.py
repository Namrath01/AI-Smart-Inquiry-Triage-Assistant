"""Tests for the optional n8n webhook notification module (src/notifications.py).

This module is deliberately outside the AI reasoning pipeline -- these tests
only check that it forwards an already-finalized result correctly, never
raises, and is fully inert when disabled. They do not touch classification,
retrieval, priority, routing, confidence, or escalation.
"""

from __future__ import annotations

import httpx
import pytest

from src import notifications

SAMPLE_RESULT = {
    "query": "My brakes are grinding.",
    "category": "service",
    "priority": "high",
    "routed_queue": "Service Scheduling Team",
    "confidence": 0.9047,
    "resolution_notes": "Please schedule a brake inspection.",
    "retrieved_past_cases": [
        {"case_id": "CASE-0003", "category": "service", "priority": "high", "similarity": 0.9}
    ],
    "escalated": False,
}


class FakeResponse:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad status", request=None, response=self)


class FakePost:
    """Records every call; can be configured to raise or return a bad status."""

    def __init__(self, raise_exc: Exception | None = None, status_code: int = 200):
        self.calls: list[dict] = []
        self.raise_exc = raise_exc
        self.status_code = status_code

    def __call__(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self.raise_exc is not None:
            raise self.raise_exc
        return FakeResponse(self.status_code)


# ---------------------------------------------------------------------------
# Enabled / disabled gating
# ---------------------------------------------------------------------------


def test_disabled_when_webhook_url_unset(monkeypatch):
    monkeypatch.delenv(notifications.WEBHOOK_URL_ENV_VAR, raising=False)
    assert notifications.is_enabled() is False

    fake_post = FakePost()
    monkeypatch.setattr(notifications.httpx, "post", fake_post)

    notifications.send_triage_notification(SAMPLE_RESULT)

    assert fake_post.calls == []


def test_disabled_when_webhook_url_blank(monkeypatch):
    monkeypatch.setenv(notifications.WEBHOOK_URL_ENV_VAR, "   ")
    assert notifications.is_enabled() is False

    fake_post = FakePost()
    monkeypatch.setattr(notifications.httpx, "post", fake_post)

    notifications.send_triage_notification(SAMPLE_RESULT)

    assert fake_post.calls == []


def test_enabled_sends_request_to_configured_url(monkeypatch):
    monkeypatch.setenv(notifications.WEBHOOK_URL_ENV_VAR, "https://n8n.example.com/webhook/triage")
    assert notifications.is_enabled() is True

    fake_post = FakePost()
    monkeypatch.setattr(notifications.httpx, "post", fake_post)

    notifications.send_triage_notification(SAMPLE_RESULT)

    assert len(fake_post.calls) == 1
    assert fake_post.calls[0]["url"] == "https://n8n.example.com/webhook/triage"


# ---------------------------------------------------------------------------
# Payload preservation
# ---------------------------------------------------------------------------


def test_payload_preserves_existing_field_names_and_values(monkeypatch):
    monkeypatch.setenv(notifications.WEBHOOK_URL_ENV_VAR, "https://n8n.example.com/webhook/triage")
    fake_post = FakePost()
    monkeypatch.setattr(notifications.httpx, "post", fake_post)

    notifications.send_triage_notification(SAMPLE_RESULT)

    payload = fake_post.calls[0]["json"]
    assert payload == {
        "query": SAMPLE_RESULT["query"],
        "category": SAMPLE_RESULT["category"],
        "priority": SAMPLE_RESULT["priority"],
        "routed_queue": SAMPLE_RESULT["routed_queue"],
        "confidence": SAMPLE_RESULT["confidence"],
        "escalated": SAMPLE_RESULT["escalated"],
        "resolution_notes": SAMPLE_RESULT["resolution_notes"],
    }
    # retrieved_past_cases is deliberately not forwarded
    assert "retrieved_past_cases" not in payload


def test_build_payload_uses_no_invented_field_names():
    payload = notifications.build_payload(SAMPLE_RESULT)
    assert set(payload.keys()) == {
        "query", "category", "priority", "routed_queue",
        "confidence", "escalated", "resolution_notes",
    }


# ---------------------------------------------------------------------------
# Failure isolation: never raises, no retry
# ---------------------------------------------------------------------------


def test_network_failure_does_not_raise(monkeypatch):
    monkeypatch.setenv(notifications.WEBHOOK_URL_ENV_VAR, "https://n8n.example.com/webhook/triage")
    fake_post = FakePost(raise_exc=httpx.ConnectError("boom"))
    monkeypatch.setattr(notifications.httpx, "post", fake_post)

    notifications.send_triage_notification(SAMPLE_RESULT)  # must not raise

    assert len(fake_post.calls) == 1  # attempted once, no retry


def test_timeout_does_not_raise(monkeypatch):
    monkeypatch.setenv(notifications.WEBHOOK_URL_ENV_VAR, "https://n8n.example.com/webhook/triage")
    fake_post = FakePost(raise_exc=httpx.TimeoutException("timed out"))
    monkeypatch.setattr(notifications.httpx, "post", fake_post)

    notifications.send_triage_notification(SAMPLE_RESULT)  # must not raise

    assert len(fake_post.calls) == 1  # attempted once, no retry


def test_bad_http_status_does_not_raise(monkeypatch):
    monkeypatch.setenv(notifications.WEBHOOK_URL_ENV_VAR, "https://n8n.example.com/webhook/triage")
    fake_post = FakePost(status_code=500)
    monkeypatch.setattr(notifications.httpx, "post", fake_post)

    notifications.send_triage_notification(SAMPLE_RESULT)  # must not raise

    assert len(fake_post.calls) == 1


def test_short_timeout_is_used(monkeypatch):
    monkeypatch.setenv(notifications.WEBHOOK_URL_ENV_VAR, "https://n8n.example.com/webhook/triage")
    fake_post = FakePost()
    monkeypatch.setattr(notifications.httpx, "post", fake_post)

    notifications.send_triage_notification(SAMPLE_RESULT)

    assert fake_post.calls[0]["timeout"] == notifications._REQUEST_TIMEOUT_SECONDS
    assert notifications._REQUEST_TIMEOUT_SECONDS <= 5.0
