"""App-level test: a Streamlit rerun/history replay must not resend a
notification for a previously generated result.

Uses streamlit.testing.v1.AppTest to drive the real app.py script. Both the
LangGraph pipeline (src.main.triage_inquiry) and the webhook dispatch
(src.notifications.send_triage_notification) are monkeypatched so this test
is fast and hermetic -- it does not call Ollama/Chroma and does not send any
real HTTP request. It verifies dispatch *counting* behavior only, not AI
pipeline correctness (already covered elsewhere) and not real webhook
delivery (covered in test_notifications.py).
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

import src.main as main_module
import src.notifications as notifications_module

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "app.py")

FAKE_RESULT = {
    "query": "placeholder",
    "category": "service",
    "priority": "high",
    "routed_queue": "Service Scheduling Team",
    "confidence": 0.9,
    "resolution_notes": "Please schedule a service visit.",
    "retrieved_past_cases": [],
    "escalated": False,
}


def _fake_triage_inquiry(query, top_k, confidence_threshold):
    return {**FAKE_RESULT, "query": query}


def test_rerun_does_not_resend_notification_for_previous_result(monkeypatch):
    calls = []
    monkeypatch.setattr(main_module, "triage_inquiry", _fake_triage_inquiry)
    monkeypatch.setattr(
        notifications_module, "send_triage_notification", lambda result: calls.append(result)
    )

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception

    # Submit one new inquiry -- should dispatch exactly once.
    at.chat_input[0].set_value("My brakes are grinding.").run()
    assert not at.exception
    assert len(calls) == 1
    assert calls[0]["query"] == "My brakes are grinding."

    # Trigger further reruns via unrelated widget interaction (sidebar
    # sliders), which replays history but submits no new chat_input. Must
    # NOT dispatch again for the already-notified result.
    at.sidebar.slider[0].set_value(3).run()
    assert not at.exception
    assert len(calls) == 1

    at.sidebar.slider[1].set_value(0.7).run()
    assert not at.exception
    assert len(calls) == 1

    # A second, distinct new inquiry should dispatch exactly one more time.
    at.chat_input[0].set_value("I was charged twice for my payment.").run()
    assert not at.exception
    assert len(calls) == 2
    assert calls[1]["query"] == "I was charged twice for my payment."
