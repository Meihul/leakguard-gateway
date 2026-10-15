from pathlib import Path
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "ui" / "chat_app.py")

PENDING_RESULT = {"response": "This needs HR approval.", "action": "approve", "findings": [],
                  "masked_prompt": None, "sources": [], "request_id": "req-1",
                  "user_message": "This needs HR approval. Your request has been sent."}


def _app_with_pending_request(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["token"] = "signed-token"
    at.session_state["username"] = "riya"
    at.session_state["role"] = "intern"
    at.session_state["history"] = [
        {"who": "user", "text": "What are the salaries of our employees?"},
        {"who": "assistant", "res": PENDING_RESULT, "prompt": "...", "elapsed_ms": 10.0},
    ]
    at.run()
    return at


def _check_status(at, status, answer=None):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"id": "req-1", "status": status, "answer": answer}
    resp.raise_for_status.return_value = None
    with patch("requests.get", return_value=resp) as get:
        next(b for b in at.button if b.label == "Check status").click().run()
    assert get.call_args.args[0].endswith("/requests/req-1")
    return at


def _markdown(at):
    return " ".join(m.value for m in at.markdown)


def test_approved_request_shows_the_answer_under_the_status(monkeypatch):
    at = _check_status(_app_with_pending_request(monkeypatch), "approved",
                       "Vikram Shah earns INR 32,00,000 (Source: Salary Sheet). Reference code: [ref]")
    assert any(c.value == "Status: approved" for c in at.caption)
    assert "Vikram Shah earns INR 32,00,000" in _markdown(at)
    assert not any(b.label == "Check status" for b in at.button)  # decided: no more checking


def test_denied_request_shows_a_clear_message(monkeypatch):
    at = _check_status(_app_with_pending_request(monkeypatch), "denied")
    assert any(c.value == "Status: denied" for c in at.caption)
    assert any("denied" in e.value and "can't be shared" in e.value for e in at.error)


def test_pending_request_can_be_checked_again(monkeypatch):
    at = _check_status(_app_with_pending_request(monkeypatch), "pending")
    assert any("Still waiting" in c.value for c in at.caption)
    assert any(b.label == "Check status" for b in at.button)
