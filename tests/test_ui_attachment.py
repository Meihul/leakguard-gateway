from pathlib import Path
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "ui" / "chat_app.py")


def _logged_in_app(monkeypatch, attached=None):
    monkeypatch.setenv("DEMO_MODE", "false")
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["token"] = "signed-token"
    at.session_state["username"] = "riya"
    at.session_state["role"] = "intern"
    if attached:
        at.session_state["attached_file"] = attached
    at.run()
    return at


def _chat_response(**extra):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"response": "INR 5 lakh", "action": "allow", "findings": [],
                              "masked_prompt": "q", "sources": ["medical_insurance.pdf"], **extra}
    return resp


def _texts(at):
    return " ".join(m.value for m in at.markdown)


ATTACHED = {"file_id": "fid-123", "name": "medical_insurance.pdf"}


def test_chip_shows_the_attached_file(monkeypatch):
    at = _logged_in_app(monkeypatch, ATTACHED)
    assert "📎 **medical_insurance.pdf** attached" in _texts(at)
    assert any(b.label == "✕" for b in at.button)


def test_no_chip_without_an_attached_file(monkeypatch):
    at = _logged_in_app(monkeypatch)
    assert "📎" not in _texts(at)
    assert not any(b.label == "✕" for b in at.button)


def test_messages_carry_the_file_id_while_attached(monkeypatch):
    at = _logged_in_app(monkeypatch, ATTACHED)
    with patch("requests.post", return_value=_chat_response(
            file_id="fid-123", file={"filename": "medical_insurance.pdf"})) as post:
        at.chat_input[0].set_value("What is the sum insured under this policy?").run()
    assert post.call_args.kwargs["json"] == {
        "prompt": "What is the sum insured under this policy?", "file_id": "fid-123"}
    assert at.session_state.attached_file == ATTACHED  # still attached afterwards


def test_close_detaches_and_questions_go_back_to_company_documents(monkeypatch):
    at = _logged_in_app(monkeypatch, ATTACHED)
    with patch("requests.delete", return_value=MagicMock(status_code=200)) as delete:
        next(b for b in at.button if b.label == "✕").click().run()
    assert delete.call_args.args[0].endswith("/files/fid-123")
    assert at.session_state.attached_file is None
    assert "📎" not in _texts(at)

    with patch("requests.post", return_value=_chat_response(sources=["Leave Policy"])) as post:
        at.chat_input[0].set_value("How many casual leaves do I get?").run()
    assert post.call_args.kwargs["json"] == {"prompt": "How many casual leaves do I get?"}


def test_expired_file_clears_the_chip_and_shows_the_message(monkeypatch):
    at = _logged_in_app(monkeypatch, ATTACHED)
    gone = MagicMock(status_code=410)
    gone.json.return_value = {"detail": "This file is no longer attached: please upload it again."}
    with patch("requests.post", return_value=gone):
        at.chat_input[0].set_value("And the deductible?").run()
    assert at.session_state.attached_file is None
    assert any("no longer attached" in e.value for e in at.error)
