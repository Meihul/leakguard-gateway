from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "ui" / "chat_app.py")
DEMO_LABELS = {"Riya (intern)", "Aman (manager)", "Priya (HR)"}


def _app(monkeypatch, demo_mode):
    # Set explicitly: load_dotenv() never overrides an existing env var, so
    # whatever is in a local .env can't leak into these tests.
    monkeypatch.setenv("DEMO_MODE", demo_mode)
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    return at


def _labels(at):
    return {b.label for b in at.button}


def _login_response(username, role):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"token": "signed-token", "username": username, "role": role}
    return resp


def test_demo_buttons_hidden_when_demo_mode_off(monkeypatch):
    at = _app(monkeypatch, "false")
    assert not DEMO_LABELS & _labels(at)
    assert len(at.text_input) == 2  # normal username/password form still there


def test_demo_buttons_shown_when_demo_mode_on(monkeypatch):
    at = _app(monkeypatch, "true")
    assert DEMO_LABELS <= _labels(at)
    assert len(at.text_input) == 2  # normal form kept alongside the buttons


@pytest.mark.parametrize("label,username,password,role", [
    ("Riya (intern)", "riya", "riya-demo123", "intern"),
    ("Aman (manager)", "aman", "aman-demo123", "manager"),
    ("Priya (HR)", "priya", "priya-demo123", "hr"),
])
def test_demo_button_calls_login_and_takes_role_from_response(monkeypatch, label, username,
                                                             password, role):
    at = _app(monkeypatch, "true")
    button = next(b for b in at.button if b.label == label)
    with patch("requests.post", return_value=_login_response(username, role)) as post:
        button.click().run()

    url = post.call_args.args[0]
    assert url.endswith("/login")
    assert post.call_args.kwargs["json"] == {"username": username, "password": password}
    assert at.session_state.token == "signed-token"
    assert at.session_state.role == role
    assert any(f"Logged in as {username} ({role})" in s.value for s in at.success)


def test_demo_button_with_rejected_login_stays_logged_out(monkeypatch):
    at = _app(monkeypatch, "true")
    button = next(b for b in at.button if b.label == "Riya (intern)")
    with patch("requests.post", return_value=MagicMock(status_code=401)):
        button.click().run()
    assert not at.session_state.token
    assert any("Invalid username or password" in e.value for e in at.error)
