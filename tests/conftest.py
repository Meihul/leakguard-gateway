"""Fixtures shared by every test module: an isolated Chroma index and
audit DB, a fixed token-signing secret, fresh session-risk state, and a
Prompt Guard mock so nothing makes real network calls."""
import os
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway import file_store, session_risk
from gateway.detectors import injection_ml
from gateway.rag import ingest as rag_ingest


def _fake_promptguard_response(score):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=str(score)))])


@pytest.fixture(autouse=True)
def _reset_file_store():
    file_store.reset()
    yield
    file_store.reset()


@pytest.fixture(autouse=True)
def _reset_session_risk():
    # Session risk is an in-memory, per-user global; reset it before every
    # test so unrelated tests (many of which share user="alice" by default)
    # never bleed risk points into each other.
    session_risk.reset()
    yield
    session_risk.reset()


@pytest.fixture(autouse=True)
def _default_promptguard_mock():
    # Keep the ML injection layer from making real network calls in tests
    # that aren't specifically exercising it; individual tests can still
    # nest their own patch.object(...) on top of this to set a score.
    with patch.object(injection_ml.client.chat.completions, "create",
                       return_value=_fake_promptguard_response(0.0)):
        yield


@pytest.fixture(scope="session", autouse=True)
def test_chroma_db():
    tmp_dir = tempfile.mkdtemp(prefix="leakguard_test_chroma_")
    os.environ["LEAKGUARD_CHROMA_PATH"] = tmp_dir
    rag_ingest.ingest(db_path=tmp_dir)
    yield tmp_dir
    os.environ.pop("LEAKGUARD_CHROMA_PATH", None)
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def test_audit_db():
    tmp_dir = tempfile.mkdtemp(prefix="leakguard_test_audit_")
    tmp_path = os.path.join(tmp_dir, "leakguard.db")
    os.environ["LEAKGUARD_DB_PATH"] = tmp_path
    yield tmp_path
    os.environ.pop("LEAKGUARD_DB_PATH", None)
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def test_auth_secret():
    # Tokens are HMAC-signed with this; tests never depend on whatever (if
    # anything) is in the real .env.
    prev = os.environ.get("LEAKGUARD_SECRET")
    os.environ["LEAKGUARD_SECRET"] = "test-secret-not-for-prod"
    yield
    if prev is None:
        os.environ.pop("LEAKGUARD_SECRET", None)
    else:
        os.environ["LEAKGUARD_SECRET"] = prev
