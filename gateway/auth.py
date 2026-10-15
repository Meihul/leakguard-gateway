"""Demo-grade authentication: salted PBKDF2 password hashes in
gateway/users.yaml, and HMAC-signed bearer tokens (secret from the
LEAKGUARD_SECRET environment variable) that carry username, role and an
8-hour expiry. Real passwords are never stored or logged anywhere.
"""
import base64
import hashlib
import hmac
import json
import os
import time
from pathlib import Path

import yaml

USERS_PATH = Path(__file__).with_name("users.yaml")
TOKEN_TTL_SECONDS = 8 * 60 * 60  # 8 hours
PBKDF2_ITERATIONS = 100_000


def _secret():
    secret = os.environ.get("LEAKGUARD_SECRET")
    if not secret:
        raise RuntimeError(
            "LEAKGUARD_SECRET is not set. Add it to your .env (see .env.example).")
    return secret.encode("utf-8")


def load_users(path=None):
    p = Path(path) if path else USERS_PATH
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("users", {})


def hash_password(password, salt_hex=None):
    """Returns (salt_hex, hash_hex). Pass an existing salt_hex to verify;
    omit it to generate a new salted hash for a new user."""
    salt = bytes.fromhex(salt_hex) if salt_hex else os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return salt.hex(), digest.hex()


def verify_password(username, password, users=None):
    """Returns the user's role if username/password match, else None."""
    users = load_users() if users is None else users
    record = users.get(username)
    if not record:
        return None
    salt_hex = record.get("salt")
    expected_hash = record.get("hash")
    if not salt_hex or not expected_hash:
        return None
    _, actual_hash = hash_password(password, salt_hex)
    if not hmac.compare_digest(actual_hash, expected_hash):
        return None
    return record.get("role")


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def create_token(username, role, ttl_seconds=TOKEN_TTL_SECONDS):
    payload = {"username": username, "role": role, "exp": time.time() + ttl_seconds}
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(_secret(), payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{signature}"


def decode_token(token):
    """Returns {"username", "role"} if the token's signature is valid and
    it hasn't expired, else None. Never raises on malformed input."""
    if not token or "." not in token or not token.isascii():
        return None
    payload_b64, signature = token.rsplit(".", 1)
    expected_sig = hmac.new(_secret(), payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected_sig):
        return None
    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("exp", 0) < time.time():
        return None
    username, role = payload.get("username"), payload.get("role")
    if not username or not role:
        return None
    return {"username": username, "role": role}


def login(username, password):
    """Returns a signed token string, or None if credentials are invalid."""
    role = verify_password(username, password)
    if role is None:
        return None
    return create_token(username, role)
