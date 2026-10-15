import hashlib
import os
import sqlite3
import time
import uuid
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "leakguard.db"


def _db_path(db_path=None):
    return str(db_path or os.environ.get("LEAKGUARD_DB_PATH", DEFAULT_DB_PATH))


def _connect(db_path=None):
    conn = sqlite3.connect(_db_path(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS approvals (
            id TEXT PRIMARY KEY,
            ts REAL NOT NULL,
            user TEXT NOT NULL,
            role TEXT NOT NULL,
            requested_doc_title TEXT,
            prompt_hash TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            approver TEXT,
            decided_ts REAL,
            answer TEXT
        )
    """)
    conn.commit()
    return conn


def hash_prompt(prompt):
    # Never store the raw prompt text; only its hash.
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def create_pending_approval(user, role, requested_doc_title, prompt, db_path=None):
    request_id = str(uuid.uuid4())
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO approvals (id, ts, user, role, requested_doc_title, prompt_hash, status) "
            "VALUES (?, ?, ?, ?, ?, ?, 'pending')",
            (request_id, time.time(), user, role, requested_doc_title, hash_prompt(prompt)),
        )
        conn.commit()
    finally:
        conn.close()
    return request_id


def get_approval(request_id, db_path=None):
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM approvals WHERE id = ?", (request_id,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def list_approvals(status=None, db_path=None):
    conn = _connect(db_path)
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM approvals WHERE status = ? ORDER BY ts", (status,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM approvals ORDER BY ts").fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def decide_approval(request_id, approver, status, answer=None, db_path=None):
    conn = _connect(db_path)
    try:
        conn.execute(
            "UPDATE approvals SET status = ?, approver = ?, decided_ts = ?, answer = ? WHERE id = ?",
            (status, approver, time.time(), answer, request_id),
        )
        conn.commit()
    finally:
        conn.close()
