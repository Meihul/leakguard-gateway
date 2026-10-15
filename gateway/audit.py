import hashlib
import json
import os
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path

from gateway.policy import is_alert

# Shared by the approvals table and the events table the dashboard reads.
# Matches ui/dashboard.py's hardcoded path, so both point at the same file.
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "audit.db"

_EVENTS_COLUMNS = {
    # column -> "TYPE [DEFAULT ...]" for ALTER TABLE / CREATE TABLE
    "ts": "TEXT",
    "user": "TEXT",
    "role": "TEXT",
    "action": "TEXT",
    "categories": "TEXT",
    "masked_snippets": "TEXT",
    "alert": "INTEGER NOT NULL DEFAULT 0",
    "latency_ms": "REAL",
    "prompt_hash": "TEXT",
    # File uploads: never the file's content or its name, only a hash of
    # the name, its type, size, and where in it each finding was.
    "file_name_hash": "TEXT",
    "file_type": "TEXT",
    "file_size": "INTEGER",
    "file_findings": "TEXT",
}


def _db_path(db_path=None):
    return str(db_path or os.environ.get("LEAKGUARD_DB_PATH", DEFAULT_DB_PATH))


def _ensure_events_schema(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT
        )
    """)
    # Add any column this table doesn't have yet, without touching existing
    # ones or existing rows — safe to run against an events table some
    # earlier version of the code (or a hand-seeded db) already created.
    existing = {row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()}
    for column, decl in _EVENTS_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE events ADD COLUMN {column} {decl}")


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
    _ensure_events_schema(conn)
    conn.commit()
    return conn


def init_db(db_path=None):
    """Create (or migrate) the approvals and events tables. Safe to call
    on every app startup — idempotent, never touches existing rows."""
    conn = _connect(db_path)
    conn.close()


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


def log_event(user, role, action, findings, prompt, latency_ms, ts=None, db_path=None,
              file_meta=None):
    """Record one request to the events table the dashboard reads.

    Never stores the raw prompt (only its hash) or a finding's full span
    (only the same "****" + last 4 chars preview used elsewhere) — findings
    here may be raw detector output or already-safe-viewed, either way.
    file_meta, for a request with an uploaded file: {"filename_hash",
    "type", "size"}; the file's content is never passed in or stored.
    """
    findings = findings or []
    ts_value = (ts or datetime.now()).isoformat()
    categories = ",".join(sorted({f["category"] for f in findings if f.get("category")}))
    masked_snippets = ",".join(
        "****" + str(f["span"])[-4:] for f in findings if f.get("span"))
    alert = 1 if any(is_alert(f.get("category")) for f in findings) else 0
    prompt_hash = hash_prompt(prompt) if prompt else None

    file_meta = file_meta or {}
    file_findings = None
    if file_meta:
        file_findings = json.dumps([
            {k: f[k] for k in ("category", "page", "line") if k in f}
            for f in findings if f.get("source") == "file"])

    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO events (ts, user, role, action, categories, masked_snippets, "
            "alert, latency_ms, prompt_hash, file_name_hash, file_type, file_size, "
            "file_findings) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts_value, user, role, action, categories, masked_snippets, alert, latency_ms,
             prompt_hash, file_meta.get("filename_hash"), file_meta.get("type"),
             file_meta.get("size"), file_findings),
        )
        conn.commit()
    finally:
        conn.close()
