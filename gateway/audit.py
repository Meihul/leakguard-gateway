import hashlib, os, sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "audit.db"
SALT = os.getenv("LEAKGUARD_SALT", "")
NO_SNIPPET = {"injection", "restricted_doc"}

def _conn():
    return sqlite3.connect(DB_PATH, timeout=10)

def init_db():
    with _conn() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, user TEXT, role TEXT, action TEXT,
            categories TEXT, detectors TEXT, masked_snippets TEXT,
            prompt_hash TEXT, latency_ms REAL, alert INTEGER DEFAULT 0)""")
        cols = [r[1] for r in con.execute("PRAGMA table_info(events)")]
        if "alert" not in cols:
            con.execute("ALTER TABLE events ADD COLUMN alert INTEGER DEFAULT 0")

def _snippet(f):
    if f.get("category") in NO_SNIPPET:
        return "-"
    s = "".join(c for c in str(f.get("span", "")) if c.isalnum())
    return "****" + s[-4:] if s else "-"

def log_event(user, role, action, findings, prompt, latency_ms, ts=None):
    init_db()
    findings = findings or []
    alert = 1 if any(f.get("alert") for f in findings) else 0
    row = (
        (ts or datetime.now()).isoformat(timespec="seconds"),
        user, role, action,
        ",".join(f.get("category", "") for f in findings),
        ",".join(f.get("detector", "") for f in findings),
        ",".join(_snippet(f) for f in findings),
        hashlib.sha256((SALT + prompt).encode()).hexdigest(),
        round(latency_ms, 1),
        alert,
    )
    with _conn() as con:
        con.execute("INSERT INTO events (ts,user,role,action,categories,detectors,"
                    "masked_snippets,prompt_hash,latency_ms,alert) VALUES (?,?,?,?,?,?,?,?,?,?)", row)