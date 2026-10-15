"""Keeps an uploaded file attached to a conversation: the file's already
scanned and MASKED text plus its reversible-masking map, in process memory
only (never written to disk), per user, keyed by a random file_id.

Entries expire FILE_TTL_SECONDS after they were last used. An expired entry
keeps only a small tombstone (owner + filename, no content) for a while, so
its owner gets a clear "expired, upload it again" message instead of a
generic not-found.
"""
import secrets
import threading
import time

FILE_TTL_SECONDS = 30 * 60
TOMBSTONE_SECONDS = 24 * 60 * 60
MAX_FILES_PER_USER = 5

_lock = threading.Lock()
_entries = {}  # file_id -> dict


class FileNotFound(Exception):
    """Unknown file_id, or one that belongs to another user (same error,
    so a file_id can't be probed for)."""


class FileExpired(Exception):
    """The requester's own file_id, but it has expired."""


def _sweep(now):
    for file_id, entry in list(_entries.items()):
        if entry.get("expired"):
            if now > entry["tombstone_until"]:
                del _entries[file_id]
        elif now > entry["expires_at"]:
            _entries[file_id] = {"user": entry["user"], "filename": entry["filename"],
                                 "expired": True, "tombstone_until": now + TOMBSTONE_SECONDS}


def put(user, filename, file_type, size, masked_text, reveal_map, counters):
    """Store an upload for this user and return its new file_id.
    counters: the placeholder counters used when masking, so later prompts
    in the conversation continue numbering after the file's placeholders."""
    now = time.time()
    file_id = secrets.token_urlsafe(16)
    with _lock:
        _sweep(now)
        mine = sorted((e["created_at"], fid) for fid, e in _entries.items()
                      if e["user"] == user and not e.get("expired"))
        for _, old_id in mine[:max(len(mine) - MAX_FILES_PER_USER + 1, 0)]:
            del _entries[old_id]
        _entries[file_id] = {
            "user": user, "filename": filename, "type": file_type, "size": size,
            "masked_text": masked_text, "reveal_map": dict(reveal_map),
            "counters": dict(counters), "created_at": now,
            "expires_at": now + FILE_TTL_SECONDS,
        }
    return file_id


def get(file_id, user):
    """Return the entry for file_id if it belongs to user; refreshes its
    expiry. Raises FileNotFound or FileExpired."""
    now = time.time()
    with _lock:
        _sweep(now)
        entry = _entries.get(file_id or "")
        if entry is None or entry["user"] != user:
            raise FileNotFound()
        if entry.get("expired"):
            raise FileExpired(entry["filename"])
        entry["expires_at"] = now + FILE_TTL_SECONDS
        return dict(entry)


def remove(file_id, user):
    with _lock:
        entry = _entries.get(file_id or "")
        if entry is not None and entry["user"] == user:
            del _entries[file_id]
            return True
    return False


def reset():
    with _lock:
        _entries.clear()
