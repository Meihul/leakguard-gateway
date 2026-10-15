"""Per-user session risk score, kept in process memory only.

Rises by one point for each finding whose action is "warn" or "confirm",
or whose category is "injection_suspected" (counted explicitly even though
its action already happens to be "confirm", so it still counts if that
mapping ever changes). Points decay out of a sliding window, so a user's
score reflects only their last `decay_seconds` of activity.
"""
import time
from collections import defaultdict

from gateway.policy import action_for, SESSION_RISK

THRESHOLD = SESSION_RISK.get("threshold", 3)
DECAY_SECONDS = SESSION_RISK.get("decay_seconds", 600)
POINTS_PER_EVENT = SESSION_RISK.get("points_per_event", 1)

# user -> [(timestamp, points), ...]
_events = defaultdict(list)


def _qualifies(finding):
    return finding["category"] == "injection_suspected" or action_for(finding["category"]) in ("warn", "confirm")


def _prune(user, now):
    cutoff = now - DECAY_SECONDS
    _events[user] = [(ts, pts) for ts, pts in _events[user] if ts > cutoff]


def score(user, now=None):
    now = time.monotonic() if now is None else now
    _prune(user, now)
    return sum(pts for _, pts in _events[user])


def record(user, findings, now=None):
    """Add points for each qualifying finding in this request."""
    now = time.monotonic() if now is None else now
    points = sum(POINTS_PER_EVENT for f in findings if _qualifies(f))
    if points:
        _events[user].append((now, points))
    _prune(user, now)


def reset(user=None):
    """Testing helper: clear risk state for one user, or everyone."""
    if user is None:
        _events.clear()
    else:
        _events.pop(user, None)
