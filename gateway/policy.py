from pathlib import Path
import yaml

_POLICY_PATH = Path(__file__).with_name("policy.yaml")

with open(_POLICY_PATH, "r", encoding="utf-8") as f:
    _CONFIG = yaml.safe_load(f)

RANK = {name: i for i, name in enumerate(_CONFIG["rank"])}
CATEGORIES = _CONFIG["categories"]
PROMPTGUARD = _CONFIG.get("promptguard", {})
RAG = _CONFIG.get("rag", {})
SESSION_RISK = _CONFIG.get("session_risk", {})


def action_for(category):
    # An unmapped or malformed category is unknown risk, not cleared risk:
    # it must never silently default to "allow".
    return CATEGORIES.get(category, {}).get("action", "warn")


def decide(findings):
    action = "allow"
    for f in findings:
        a = action_for(f["category"])
        if RANK.get(a, 0) > RANK.get(action, 0):
            action = a
    return action


_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def severity_rank(severity):
    return _SEVERITY_RANK.get(severity, 0)


def resolve_overlaps(findings):
    """Keep only the highest-severity finding among any overlapping spans."""
    ordered = sorted(findings, key=lambda f: (-severity_rank(f["severity"]), f["start"]))
    kept = []
    for f in ordered:
        if any(f["start"] < k["end"] and k["start"] < f["end"] for k in kept):
            continue
        kept.append(f)
    return kept


# Categories whose findings must flag the dashboard regardless of their
# configured action (secrets stay mask-and-notify, not block, but should
# still surface as an alert).
SECRET_CATEGORIES = {"api_key", "password", "private_key"}


def is_alert(category):
    return action_for(category) == "block" or category == "canary_leak" or category in SECRET_CATEGORIES


_DEFAULT_USER_MESSAGES = {
    "warn": "This message triggered a security review.",
    "mask": "We removed some sensitive information before sending your message to the AI.",
    "confirm": "This needs your confirmation before we continue.",
    "approve": "This needs approval before we can continue.",
    "block": "We blocked this message for security reasons.",
}


def user_message_for(findings, action):
    """Plain-English explanation for the given decided action, built from
    the user_message of each distinct category that drove it; falls back
    to a generic per-action message if none of them define one."""
    seen = set()
    msgs = []
    for f in findings:
        cat = f["category"]
        if action_for(cat) == action and cat not in seen:
            seen.add(cat)
            msg = CATEGORIES.get(cat, {}).get("user_message")
            if msg:
                msgs.append(msg)
    if msgs:
        return " ".join(msgs)
    return _DEFAULT_USER_MESSAGES.get(action)
