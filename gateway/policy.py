from pathlib import Path
import yaml

_POLICY_PATH = Path(__file__).with_name("policy.yaml")

with open(_POLICY_PATH, "r", encoding="utf-8") as f:
    _CONFIG = yaml.safe_load(f)

RANK = {name: i for i, name in enumerate(_CONFIG["rank"])}
CATEGORIES = _CONFIG["categories"]
PROMPTGUARD = _CONFIG.get("promptguard", {})


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
