from pathlib import Path
import yaml

_POLICY_PATH = Path(__file__).with_name("policy.yaml")

with open(_POLICY_PATH, "r", encoding="utf-8") as f:
    _CONFIG = yaml.safe_load(f)

RANK = {name: i for i, name in enumerate(_CONFIG["rank"])}
CATEGORIES = _CONFIG["categories"]
PROMPTGUARD = _CONFIG.get("promptguard", {})


def action_for(category):
    return CATEGORIES.get(category, {}).get("action", "allow")


def decide(findings):
    action = "allow"
    for f in findings:
        a = action_for(f["category"])
        if RANK.get(a, 0) > RANK.get(action, 0):
            action = a
    return action
