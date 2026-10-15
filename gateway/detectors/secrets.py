import re

_RULES = [
    ("api_key", "critical", re.compile(r"sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}"), None),
    ("password", "critical", re.compile(r"(?i)\b(password|passwd|pwd)\s*[:=]\s*\S+"), None),
]


def detect_secrets(text):
    findings = []
    for category, severity, pattern, check in _RULES:
        for m in pattern.finditer(text):
            if check and not check(m.group()):
                continue
            findings.append({"category": category, "span": m.group(), "start": m.start(),
                             "end": m.end(), "severity": severity, "confidence": 0.95,
                             "detector": f"regex.{category}"})
    return findings
