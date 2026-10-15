import re

# (secret type, category, severity, confidence, pattern)
_RULES = [
    ("openai_key", "api_key", "critical", 0.95, re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
    ("aws_access_key", "api_key", "critical", 0.95, re.compile(r"AKIA[0-9A-Z]{16}")),
    ("github_token", "api_key", "critical", 0.95, re.compile(r"ghp_[A-Za-z0-9]{36}")),
    ("google_key", "api_key", "critical", 0.9, re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("password", "password", "critical", 0.95,
     re.compile(r"(?i)\b(password|passwd|pwd)\s*[:=]\s*\S+")),
    ("private_key", "private_key", "critical", 0.95,
     re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]


def detect_secrets(text):
    findings = []
    for secret_type, category, severity, confidence, pattern in _RULES:
        for m in pattern.finditer(text):
            findings.append({"category": category, "span": m.group(), "start": m.start(),
                             "end": m.end(), "severity": severity, "confidence": confidence,
                             "detector": f"secrets.{secret_type}"})
    return findings
