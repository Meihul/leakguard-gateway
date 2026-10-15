import re

_PATTERN = re.compile(
    r"(?i)ignore\s+(all\s+)?(the\s+)?(previous|prior|above)\s+instructions"
    r"|reveal\s+(your\s+)?system\s+prompt|show\s+(me\s+)?your\s+(system\s+prompt|instructions)"
    r"|disregard\s+(your\s+)?rules|developer\s+mode|pichle\s+instructions\s+bhool")


def detect_injection(text):
    findings = []
    for m in _PATTERN.finditer(text):
        findings.append({"category": "injection", "span": m.group(), "start": m.start(),
                         "end": m.end(), "severity": "critical", "confidence": 0.95,
                         "detector": "regex.injection"})
    return findings
