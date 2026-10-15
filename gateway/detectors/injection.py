import re

_PATTERN_BLOCK = re.compile(
    r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above)\s+instructions"
    r"|disregard\s+(your\s+)?rules"
    r"|reveal\s+(your\s+)?system\s+prompt"
    r"|show\s+(me\s+)?your\s+(system\s+prompt|instructions)"
    r"|what\s+were\s+you\s+told"
    r"|developer\s+mode"
    r"|jailbreak"
    r"|other\s+users'?\s+data"
    r"|show\s+me\s+everyone'?s"
    r"|pichle\s+instructions\s+bhool(\s+jao)?"
    r"|apne\s+rules\s+ignore\s+karo"
    r"|system\s+prompt\s+dikhao",
    re.IGNORECASE,
)

_PATTERN_SUSPECTED = re.compile(
    r"act\s+as\s+(a|an)?"
    r"|you\s+are\s+now\s+(a|an)?"
    r"|pretend\s+you\s+are",
    re.IGNORECASE,
)


def detect_injection(text):
    findings = []
    for m in _PATTERN_BLOCK.finditer(text):
        findings.append({"category": "injection", "span": m.group(), "start": m.start(),
                         "end": m.end(), "severity": "critical", "confidence": 0.95,
                         "detector": "injection.rules"})
    for m in _PATTERN_SUSPECTED.finditer(text):
        findings.append({"category": "injection_suspected", "span": m.group(), "start": m.start(),
                         "end": m.end(), "severity": "medium", "confidence": 0.7,
                         "detector": "injection.rules"})
    return findings
