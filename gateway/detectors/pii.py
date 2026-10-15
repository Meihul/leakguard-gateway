import re

_D = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],
      [3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
      [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],
      [9,8,7,6,5,4,3,2,1,0]]
_P = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],
      [8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,7,6,8,0],[4,2,8,6,5,7,3,9,0,1],
      [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]


def verhoeff_ok(num):
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def luhn_ok(num):
    total = 0
    for i, ch in enumerate(reversed(num)):
        d = int(ch)
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# (category, severity, confidence, pattern, optional checksum check on the digits-only span)
_RULES = [
    ("aadhaar", "high", 0.95, re.compile(r"(?<!\d)[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?!\d)"),
     lambda m: verhoeff_ok(re.sub(r"\D", "", m))),
    ("pan", "high", 0.8, re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), None),
    ("phone", "medium", 0.8, re.compile(r"(?<!\d)(?:\+91[\s-]?|0)?[6-9]\d{9}(?!\d)"), None),
    ("card", "high", 0.95, re.compile(r"(?<!\d)\d(?:[\s-]?\d){12,18}(?!\d)"),
     lambda m: luhn_ok(re.sub(r"\D", "", m))),
]


def detect_pii(text):
    findings = []
    for category, severity, confidence, pattern, check in _RULES:
        for m in pattern.finditer(text):
            if check and not check(m.group()):
                continue
            findings.append({"category": category, "span": m.group(), "start": m.start(),
                             "end": m.end(), "severity": severity, "confidence": confidence,
                             "detector": f"pii.{category}"})
    return findings
