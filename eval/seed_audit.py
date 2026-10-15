import random, sys
from datetime import datetime, timedelta
sys.path.insert(0, ".")
from gateway.audit import log_event

CATS = [("aadhaar", "mask", "regex.aadhaar", "2345 6789 0124"), ("pan", "mask", "regex.pan", "ABCDE1234F"),
        ("api_key", "block", "regex.api_key", "sk-test1234567890abcd"), ("injection", "block", "injection.rules", "x"),
        ("restricted_doc", "block", "rbac", "x"), (None, "allow", "", "")]
now = datetime.now()
for _ in range(50):
    c, action, det, span = random.choice(CATS)
    findings = [{"category": c, "detector": det, "span": span}] if c else []
    log_event(random.choice(["riya", "arjun", "meera"]), random.choice(["intern", "manager", "hr"]),
              action, findings, f"prompt {random.random()}", random.uniform(8, 40),
              ts=now - timedelta(minutes=random.randint(0, 180)))

print("Seeded 50 events into audit.db")