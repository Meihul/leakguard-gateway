import os, re, unicodedata
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))
MODEL = os.getenv("LLM_MODEL")
app = FastAPI(title="LeakGuard Gateway")

# ---------- Verhoeff checksum (Aadhaar) ----------
D = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],
     [3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
     [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],
     [9,8,7,6,5,4,3,2,1,0]]
P = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],
     [8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,7,6,8,0],[4,2,8,6,5,7,3,9,0,1],
     [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]

def verhoeff_ok(num):
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = D[c][P[i % 8][int(ch)]]
    return c == 0

# ---------- Normalize ----------
def normalize(text):
    for ch in ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"]:
        text = text.replace(ch, "")
    return unicodedata.normalize("NFKC", text)

# ---------- Detectors (placeholder until Swasti's are merged) ----------
RULES = [
    ("aadhaar", "high", re.compile(r"(?<!\d)[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?!\d)"),
     lambda m: verhoeff_ok(re.sub(r"\D", "", m))),
    ("pan", "high", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), None),
    ("phone", "medium", re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"), None),
    ("api_key", "critical", re.compile(r"sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}"), None),
    ("password", "critical", re.compile(r"(?i)\b(password|passwd|pwd)\s*[:=]\s*\S+"), None),
    ("injection", "critical", re.compile(
        r"(?i)ignore\s+(all\s+)?(the\s+)?(previous|prior|above)\s+instructions"
        r"|reveal\s+(your\s+)?system\s+prompt|show\s+(me\s+)?your\s+(system\s+prompt|instructions)"
        r"|disregard\s+(your\s+)?rules|developer\s+mode|pichle\s+instructions\s+bhool"), None),
]

def detect(text):
    findings = []
    for category, severity, pattern, check in RULES:
        for m in pattern.finditer(text):
            if check and not check(m.group()):
                continue
            findings.append({"category": category, "span": m.group(), "start": m.start(),
                             "end": m.end(), "severity": severity, "confidence": 0.95,
                             "detector": f"regex.{category}"})
    return findings

# ---------- Policy ----------
POLICY = {"aadhaar": "mask", "pan": "mask", "phone": "mask",
          "api_key": "block", "password": "block", "injection": "block"}
RANK = {"allow": 0, "mask": 1, "block": 2}

def decide(findings):
    action = "allow"
    for f in findings:
        a = POLICY.get(f["category"], "allow")
        if RANK[a] > RANK[action]:
            action = a
    return action

def mask(text, findings):
    for f in sorted(findings, key=lambda f: f["start"], reverse=True):
        if POLICY.get(f["category"]) == "mask":
            text = text[:f["start"]] + f"<{f['category'].upper()}>" + text[f["end"]:]
    return text

def safe_view(findings):
    # never send raw sensitive values back out
    return [{**f, "span": "****" + f["span"][-4:]} for f in findings]

# ---------- LLM ----------
SYSTEM_PROMPT = ("You are a helpful internal company assistant. A security gateway replaces "
                 "sensitive values in the user's message with placeholders like <AADHAAR>, "
                 "<PAN> or <PHONE>. Treat them as the real values, complete the user's task "
                 "normally, and keep the placeholders unchanged in your answer.")

def ask_llm(prompt):
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            timeout=20)
        return r.choices[0].message.content
    except Exception as e:
        return f"[LLM unavailable: {type(e).__name__}] Gateway still protected the prompt."

# ---------- Endpoint ----------
class ChatRequest(BaseModel):
    user: str
    role: str
    prompt: str

@app.post("/chat")
def chat(req: ChatRequest):
    text = normalize(req.prompt)
    findings = detect(text)
    action = decide(findings)
    if action == "block":
        reasons = sorted({f["category"] for f in findings if POLICY.get(f["category"]) == "block"})
        return {"response": f"Blocked by LeakGuard: {', '.join(reasons)} detected.",
                "action": "block", "findings": safe_view(findings), "masked_prompt": None}
    masked = mask(text, findings)
    return {"response": ask_llm(masked), "action": action,
            "findings": safe_view(findings), "masked_prompt": masked}