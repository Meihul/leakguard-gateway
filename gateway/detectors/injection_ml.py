import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from gateway.policy import PROMPTGUARD

load_dotenv()

MODEL = "meta-llama/llama-prompt-guard-2-86m"
TIMEOUT_SECONDS = 5

client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))


def detect_injection_ml(text):
    start = time.monotonic()
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": text}],
            timeout=TIMEOUT_SECONDS,
        )
        score = float(r.choices[0].message.content)
    except Exception:
        return []
    latency_ms = round((time.monotonic() - start) * 1000, 1)

    block_threshold = PROMPTGUARD.get("block_threshold", 0.9)
    warn_threshold = PROMPTGUARD.get("warn_threshold", 0.5)

    if score >= block_threshold:
        category, severity = "injection", "critical"
    elif score >= warn_threshold:
        category, severity = "injection_suspected", "medium"
    else:
        return []

    return [{
        "category": category,
        "span": text,
        "start": 0,
        "end": len(text),
        "severity": severity,
        "confidence": score,
        "detector": "promptguard.86m",
        "latency_ms": latency_ms,
    }]
