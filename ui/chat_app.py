import re
import requests
import streamlit as st

st.set_page_config(page_title="AI Security Gateway", page_icon="🛡️")

st.sidebar.title("User")
user = st.sidebar.text_input("Username", "riya")
role = st.sidebar.selectbox("Role", ["intern", "manager", "hr"])
demo = st.sidebar.toggle("Demo mode", value=True)
if st.sidebar.button("Clear chat"):
    st.session_state.history = []

if "history" not in st.session_state:
    st.session_state.history = []

BADGE = {
    "allow": ":green[✅ Allowed]",
    "mask": ":orange[🟡 Masked]",
    "block": ":red[⛔ Blocked]",
}

def mask_last4(value: str) -> str:
    """Never show the raw value: keep only the last 4 letters/digits."""
    alnum_total = sum(c.isalnum() for c in value)
    out, seen = [], 0
    for c in value:
        if c.isalnum():
            seen += 1
            out.append(c if seen > alnum_total - 4 else "X")
        else:
            out.append(c)
    return "".join(out)

def block_reason(findings):
    cats = sorted({f["category"] for f in findings})
    return "Blocked: " + ", ".join(c.replace("_", " ") for c in cats) + " detected" if cats else "Blocked by policy"

# ---------- demo mode: 4 fake responses ----------
def fake_gateway(prompt: str):
    p = prompt.lower()
    if "ignore" in p and "instructions" in p:
        return {"response": "", "action": "block", "masked_prompt": prompt,
                "findings": [{"category": "injection", "span": "Ignore all previous instructions",
                              "severity": "high", "confidence": 0.9, "detector": "injection.rules"}]}
    if "sk-" in p or "api_key" in p:
        return {"response": "", "action": "block", "masked_prompt": prompt,
                "findings": [{"category": "api_key", "span": "sk-test1234567890abcdefghij",
                              "severity": "high", "confidence": 0.95, "detector": "secrets.openai"}]}
    if "aadhaar" in p:
        return {"response": "Done, I've noted your new address.", "action": "mask",
                "masked_prompt": prompt.replace("2345 6789 0124", "<AADHAAR_1>"),
                "findings": [{"category": "aadhaar", "span": "2345 6789 0124",
                              "severity": "high", "confidence": 0.95, "detector": "pii.aadhaar"}]}
    return {"response": "Demo reply: here are 3 points on the leave policy.", "action": "allow",
            "masked_prompt": prompt, "findings": []}

def call_gateway(prompt: str):
    if demo:
        return fake_gateway(prompt)
    r = requests.post("http://localhost:8000/chat",
                      json={"user": user, "role": role, "prompt": prompt}, timeout=30)
    r.raise_for_status()
    return r.json()

# ---------- rendering ----------
def render_result(res):
    action = res["action"]
    st.markdown(BADGE.get(action, action))

    if action == "block":
        st.error(block_reason(res["findings"]))
    else:
        st.markdown(res["response"])

    if action == "mask":
        st.caption("What the LLM actually received:")
        st.code(res["masked_prompt"], language=None)

    if res["findings"]:
        with st.expander("Why?"):
            for f in res["findings"]:
                st.write(f"**{f['category']}** · {f['detector']} · "
                         f"confidence {f['confidence']} · severity {f['severity']}")
                st.code(mask_last4(f.get("span", "")), language=None)

for msg in st.session_state.history:
    with st.chat_message(msg["who"]):
        if msg["who"] == "user":
            st.markdown(msg["text"])
        else:
            render_result(msg["res"])

# ---------- input ----------
if prompt := st.chat_input("Ask the company assistant"):
    st.session_state.history.append({"who": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        res = call_gateway(prompt)
    except Exception:
        st.error("Gateway not reachable")
        st.stop()
    with st.chat_message("assistant"):
        render_result(res)
    st.session_state.history.append({"who": "assistant", "res": res})