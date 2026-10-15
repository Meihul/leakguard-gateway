import requests
import streamlit as st

st.set_page_config(page_title="AI Security Gateway", page_icon="🛡️")

BASE_URL = "http://localhost:8000"

st.sidebar.title("User")
user = st.sidebar.text_input("Username", "riya")
role = st.sidebar.selectbox("Role", ["intern", "manager", "hr"])
if st.sidebar.button("Clear chat"):
    st.session_state.history = []
    st.session_state.status = {}

st.session_state.setdefault("history", [])
st.session_state.setdefault("status", {})   # request_id -> latest status text
st.session_state.setdefault("error", None)

BADGE = {
    "allow": ":green[✅ Allowed]",
    "mask": ":orange[🟡 Masked]",
    "block": ":red[⛔ Blocked]",
    "confirm": ":orange[⚠️ Needs confirmation]",
    "approve": ":blue[🕓 Sent for approval]",
}

def mask_last4(value: str) -> str:
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
    if any(f.get("category") == "restricted_doc" for f in findings):
        return "You don't have access to the requested information"
    cats = sorted({f.get("category", "unknown") for f in findings})
    return ("Blocked: " + ", ".join(c.replace("_", " ") for c in cats) + " detected") if cats else "Blocked by policy"

# ---------- backend calls ----------
def call_gateway(prompt, confirm=False):
    payload = {"user": user, "role": role, "prompt": prompt}
    if confirm:
        payload["confirm"] = True
    r = requests.post(f"{BASE_URL}/chat", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def send_anyway(prompt):
    """Button callback: resend the same prompt with confirm=true."""
    try:
        res = call_gateway(prompt, confirm=True)
        st.session_state.history.append({"who": "assistant", "res": res, "prompt": prompt})
    except Exception as e:
        st.session_state.error = f"Gateway not reachable: {e}"

def check_status(request_id):
    try:
        r = requests.get(f"{BASE_URL}/requests/{request_id}", timeout=15)
        r.raise_for_status()
        data = r.json()
        st.session_state.status[request_id] = str(data.get("status", data))
    except Exception as e:
        st.session_state.status[request_id] = f"Could not check status: {e}"

# ---------- rendering ----------
def render_result(res, idx, prompt, is_last):
    action = res.get("action", "allow")
    findings = res.get("findings") or []
    st.markdown(BADGE.get(action, action))

    # yellow box for every non-allow action
    if action != "allow" and res.get("user_message"):
        st.warning(res["user_message"])

    if action == "block":
        st.error(block_reason(findings))

    elif action == "confirm":
        if is_last:
            st.button("Send anyway", key=f"confirm_{idx}",
                      on_click=send_anyway, args=(prompt,))

    elif action == "approve":
        st.info("Sent for approval")
        rid = res.get("request_id")
        if rid is not None:
            if is_last:
                st.button("Check status", key=f"status_{idx}",
                          on_click=check_status, args=(rid,))
            if rid in st.session_state.status:
                st.caption(f"Status: {st.session_state.status[rid]}")

    else:  # allow / mask
        st.markdown(res.get("response", ""))
        if res.get("sources"):
            st.caption("Sources: " + ", ".join(res["sources"]))
        if action == "mask" and res.get("masked_prompt"):
            st.markdown(
                "<div style='background:#1b1f2a;padding:8px 12px;border-radius:6px;"
                "color:#9aa3b5;font-size:.85rem'><b>What the AI received</b><br>"
                f"{res['masked_prompt']}</div>", unsafe_allow_html=True)

    if findings:
        with st.expander("Why?"):
            for f in findings:
                st.write(f"**{f.get('category')}** · {f.get('detector')} · "
                         f"confidence {f.get('confidence')} · severity {f.get('severity')}")
                st.code(mask_last4(str(f.get("span", ""))), language=None)

# ---------- page ----------
if st.session_state.error:
    st.error(st.session_state.error)
    st.session_state.error = None

last = len(st.session_state.history) - 1
for i, msg in enumerate(st.session_state.history):
    with st.chat_message(msg["who"]):
        if msg["who"] == "user":
            st.markdown(msg["text"])
        else:
            render_result(msg["res"], i, msg.get("prompt", ""), i == last)

if prompt := st.chat_input("Ask the company assistant"):
    st.session_state.history.append({"who": "user", "text": prompt})
    try:
        res = call_gateway(prompt)
        st.session_state.history.append({"who": "assistant", "res": res, "prompt": prompt})
    except Exception as e:
        st.session_state.error = f"Gateway not reachable: {e}"
    st.rerun()