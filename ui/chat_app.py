import os
import re
import time
from html import escape
from pathlib import Path

import requests
import streamlit as st
import yaml
from dotenv import load_dotenv

from theme import SHIELD_SVG, alert, apply_theme, meta, pill

st.set_page_config(page_title="LeakGuard", page_icon="🛡️", layout="wide")
apply_theme()

load_dotenv()
BASE_URL = "http://localhost:8000"
DEMO_MODE = os.getenv("DEMO_MODE", "false").strip().lower() == "true"
DEMO_ACCOUNTS_PATH = Path(__file__).with_name("demo_accounts.yaml")


def load_demo_accounts():
    try:
        with open(DEMO_ACCOUNTS_PATH, "r", encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("accounts", [])
    except Exception:
        return []

st.session_state.setdefault("token", None)
st.session_state.setdefault("username", None)
st.session_state.setdefault("role", None)
st.session_state.setdefault("login_error", None)


def auth_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}


def do_login(username, password):
    try:
        r = requests.post(f"{BASE_URL}/login",
                          json={"username": username, "password": password}, timeout=15)
    except Exception as e:
        st.session_state.login_error = f"Gateway not reachable: {e}"
        return
    if r.status_code == 200:
        data = r.json()
        st.session_state.token = data["token"]
        st.session_state.username = data["username"]
        st.session_state.role = data["role"]
        st.session_state.login_error = None
    else:
        st.session_state.login_error = "Invalid username or password."


def do_logout():
    for key in ("token", "username", "role", "history", "status", "pending_prompt",
                "pending_upload", "pending_file_id", "attached_file"):
        st.session_state.pop(key, None)


def login_page():
    with st.container(key="lg-login-card"):
        st.markdown(f'''<div class="lg-brand">{SHIELD_SVG}
            <div class="lg-title">LeakGuard</div>
            <div class="lg-tagline">AI security gateway</div></div>''', unsafe_allow_html=True)
        with st.form("login_form", border=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", width="stretch")
        if submitted:
            do_login(username, password)
            if st.session_state.token:
                st.rerun()

        if DEMO_MODE:
            accounts = load_demo_accounts()
            if accounts:
                st.caption("Demo mode: one-click login")
                # Same /login call as the form above; the callback runs before
                # the next script run, so the app opens straight away.
                for acct in accounts:
                    st.button(acct["label"], key=f"demo_login_{acct['username']}",
                              on_click=do_login, args=(acct["username"], acct["password"]),
                              width="stretch")

        if st.session_state.login_error:
            alert("block", st.session_state.login_error, key="login-error")


if not st.session_state.token:
    login_page()
    st.stop()

# ---------- shared sidebar: identity, rendered on every page ----------
with st.sidebar:
    st.title("LeakGuard")
    alert("allow", f"Logged in as {st.session_state.username} ({st.session_state.role})",
          key="identity")
    if st.button("Logout"):
        do_logout()
        st.rerun()

_PLACEHOLDER_RE = re.compile(r"<[A-Z_]+_\d+>")


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


def build_summary(res, elapsed_ms):
    """One-line summary built only from real response data: how many
    items were masked (counted from the actual placeholders in
    masked_prompt), how many attack-type findings fired, and how long it
    took (the response's own latency_ms if the backend ever sends one,
    otherwise the time measured here in the UI)."""
    masked_count = len(_PLACEHOLDER_RE.findall(res.get("masked_prompt") or ""))
    findings = res.get("findings") or []
    attack_count = sum(1 for f in findings if "injection" in (f.get("category") or ""))
    latency = res.get("latency_ms")
    if not isinstance(latency, (int, float)):
        latency = elapsed_ms
    item_word = "item" if masked_count == 1 else "items"
    attack_word = "attack" if attack_count == 1 else "attacks"
    return f"{masked_count} {item_word} masked · {attack_count} {attack_word} · {latency:.0f} ms"


# ---------- backend calls ----------
UPLOAD_TYPES = ["txt", "pdf", "docx", "csv"]
DEFAULT_FILE_PROMPT = "Please review the attached file."


class GatewayRejected(Exception):
    """The gateway refused the request (e.g. bad file type or size); the
    message is the gateway's own explanation, safe to show as-is."""


class AttachedFileGone(GatewayRejected):
    """The attached file_id expired or is unknown; the chip is cleared."""


def call_gateway(prompt, confirm=False, upload=None, file_id=None):
    """upload: {"name", "data", "type"} or None. With a new file, the
    request is sent as multipart form data; otherwise as JSON, with the
    file_id of the attached file (if any) so the answer comes from it."""
    start = time.monotonic()
    if upload:
        form = {"prompt": prompt}
        if confirm:
            form["confirm"] = "true"
        r = requests.post(f"{BASE_URL}/chat", data=form, headers=auth_headers(), timeout=60,
                          files={"file": (upload["name"], upload["data"],
                                          upload.get("type") or "application/octet-stream")})
    else:
        payload = {"prompt": prompt}
        if confirm:
            payload["confirm"] = True
        if file_id:
            payload["file_id"] = file_id
        r = requests.post(f"{BASE_URL}/chat", json=payload, headers=auth_headers(), timeout=60)
    elapsed_ms = (time.monotonic() - start) * 1000
    if file_id and r.status_code in (404, 410):
        st.session_state.attached_file = None
        raise AttachedFileGone(r.json().get("detail", "The attached file is no longer available."))
    if r.status_code == 400:
        raise GatewayRejected(r.json().get("detail", "The gateway rejected this request."))
    r.raise_for_status()
    res = r.json()
    if res.get("file_id"):
        # A successful upload (or follow-up) keeps the file attached.
        st.session_state.attached_file = {"file_id": res["file_id"],
                                          "name": (res.get("file") or {}).get("filename", "file")}
    return res, elapsed_ms


def detach_file():
    """Chip close callback: forget the attached file here and on the gateway."""
    attached = st.session_state.get("attached_file")
    st.session_state.attached_file = None
    if attached:
        try:
            requests.delete(f"{BASE_URL}/files/{attached['file_id']}", headers=auth_headers(),
                            timeout=10)
        except Exception:
            pass  # it expires on the gateway anyway


def _gateway_error(e):
    return str(e) if isinstance(e, GatewayRejected) else f"Gateway not reachable: {e}"


def send_anyway(prompt, upload=None, file_id=None):
    """Button callback: resend the same prompt (and file) with confirm=true."""
    try:
        res, elapsed_ms = call_gateway(prompt, confirm=True, upload=upload, file_id=file_id)
        st.session_state.history.append({"who": "assistant", "res": res, "prompt": prompt,
                                         "upload": upload, "file_id": file_id,
                                         "elapsed_ms": elapsed_ms})
    except Exception as e:
        st.session_state.error = _gateway_error(e)


def check_status(request_id):
    try:
        r = requests.get(f"{BASE_URL}/requests/{request_id}", headers=auth_headers(), timeout=15)
        r.raise_for_status()
        data = r.json()
        st.session_state.status[request_id] = {"status": data.get("status"),
                                               "answer": data.get("answer")}
    except Exception as e:
        st.session_state.status[request_id] = {"error": f"Could not check status: {e}"}


def render_request_status(info, key):
    """Status of an approval request, with the answer once approved."""
    if info.get("error"):
        st.caption(info["error"])
        return
    status = info.get("status")
    st.caption(f"Status: {status}")
    if status == "approved":
        if info.get("answer"):
            st.markdown(info["answer"])
        else:
            alert("mask", "Approved, but no answer was returned. Please ask again.",
                  key=f"noanswer-{key}")
    elif status == "denied":
        alert("block", "Your request was denied, so this information can't be shared with "
              "you. Contact HR if you think you need access.", key=f"denied-{key}")
    else:
        st.caption("Still waiting for HR or a manager to review it.")


# ---------- rendering ----------
def finding_origin(f):
    if f.get("source") != "file":
        return ""
    if f.get("page"):
        return f" · from file, page {f['page']}"
    if f.get("line"):
        return f" · from file, line {f['line']}"
    return " · from file"


def render_result(res, idx, prompt, is_last, elapsed_ms=None, upload=None, file_id=None):
    action = res.get("action", "allow")
    findings = res.get("findings") or []
    st.markdown(pill(action), unsafe_allow_html=True)

    # explanation card (coloured by the action) for every non-allow action
    if action != "allow" and res.get("user_message"):
        alert(action, res["user_message"], key=f"msg-{idx}")

    if action == "block":
        alert("block", block_reason(findings), key=f"reason-{idx}")

    elif action == "confirm":
        if is_last:
            st.button("Send anyway", key=f"confirm_{idx}",
                      on_click=send_anyway, args=(prompt, upload, file_id))

    elif action == "approve":
        alert("approve", "Sent for approval", key=f"sent-{idx}")
        rid = res.get("request_id")
        if rid is not None:
            info = st.session_state.status.get(rid)
            if not info or info.get("status") not in ("approved", "denied"):
                st.button("Check status", key=f"status_{idx}",
                          on_click=check_status, args=(rid,))
            if info:
                render_request_status(info, key=idx)

    else:  # allow / mask
        st.markdown(res.get("response", ""))
        if res.get("sources"):
            st.caption("Sources: " + ", ".join(res["sources"]))
        if action == "mask" and res.get("masked_prompt"):
            st.markdown(
                '<div class="lg-received"><div class="lg-label">What the AI received</div>'
                f'<div class="lg-mono">{escape(res["masked_prompt"])}</div></div>',
                unsafe_allow_html=True)

    if findings:
        with st.expander("Why?"):
            for f in findings:
                st.markdown(
                    f'<div class="lg-finding"><b>{escape(str(f.get("category")))}</b>'
                    f'<span class="lg-dim"> · {escape(str(f.get("detector")))} · confidence '
                    f'{escape(str(f.get("confidence")))} · severity '
                    f'{escape(str(f.get("severity")))}{escape(finding_origin(f))}</span></div>',
                    unsafe_allow_html=True)
                st.code(mask_last4(str(f.get("span", ""))), language=None)

    if elapsed_ms is not None:
        meta(build_summary(res, elapsed_ms))


def chat_page():
    if st.sidebar.button("Clear chat"):
        st.session_state.history = []
        st.session_state.status = {}
        detach_file()

    st.session_state.setdefault("history", [])
    st.session_state.setdefault("status", {})   # request_id -> latest status text
    st.session_state.setdefault("error", None)
    st.session_state.setdefault("pending_prompt", None)
    st.session_state.setdefault("pending_upload", None)
    st.session_state.setdefault("pending_file_id", None)
    st.session_state.setdefault("attached_file", None)  # {"file_id", "name"} while attached

    if st.session_state.error:
        alert("block", st.session_state.error, key="chat-error")
        st.session_state.error = None

    last = len(st.session_state.history) - 1
    for i, msg in enumerate(st.session_state.history):
        with st.chat_message(msg["who"]):
            if msg["who"] == "user":
                st.markdown(msg["text"])
                if msg.get("filename"):
                    st.caption(f"📎 {msg['filename']}")
            else:
                render_result(msg["res"], i, msg.get("prompt", ""), i == last,
                             msg.get("elapsed_ms"), msg.get("upload"), msg.get("file_id"))

    # A prompt was just submitted (previous run) and is awaiting its
    # response: show it being worked on, make the actual backend call, then
    # stop disabling the input. Splitting this into its own run (rather than
    # doing it inline below) is what makes the user's own message appear
    # immediately, before the backend call even starts.
    if st.session_state.pending_prompt is not None:
        prompt = st.session_state.pending_prompt
        upload = st.session_state.pending_upload
        file_id = st.session_state.pending_file_id
        with st.chat_message("assistant"):
            with st.spinner("LeakGuard is checking your message…"):
                try:
                    res, elapsed_ms = call_gateway(prompt, upload=upload, file_id=file_id)
                    st.session_state.history.append(
                        {"who": "assistant", "res": res, "prompt": prompt, "upload": upload,
                         "file_id": file_id, "elapsed_ms": elapsed_ms})
                except Exception as e:
                    st.session_state.error = _gateway_error(e)
        st.session_state.pending_prompt = None
        st.session_state.pending_upload = None
        st.session_state.pending_file_id = None
        st.rerun()

    # The attached-file chip, just above the input: every message is sent
    # with this file_id (answered from the file only) until it is removed.
    attached_file = st.session_state.attached_file
    if attached_file:
        with st.container(key="lg-chip"):
            chip, close = st.columns([12, 1], vertical_alignment="center")
            chip.markdown(f"📎 **{attached_file['name']}** attached")
            close.button("✕", key="detach_file", help="Remove this file; questions go back to "
                         "company documents", on_click=detach_file)

    # accept_file puts an upload (paperclip) button right next to the input.
    # The file is only held in this session's memory and sent to the gateway.
    submitted = st.chat_input("Ask the company assistant, or attach a file",
                              accept_file=True, file_type=UPLOAD_TYPES, max_upload_size=5,
                              disabled=st.session_state.pending_prompt is not None)
    if submitted:
        text = (submitted.text or "").strip()
        attached = submitted.files[0] if submitted.files else None
        upload = None
        if attached is not None:
            upload = {"name": attached.name, "data": attached.getvalue(), "type": attached.type}
            text = text or DEFAULT_FILE_PROMPT
        if text:
            # A new file replaces the attached one; otherwise keep using it.
            file_id = attached_file["file_id"] if (attached_file and not upload) else None
            filename = upload["name"] if upload else (attached_file or {}).get("name")
            st.session_state.history.append({"who": "user", "text": text, "filename": filename})
            st.session_state.pending_prompt = text
            st.session_state.pending_upload = upload
            st.session_state.pending_file_id = file_id
            st.rerun()


# ---------- page ----------
pg = st.navigation([
    st.Page(chat_page, title="Chat", icon="💬", default=True),
    st.Page("pages/1_Approvals.py", title="Approvals", icon="🗂️"),
    st.Page("pages/2_Dashboard.py", title="Dashboard", icon="📊"),
])
pg.run()
