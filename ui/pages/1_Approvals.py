import requests
import streamlit as st

from theme import alert, apply_theme

apply_theme()

BASE_URL = "http://localhost:8000"

st.title("Approvals")

# Role comes from the login (st.session_state, set from the gateway's
# /login response) — never a selector. The gateway enforces this too.
role = st.session_state.get("role")
if role not in ("hr", "manager"):
    alert("mask", "Only HR and managers can view this page.", key="no-access")
    st.stop()


def auth_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}


st.caption(f"Signed in as {st.session_state.get('username')} ({role}). "
          "Requests waiting for a decision.")
if st.button("Refresh"):
    st.rerun()


def decide(approval_id, decision):
    r = requests.post(f"{BASE_URL}/approvals/{approval_id}",
                      json={"decision": decision}, headers=auth_headers(), timeout=15)
    r.raise_for_status()


try:
    r = requests.get(f"{BASE_URL}/approvals", params={"status": "pending"},
                     headers=auth_headers(), timeout=15)
    r.raise_for_status()
    items = r.json()
except Exception as e:
    alert("block", f"Gateway not reachable: {e}", key="unreachable")
    st.stop()

if isinstance(items, dict):
    items = items.get("approvals", [])

if not items:
    alert("allow", "Nothing pending.", key="nothing-pending")

for it in items:
    aid = it.get("id")
    with st.container(border=True, key=f"lgcard-{aid}"):
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"**{it.get('user', '?')}** ({it.get('role', '?')}) requesting: "
                        f"{it.get('requested_doc_title', 'a restricted document')}")
        with c2:
            a, d = st.columns(2)
            if a.button("Approve", key=f"a_{aid}"):
                try:
                    decide(aid, "approved"); st.rerun()
                except Exception as e:
                    alert("block", str(e), key=f"err-a-{aid}")
            if d.button("Deny", key=f"d_{aid}"):
                try:
                    decide(aid, "denied"); st.rerun()
                except Exception as e:
                    alert("block", str(e), key=f"err-d-{aid}")
