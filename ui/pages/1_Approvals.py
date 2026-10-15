import requests
import streamlit as st

st.set_page_config(page_title="Approvals", layout="wide")
BASE_URL = "http://localhost:8000"

st.title("Approvals")
approver = st.text_input("Your name (HR approver)", value="priya")
st.caption("Requests waiting for an HR decision.")
if st.button("Refresh"):
    st.rerun()

def decide(approval_id, decision):
    r = requests.post(f"{BASE_URL}/approvals/{approval_id}",
                      json={"approver": approver.strip(), "role": "hr", "decision": decision},
                      timeout=15)
    r.raise_for_status()

try:
    r = requests.get(f"{BASE_URL}/approvals", params={"status": "pending"}, timeout=15)
    r.raise_for_status()
    items = r.json()
except Exception as e:
    st.error(f"Gateway not reachable: {e}")
    st.stop()

if isinstance(items, dict):
    items = items.get("approvals", [])

if not items:
    st.success("Nothing pending.")

for it in items:
    aid = it.get("id")
    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"**{it.get('user', '?')}** ({it.get('role', '?')}) · "
                        f"{', '.join(it.get('categories', [])) if isinstance(it.get('categories'), list) else it.get('categories', '')}")
            st.caption(it.get("user_message") or it.get("reason") or "")
        with c2:
            a, d = st.columns(2)
            if a.button("Approve", key=f"a_{aid}", disabled=not approver.strip()):
                try:
                    decide(aid, "approve"); st.rerun()
                except Exception as e:
                    st.error(str(e))
            if d.button("Deny", key=f"d_{aid}"):
                try:
                    decide(aid, "deny"); st.rerun()
                except Exception as e:
                    st.error(str(e))