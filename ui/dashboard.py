import json
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="LeakGuard Dashboard", layout="wide")


LOG_DB = Path("data/audit.db")    
LOG_TABLE = "audit"              
LOG_JSONL = Path("data/audit.jsonl") 

NEVER_SHOW = {"span", "value", "prompt", "text", "raw", "masked_prompt"}

def demo_data(n=120):
    cats = ["aadhaar", "pan", "phone", "card", "api_key", "password", "injection"]
    rules = {"aadhaar": "pii.aadhaar", "pan": "pii.pan", "phone": "pii.phone",
             "card": "pii.card", "api_key": "secrets.key", "password": "secrets.password",
             "injection": "injection.rules"}
    now = datetime.now()
    rows = []
    for i in range(n):
        c = random.choice(cats)
        action = "block" if c in ("api_key", "password", "injection") else "mask"
        rows.append({"ts": now - timedelta(minutes=random.randint(0, 180)),
                     "user": random.choice(["riya", "arjun", "meera", "kabir"]),
                     "role": random.choice(["intern", "manager", "hr"]),
                     "category": c, "rule": rules[c], "action": action,
                     "latency_ms": round(random.uniform(8, 40), 1)})
    return pd.DataFrame(rows)

def load_log():
    if LOG_DB.exists():
        try:
            with sqlite3.connect(LOG_DB) as con:
                return pd.read_sql_query(f"SELECT * FROM {LOG_TABLE}", con), "live (SQLite)"
        except Exception:
            pass
    if LOG_JSONL.exists():
        try:
            rows = [json.loads(l) for l in LOG_JSONL.read_text().splitlines() if l.strip()]
            return pd.DataFrame(rows), "live (JSONL)"
        except Exception:
            pass
    return demo_data(), "DEMO DATA"

df, source = load_log()
df = df.drop(columns=[c for c in df.columns if c in NEVER_SHOW]) 
df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

st.title("LeakGuard: Security Dashboard")
if st.button("🔄 Refresh"):
    st.rerun()

# ---- sidebar ----
st.markdown("""
<style>
[data-testid="stSidebar"] { background:#0E1117; border-right:1px solid #1F2430; }
[data-testid="stSidebar"] .block-container { padding-top:1.5rem; }
.lg-brand { font-size:1.35rem; font-weight:700; letter-spacing:.3px; margin:0; }
.lg-sub   { font-size:.8rem; color:#8B93A7; margin:0 0 1.2rem 0; }
.lg-label { font-size:.7rem; font-weight:600; letter-spacing:1.2px; color:#8B93A7;
            text-transform:uppercase; margin:1.2rem 0 .3rem 0; }
.lg-pill  { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem;
            border:1px solid #2A3142; color:#C5CBD9; }
.lg-foot  { font-size:.72rem; color:#6B7388; line-height:1.4; margin-top:2rem;
            border-top:1px solid #1F2430; padding-top:.8rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<p class="lg-brand">LeakGuard</p>'
                '<p class="lg-sub">AI security gateway</p>', unsafe_allow_html=True)

    st.markdown('<div class="lg-label">Role</div>', unsafe_allow_html=True)
    role_pick = st.radio("Role", ["All", "Intern", "Manager", "HR"],
                         label_visibility="collapsed")

    st.markdown('<div class="lg-label">Action</div>', unsafe_allow_html=True)
    action_pick = st.radio("Action", ["All", "Masked", "Blocked"],
                           label_visibility="collapsed")

    st.markdown('<div class="lg-label">Data</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="lg-pill">{source}</span>', unsafe_allow_html=True)

    st.markdown('<div class="lg-foot">Logs store category, rule and action only. '
                'Raw values are never saved or shown.</div>', unsafe_allow_html=True)

view = df.copy()
if role_pick != "All":
    view = view[view["role"].str.lower() == role_pick.lower()]
if action_pick != "All":
    view = view[view["action"] == {"Masked": "mask", "Blocked": "block"}[action_pick]]

total = len(view)
masked = int((view["action"] == "mask").sum())
blocked = int((view["action"] == "block").sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total incidents", total)
c2.metric("Masked", masked)
c3.metric("Blocked", blocked)
if "latency_ms" in view.columns and total:
    c4.metric("Avg added latency", f"{view['latency_ms'].mean():.1f} ms")
else:
    c4.metric("Block rate", f"{(blocked / total * 100):.0f}%" if total else "n/a")



LABELS = {"aadhaar": "Aadhaar", "pan": "PAN", "phone": "Phone", "card": "Card number",
          "api_key": "API key", "password": "Password", "injection": "Prompt injection"}
COLORS = alt.Scale(domain=["mask", "block"], range=["#F2A93B", "#E5484D"])

st.divider() 

# ---- charts ----
left, right = st.columns(2)

with left:
    st.subheader("What we caught")
    if total:
        counts = (view.assign(label=view["category"].map(LABELS).fillna(view["category"]))
                      .groupby(["label", "action"]).size().reset_index(name="count"))
        order = counts.groupby("label")["count"].sum().sort_values(ascending=False).index.tolist()
        bars = (alt.Chart(counts)
                .mark_bar(cornerRadiusEnd=3, height=22)
                .encode(
                    y=alt.Y("label:N", sort=order, title=None,
                            axis=alt.Axis(labelFontSize=13, domain=False, ticks=False)),
                    x=alt.X("count:Q", title=None,
                            axis=alt.Axis(grid=False, tickMinStep=1, domain=False)),
                    color=alt.Color("action:N", scale=COLORS,
                                    legend=alt.Legend(title=None, orient="top-right")),
                    tooltip=["label", "action", "count"])
                .properties(height=max(220, 38 * len(order))))
        st.altair_chart(bars, use_container_width=True)

with right:
    st.subheader("Incidents over time")
    if total:
        trend = view.set_index("ts").resample("15min").size().reset_index(name="count")
        line = (alt.Chart(trend)
                .mark_area(line={"color": "#5B9CF6"}, color="#5B9CF6", opacity=0.15)
                .encode(x=alt.X("ts:T", title=None, axis=alt.Axis(grid=False, format="%H:%M")),
                        y=alt.Y("count:Q", title=None, axis=alt.Axis(tickMinStep=1)),
                        tooltip=["ts", "count"])
                .properties(height=max(220, 38 * 7)))
        st.altair_chart(line, use_container_width=True)

st.subheader("Recent incidents")
cols = [c for c in ["ts", "user", "role", "category", "rule", "action"] if c in view.columns]
recent = view.sort_values("ts", ascending=False).head(15)[cols].copy()
recent["why"] = recent["category"] + " detected by " + recent["rule"] + " → " + recent["action"]
st.dataframe(recent, use_container_width=True, hide_index=True)