import sqlite3
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="LeakGuard Dashboard", layout="wide")

DB_PATH = Path(__file__).resolve().parent.parent / "audit.db"

LABELS = {"aadhaar": "Aadhaar", "pan": "PAN", "phone": "Phone", "card": "Card number",
          "api_key": "API key", "password": "Password", "injection": "Prompt injection",
          "restricted_doc": "Restricted doc"}
COLORS = alt.Scale(domain=["mask", "block"], range=["#3FB8AF", "#E5484D"])

# ---- load data from the audit log ----
if not DB_PATH.exists():
    st.info("No audit.db yet. Run: python eval/seed_audit.py")
    st.stop()

with sqlite3.connect(DB_PATH) as con:
    ev = pd.read_sql_query("SELECT * FROM events", con)
ev["ts"] = pd.to_datetime(ev["ts"])
source = "live (audit.db)"

# ---- sidebar ----
st.markdown("""
<style>
[data-testid="stSidebar"] { background:#0E1117; border-right:1px solid #1F2430; }
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
    st.markdown('<p class="lg-brand">LeakGuard</p><p class="lg-sub">AI security gateway</p>',
                unsafe_allow_html=True)
    st.markdown('<div class="lg-label">Role</div>', unsafe_allow_html=True)
    role_pick = st.radio("Role", ["All", "Intern", "Manager", "HR"], label_visibility="collapsed")
    st.markdown('<div class="lg-label">Action</div>', unsafe_allow_html=True)
    action_pick = st.radio("Action", ["All", "Masked", "Blocked"], label_visibility="collapsed")
    st.markdown('<div class="lg-label">Data</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="lg-pill">{source}</span>', unsafe_allow_html=True)
    st.markdown('<div class="lg-foot">Logs store category, rule, action and a prompt hash only. '
                'Raw prompts are never saved.</div>', unsafe_allow_html=True)

# ---- filters ----
ev_f = ev.copy()
if role_pick != "All":
    ev_f = ev_f[ev_f["role"].str.lower() == role_pick.lower()]
if action_pick != "All":
    ev_f = ev_f[ev_f["action"] == {"Masked": "mask", "Blocked": "block"}[action_pick]]

# one row per finding, for the category chart
view = (ev_f.assign(category=ev_f["categories"].fillna("").str.split(","))
            .explode("category"))
view = view[view["category"].fillna("") != ""]

total = len(ev_f)
masked = int((ev_f["action"] == "mask").sum())
blocked = int((ev_f["action"] == "block").sum())

# ---- header + top numbers ----
st.title("LeakGuard: Security Dashboard")
if st.button("Refresh"):
    st.rerun()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total requests", total)
c2.metric("Masked", masked)
c3.metric("Blocked", blocked)
c4.metric("% blocked", f"{blocked / total * 100:.0f}%" if total else "n/a")
c5.metric("Avg added delay", f"{ev_f['latency_ms'].mean():.1f} ms" if total else "n/a")

st.divider()

# ---- charts ----
left, right = st.columns(2)

with left:
    st.subheader("What we caught")
    if len(view):
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
                                    legend=alt.Legend(title=None, orient="top",
                                                      direction="horizontal",
                                                      labelFontSize=12, symbolType="circle")),
                    tooltip=["label", "action", "count"])
                .properties(height=max(220, 38 * len(order))))
        st.altair_chart(bars, width="stretch")
    else:
        st.caption("Nothing caught for this filter.")

with right:
    st.subheader("Incidents over time")
    if total:
        trend = ev_f.set_index("ts").resample("10min").size().reset_index(name="count")
        area = (alt.Chart(trend)
                .mark_area(line={"color": "#5B9CF6"}, color="#5B9CF6", opacity=0.15)
                .encode(x=alt.X("ts:T", title=None, axis=alt.Axis(grid=False, format="%H:%M")),
                        y=alt.Y("count:Q", title=None, axis=alt.Axis(tickMinStep=1)),
                        tooltip=["ts", "count"])
                .properties(height=266))
        st.altair_chart(area, width="stretch")

# ---- recent incidents ----
st.subheader("Recent incidents")
cols = ["ts", "user", "role", "action", "categories", "masked_snippets"]
recent = ev_f.sort_values("ts", ascending=False).head(15)[cols]
st.dataframe(recent, width="stretch", hide_index=True)