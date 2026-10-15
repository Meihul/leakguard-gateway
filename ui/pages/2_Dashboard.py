import sqlite3
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from theme import ACCENT, STATUS, alert, apply_theme, label, style_chart

apply_theme()

# Role comes from the login, not a selector.
if st.session_state.get("role") not in ("hr", "manager"):
    st.title("LeakGuard: Security Dashboard")
    alert("mask", "Only HR and managers can view this page.", key="no-access")
    st.stop()

DB_PATH = Path(__file__).resolve().parent.parent.parent / "audit.db"

LABELS = {"aadhaar": "Aadhaar", "pan": "PAN", "phone": "Phone", "card": "Card number",
          "api_key": "API key", "password": "Password", "injection": "Prompt injection",
          "restricted_doc": "Restricted doc"}
# Actions in status colours. Red/green and red/amber are hard to tell apart
# with red-green colour blindness, so colour is never the only cue: blocking
# actions are solid bars, the others outlined with a distinct border style.
# Each action keeps the same colour and style whichever subset is shown.
ACTIONS = ["block", "approve", "mask", "confirm", "warn", "allow"]
SOLID = {"block", "approve"}
_DASH = {"confirm": [5, 3], "warn": [6, 2, 2, 2], "allow": [2, 2]}
_LEGEND_LABELS = {"block": "Blocked · solid", "approve": "Approval · solid",
                  "mask": "Masked · outline", "confirm": "Confirm · dashed",
                  "warn": "Warn · dash-dot", "allow": "Allowed · dotted"}
LEGEND_LABEL_EXPR = ("{" + ", ".join(f"'{k}': '{v}'" for k, v in _LEGEND_LABELS.items())
                     + "}[datum.label] || datum.label")


def action_scales(present):
    domain = [a for a in ACTIONS if a in present] + sorted(set(present) - set(ACTIONS))
    return (alt.Scale(domain=domain, range=[STATUS.get(a, "#8B93A1") for a in domain]),
            alt.Scale(domain=domain, range=[_DASH.get(a, [1, 0]) for a in domain]))

# ---- load data from the audit log ----
if not DB_PATH.exists():
    alert("approve", "No audit.db yet. Run: python eval/seed_audit.py", key="no-db")
    st.stop()

with sqlite3.connect(DB_PATH) as con:
    ev = pd.read_sql_query("SELECT * FROM events", con)
ev["ts"] = pd.to_datetime(ev["ts"])
source = "live (audit.db)"

# ---- sidebar filters (these filter the audit data; they don't change who you are) ----
with st.sidebar:
    label("Filter: role")
    role_pick = st.radio("Filter: role", ["All", "Intern", "Manager", "HR"],
                        label_visibility="collapsed")
    label("Filter: action")
    action_pick = st.radio("Filter: action", ["All", "Masked", "Blocked"],
                           label_visibility="collapsed")
    st.caption(f"Data: {source}")
    st.caption("Logs store category, rule, action and a prompt hash only. "
              "Raw prompts are never saved.")

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
        colors, dash = action_scales(set(counts["action"]))
        # Vega-Lite only honours per-bar fill opacity from a numeric field.
        counts["fill_opacity"] = counts["action"].map(lambda a: 1.0 if a in SOLID else 0.18)
        legend = alt.Legend(title=None, orient="top", direction="horizontal",
                            labelFontSize=12, symbolType="square", symbolStrokeWidth=1.5,
                            labelExpr=LEGEND_LABEL_EXPR)
        bars = (alt.Chart(counts)
                .mark_bar(cornerRadiusEnd=3, height=22, strokeWidth=1.5)
                .encode(
                    y=alt.Y("label:N", sort=order, title=None,
                            axis=alt.Axis(labelFontSize=13, labelColor="#E6E8EB",
                                          domain=False, ticks=False)),
                    x=alt.X("count:Q", title=None,
                            axis=alt.Axis(grid=True, tickMinStep=1, domain=False)),
                    color=alt.Color("action:N", scale=colors, legend=legend),
                    stroke=alt.Stroke("action:N", scale=colors, legend=legend),
                    fillOpacity=alt.FillOpacity("fill_opacity:Q", scale=None, legend=None),
                    strokeDash=alt.StrokeDash("action:N", scale=dash, legend=legend),
                    tooltip=["label", "action", "count"])
                .properties(height=max(220, 38 * len(order))))
        st.altair_chart(style_chart(bars), width="stretch", theme=None)
    else:
        st.caption("Nothing caught for this filter.")

with right:
    st.subheader("Incidents over time")
    if total:
        trend = ev_f.set_index("ts").resample("10min").size().reset_index(name="count")
        area = (alt.Chart(trend)
                .mark_area(line={"color": ACCENT, "strokeWidth": 2}, color=ACCENT, opacity=0.15)
                .encode(x=alt.X("ts:T", title=None, axis=alt.Axis(grid=False, format="%H:%M")),
                        y=alt.Y("count:Q", title=None, axis=alt.Axis(tickMinStep=1, grid=True)),
                        tooltip=["ts", "count"])
                .properties(height=266))
        st.altair_chart(style_chart(area), width="stretch", theme=None)

# ---- recent incidents ----
st.subheader("Recent incidents")
cols = ["ts", "user", "role", "action", "categories", "masked_snippets"]
recent = ev_f.sort_values("ts", ascending=False).head(15)[cols]
st.dataframe(recent, width="stretch", hide_index=True)
