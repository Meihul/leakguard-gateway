"""LeakGuard's dark "security console" look for the Streamlit UI.

apply_theme() is called at the top of every page. Everything here is
appearance only. Status colours are used ONLY for small outlined pill
badges, the 3px left border of message/alert cards, and charts that show
actions; each is always paired with a text label, never colour alone.

All text/background pairs below meet WCAG AA (>= 4.5:1); the lowest is
blocked #F85149 on card #12151A at 5.46:1, muted #8B93A1 on card at 5.91:1.
"""
from html import escape

import streamlit as st

BG = "#0B0D10"
CARD = "#12151A"
SIDEBAR = "#090B0E"
BORDER = "#232830"
TEXT = "#E6E8EB"
MUTED = "#8B93A1"
ACCENT = "#4CC2D6"

STATUS = {
    "allow": "#3FB950",
    "mask": "#D29922",
    "warn": "#D29922",
    "confirm": "#D29922",
    "block": "#F85149",
    "approve": "#58A6FF",
}

PILL_LABELS = {
    "allow": "✓ Allowed",
    "mask": "● Masked",
    "block": "✕ Blocked",
    "confirm": "! Needs confirmation",
    "approve": "◷ Sent for approval",
}

_ALERT_FUNCS = {"allow": "success", "mask": "warning", "warn": "warning",
                "confirm": "warning", "block": "error", "approve": "info"}

SHIELD_SVG = (
    '<svg width="40" height="40" viewBox="0 0 24 24" fill="none" aria-hidden="true" '
    f'stroke="{ACCENT}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 3l7 3v5c0 4.5-3 8.3-7 10-4-1.7-7-5.5-7-10V6l7-3z"/>'
    '<path d="M9 12l2 2 4-4"/></svg>'
)

_FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500"
          "&family=IBM+Plex+Sans:wght@400;500;600&display=swap")

_status_rules = "\n".join(
    f"""[data-testid="stChatMessage"]:has(.lg-pill-{k}) {{ border-left: 3px solid {c} !important; }}
.lg-pill-{k} {{ color: {c}; }}
[class*="st-key-lgalert-{k}"] [data-testid="stAlertContainer"] {{ border-left-color: {c} !important; }}"""
    for k, c in STATUS.items())

_CSS = f"""
<style>
@import url('{_FONTS}');

/* ---- base: fonts, background ---- */
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp th, .stApp td,
[data-testid="stSidebar"] {{
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
}}
.stApp code, .stApp pre, .lg-mono, .lg-meta, .lg-finding,
[data-testid="stCode"] code, [data-testid="stCode"] pre {{
    font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
}}
.stApp {{
    background: radial-gradient(ellipse 120% 55% at 50% 0%, #121821 0%, {BG} 70%) fixed;
    color: {TEXT};
}}
[data-testid="stHeader"] {{ background: transparent; }}
.stApp h1 {{ font-weight: 600; letter-spacing: -0.01em; }}
.stApp h2, .stApp h3 {{ font-weight: 500; }}
/* Streamlit fades captions to 60% opacity, which would drop muted text below
   WCAG AA; keep it at full opacity in the muted colour instead (5.9:1 on cards). */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{
    color: {MUTED} !important; opacity: 1 !important;
}}

/* ---- hide Streamlit chrome ---- */
[data-testid="stAppDeployButton"], footer {{ display: none !important; }}

/* ---- sidebar ---- */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {{ background: {SIDEBAR}; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}

/* ---- cards: chat messages, expanders, forms, metrics, bordered containers ---- */
[data-testid="stChatMessage"] {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 0.85rem 1rem;
}}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {{
    background: {BORDER} !important; color: {MUTED} !important;
}}
[data-testid="stExpander"] details {{
    background: {CARD}; border: 1px solid {BORDER} !important; border-radius: 10px;
}}
[data-testid="stForm"], [class*="st-key-lgcard"] {{
    background: {CARD}; border: 1px solid {BORDER} !important; border-radius: 10px;
}}
[data-testid="stMetric"] {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 14px 16px;
}}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{
    font-family: 'IBM Plex Sans', system-ui, sans-serif; font-size: clamp(1.4rem, 2vw, 2rem);
    font-weight: 500; color: {TEXT};
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{
    color: {MUTED} !important; font-size: 0.8rem;
}}

/* ---- alerts: dark card, coloured 3px left border (colour from lgalert-<status> key) ---- */
[data-testid="stAlertContainer"] {{
    background: {CARD} !important; border: 1px solid {BORDER} !important;
    border-left: 3px solid {MUTED} !important; border-radius: 10px !important;
}}
[data-testid="stAlertContainer"], [data-testid="stAlertContainer"] p {{ color: {TEXT} !important; }}

/* ---- inputs: darker than the cards they sit on, with a visible border ---- */
[data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"] {{
    background: {BG} !important; border: 1px solid {BORDER} !important; border-radius: 8px;
}}
[data-testid="stTextInputRootElement"] input, [data-testid="stTextAreaRootElement"] textarea {{
    background: transparent !important; color: {TEXT} !important;
}}
[data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stTextAreaRootElement"]:focus-within {{
    border-color: {ACCENT} !important;
}}
[data-testid="stChatInput"] > div {{
    background: {CARD} !important; border: 1px solid {BORDER} !important; border-radius: 10px;
}}

/* ---- status pills ---- */
.lg-pill {{
    display: inline-block; padding: 1px 10px; border: 1px solid currentColor;
    border-radius: 999px; font-size: 0.75rem; font-weight: 500; letter-spacing: 0.02em;
    line-height: 1.5; background: transparent;
}}
{_status_rules}

/* ---- small text styles ---- */
.lg-meta {{ color: {MUTED}; font-size: 0.78rem; }}
.lg-finding {{ color: {TEXT}; font-size: 0.82rem; margin: 0.25rem 0; }}
.lg-finding .lg-dim {{ color: {MUTED}; }}
.lg-label {{
    color: {MUTED}; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; margin: 1.2rem 0 0.3rem 0;
}}
.lg-received {{
    background: {BG}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 8px 12px; margin-top: 0.5rem;
}}
.lg-received .lg-label {{ margin: 0 0 0.25rem 0; }}
.lg-received .lg-mono {{ color: {TEXT}; font-size: 0.82rem; white-space: pre-wrap; }}

/* ---- buttons: outlined accent, filled on hover ---- */
button[data-testid="stBaseButton-secondary"], button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-secondaryFormSubmit"],
button[data-testid="stBaseButton-primaryFormSubmit"] {{
    background: transparent !important; border: 1px solid {ACCENT} !important;
    color: {ACCENT} !important; border-radius: 8px;
}}
button[data-testid^="stBaseButton-"]:is([data-testid$="secondary"], [data-testid$="primary"],
  [data-testid$="FormSubmit"]):hover {{
    background: {ACCENT} !important; color: {BG} !important;
}}
button[data-testid^="stBaseButton-"] p {{ color: inherit !important; }}

/* ---- login card ---- */
.st-key-lg-login-card {{
    max-width: 380px; margin: 10vh auto 0 auto; background: {CARD};
    border: 1px solid {BORDER}; border-radius: 10px; padding: 28px 28px 20px 28px;
}}
.st-key-lg-login-card [data-testid="stForm"] {{
    background: transparent; border: none !important; padding: 0;
}}
.lg-brand {{ text-align: center; margin-bottom: 0.5rem; }}
.lg-brand .lg-title {{ font-size: 1.6rem; font-weight: 600; color: {TEXT}; margin-top: 0.4rem; }}
.lg-brand .lg-tagline {{ color: {MUTED}; font-size: 0.9rem; }}

/* ---- attached-file chip ---- */
.st-key-lg-chip {{
    max-width: 440px; background: {CARD}; border: 1px solid {BORDER};
    border-radius: 999px; padding: 2px 6px 2px 14px;
}}
.st-key-lg-chip p {{ font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 0.82rem; }}
.st-key-lg-chip button {{ border: none !important; min-height: 0; padding: 0 8px; }}
</style>
"""


def apply_theme():
    st.markdown(_CSS, unsafe_allow_html=True)


def pill(action):
    """Small outlined status badge, e.g. pill("block")."""
    label = PILL_LABELS.get(action, action)
    return f'<span class="lg-pill lg-pill-{escape(action)}">{escape(label)}</span>'


def alert(status, text, key):
    """st.success/warning/error/info, restyled as a dark card whose left
    border has the status colour. key must be unique on the page."""
    with st.container(key=f"lgalert-{status}-{key}"):
        getattr(st, _ALERT_FUNCS.get(status, "info"))(text)


def label(text):
    st.markdown(f'<div class="lg-label">{escape(text)}</div>', unsafe_allow_html=True)


def meta(text):
    """Muted monospace line (timings, IDs)."""
    st.markdown(f'<div class="lg-meta">{escape(text)}</div>', unsafe_allow_html=True)


def style_chart(chart):
    """Dark-console styling for an Altair chart: grey axes and gridlines."""
    return (chart.configure(background="transparent", font="IBM Plex Sans")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=BORDER,
                            domainColor=BORDER, tickColor=BORDER, labelFontSize=12)
            .configure_legend(labelColor=MUTED, titleColor=MUTED, labelFontSize=12))
