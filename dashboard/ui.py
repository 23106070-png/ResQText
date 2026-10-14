"""Shared look & feel: navy/white theme, status chips, report cards, Plotly template, helpers."""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import config
import re

import security
from ingestion.service import IngestionService
from nlp.pipeline import Pipeline

NAVY, INK, SLATE, LINE, PAPER = "#10233F", "#10233F", "#5B6B82", "#D9E0EA", "#F4F6FA"
CAT_COLORS = {
    "Rescue assistance": "#B3261E", "Medical assistance": "#D9822B", "Food and drinking water": "#2F7F8F",
    "Shelter and evacuation": "#5B6FD6", "Infrastructure damage": "#6B7785", "General information": "#9AA8BA",
}
STATUS_STYLE = {  # review status -> (label, text colour, background)
    "pending": ("Awaiting review", "#8A5A00", "#FFF4DB"), "approved": ("Verified by reviewer", "#1B6B47", "#DDF3E7"),
    "corrected": ("Corrected by reviewer", "#1F4E9C", "#E1EBFA"), "rejected": ("Marked incorrect", "#4A5565", "#E8ECF1"),
    "needs_info": ("Needs investigation", "#9B1C1C", "#FBE4E4"), "duplicate": ("Marked duplicate", "#4A5565", "#E8ECF1"),
}
VERIF_STYLE = {  # provenance badge: (short label, text colour, background)
    "human_verified": ("Verified by reviewer", "#1B6B47", "#DDF3E7"), "official_source": ("Official source", "#1F4E9C", "#E1EBFA"),
    "source_reported": ("Publisher-reported", "#5B4A9C", "#ECE7F8"), "unverified": ("Unverified", "#8A5A00", "#FFF4DB"),
    "incorrect": ("Marked incorrect", "#4A5565", "#E8ECF1"), "duplicate": ("Marked duplicate", "#4A5565", "#E8ECF1"),
    "investigating": ("Needs investigation", "#9B1C1C", "#FBE4E4"),
}
ORIGIN_STYLE = {"official_event": "#1F4E9C", "official_alert": "#1F4E9C", "situation_report": "#5B4A9C", "user_report": "#6B7785"}
LOC_STYLE = {"verified": ("#2E7D5B", "Verified (approx.)"), "manual": ("#1F4E9C", "Set by reviewer"),
             "geocoded": ("#2E7D5B", "Geocoded (approx.)"), "ambiguous": ("#B7791F", "Ambiguous"),
             "unverified": ("#B7791F", "Unverified"), "unknown": ("#8A97A8", "Unknown"),
             "source_centroid": ("#5B6FD6", "Event centroid, approx."), "source_area_text": ("#B7791F", "Area text, not mapped"),
             "source_country": ("#8A97A8", "Country only")}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea { font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp { background: #F4F6FA; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }
h1, h2, h3 { color: #10233F; letter-spacing: -0.01em; }
section[data-testid="stSidebar"] { background: #0B1B33; border-right: 0; }
section[data-testid="stSidebar"] * { color: #DCE5F2; }
section[data-testid="stSidebar"] a[aria-current="page"], section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] { background: rgba(255,255,255,.12); }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.14); }
.rq-brand { font-size: 1.25rem; font-weight: 600; color: #fff !important; letter-spacing: -0.01em; }
.rq-brand small { display:block; font-size:.74rem; font-weight:400; color:#9FB2CC !important; margin-top:2px; }
.rq-side-note { font-size: .78rem; line-height: 1.45; color: #9FB2CC !important; }
.rq-banner { background:#fff; border:1px solid #D9E0EA; border-left:4px solid #1F4E9C; border-radius:6px; padding:.55rem .8rem; font-size:.82rem; color:#33445E; margin-bottom:1rem; }
.rq-banner b { color:#10233F; }
.rq-title { font-size:1.65rem; font-weight:600; color:#10233F; margin:0; letter-spacing:-0.015em; }
.rq-sub { color:#5B6B82; font-size:.95rem; margin:.15rem 0 1.1rem 0; max-width: 70ch; }
.rq-section { font-size:1.02rem; font-weight:600; color:#10233F; margin:1.4rem 0 .5rem 0; }
.rq-kpi { background:#fff; border:1px solid #D9E0EA; border-radius:8px; padding:.85rem 1rem; height:100%; }
.rq-kpi .v { font-size:1.7rem; font-weight:600; color:#10233F; line-height:1.15; }
.rq-kpi .l { font-size:.82rem; color:#5B6B82; margin-top:2px; }
.rq-kpi.alert .v { color:#B3261E; }
.rq-card { background:#fff; border:1px solid #D9E0EA; border-left:5px solid #9AA8BA; border-radius:6px; padding:.7rem .9rem; margin-bottom:.55rem; }
.rq-card-top { display:flex; gap:.6rem; align-items:center; font-size:.76rem; color:#5B6B82; flex-wrap:wrap; margin-bottom:.3rem; }
.rq-id { font-weight:600; color:#10233F; }
.rq-text { font-size:.93rem; color:#1B2B45; line-height:1.45; max-width: 80ch; }
.rq-meta { display:flex; gap:.5rem; align-items:center; flex-wrap:wrap; margin-top:.5rem; font-size:.8rem; color:#44556E; }
.rq-chip { display:inline-flex; align-items:center; gap:.35rem; border:1px solid; border-radius:4px; padding:1px 7px; font-size:.76rem; font-weight:500; background:#fff; }
.rq-chip i { width:7px; height:7px; border-radius:50%; display:inline-block; }
.rq-pill { border-radius:10px; padding:1px 8px; font-size:.72rem; font-weight:500; }
.rq-sim { border:1px dashed #8A5A00; color:#8A5A00; background:#FFF9EB; border-radius:4px; padding:0 6px; font-size:.7rem; font-weight:500; }
.rq-dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:4px; }
.rq-bar { background:#E8EDF4; border-radius:3px; height:8px; width:100%; overflow:hidden; }
.rq-bar > div { height:8px; border-radius:3px; }
.rq-note { font-size:.82rem; color:#44556E; background:#EEF3FA; border-radius:6px; padding:.55rem .75rem; margin:.4rem 0; }
.rq-warn { font-size:.84rem; color:#6B4300; background:#FFF4DB; border:1px solid #F0D9A3; border-radius:6px; padding:.55rem .75rem; margin:.4rem 0; }
.rq-ev mark { background:#FFE8A3; color:#10233F; padding:0 2px; border-radius:2px; }
.rq-ev mark.loc { background:#CFE0FA; } .rq-ev mark.res { background:#CDEBDD; } .rq-ev mark.ppl { background:#E6D7F5; }
div[data-testid="stButton"] > button[kind="primary"] { background:#1F4E9C; border:0; }
div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button { border-radius:6px; font-weight:500; }
:focus-visible { outline: 2px solid #1F4E9C !important; outline-offset: 2px; }
@media (max-width: 760px) { .rq-title { font-size:1.35rem; } .block-container { padding-left:.8rem; padding-right:.8rem; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""

# ---- Plotly template ---------------------------------------------------------------------------------
pio.templates["resqtext"] = go.layout.Template(layout=go.Layout(
    font=dict(family="IBM Plex Sans, system-ui, sans-serif", color=INK, size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=36, b=10),
    colorway=list(CAT_COLORS.values()), xaxis=dict(gridcolor="#E3E9F1", zeroline=False, linecolor=LINE),
    yaxis=dict(gridcolor="#E3E9F1", zeroline=False, linecolor=LINE), legend=dict(orientation="h", y=-0.2),
    title=dict(font=dict(size=14, color=INK), x=0)))
pio.templates.default = "resqtext"


def style_fig(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(height=height, template="resqtext")
    return fig


def setup(title: str, subtitle: str = "") -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f"<h1 class='rq-title'>{html.escape(title)}</h1><p class='rq-sub'>{html.escape(subtitle)}</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='rq-banner'><b>Decision support only.</b> {html.escape(config.DISCLAIMER.split('. ', 1)[1] if '. ' in config.DISCLAIMER else config.DISCLAIMER)}</div>", unsafe_allow_html=True)


def sidebar_brand(pipeline: Pipeline) -> None:
    st.sidebar.markdown("<div class='rq-brand'>ResQText<small>Disaster message intelligence</small></div>", unsafe_allow_html=True)
    ok = pipeline.model_ready()
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<div class='rq-side-note'>Classifier: {'ready' if ok else 'not trained yet'}<br>"
                        f"Prototype. Not connected to live emergency services.</div>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_pipeline() -> Pipeline:
    return Pipeline()


def reviewer_name() -> str:
    return st.session_state.get("reviewer", "")


def require_model(p: Pipeline) -> None:
    """Stop the page with a clear next step if no trained model exists."""
    if p.model_ready():
        return
    st.markdown("<div class='rq-warn'><b>No trained classifier found.</b> Train the baseline on the bundled "
                "<b>synthetic</b> sample data to try the app. Results from synthetic data are not real-world performance.</div>", unsafe_allow_html=True)
    if st.button("Train baseline on sample data", type="primary"):
        with st.spinner("Training..."):
            from training.train_classifier import train
            train(verbose=False)
            p.reload_model()
        st.rerun()
    st.stop()


def kpi(col, value, label: str, alert: bool = False) -> None:
    col.markdown(f"<div class='rq-kpi {'alert' if alert else ''}'><div class='v'>{value}</div><div class='l'>{html.escape(label)}</div></div>", unsafe_allow_html=True)


def chip(label: str) -> str:
    c = CAT_COLORS.get(label, "#6B7785")
    return f"<span class='rq-chip' style='border-color:{c}55;color:#1B2B45'><i style='background:{c}'></i>{html.escape(label)}</span>"


def status_pill(status: str) -> str:
    label, fg, bg = STATUS_STYLE.get(status, (status, "#333", "#eee"))
    return f"<span class='rq-pill' style='color:{fg};background:{bg}'>{label}</span>"


def loc_badge(name: str, status: str) -> str:
    color, label = LOC_STYLE.get(status, ("#8A97A8", status))
    shown = html.escape(name) if status != "unknown" else "Unknown"
    return f"<span><span class='rq-dot' style='background:{color}'></span>{shown} <span style='color:#7A889C'>({label})</span></span>"


def fmt_time(ts) -> str:
    try:
        return datetime.fromisoformat(str(ts)).strftime("%d %b, %H:%M")
    except ValueError:
        return str(ts)


def report_card(row, show_conf: bool = True) -> str:
    labels = list(row["labels"]) if isinstance(row["labels"], (list, tuple)) else []
    edge = CAT_COLORS.get(labels[0], "#9AA8BA") if labels else "#9AA8BA"
    text = row["original_text"]
    text = html.escape(text if len(text) <= 230 else text[:227] + "...")
    sim = "<span class='rq-sim'>SIMULATED</span>" if row.get("is_simulated") else ""
    conf = ""
    if show_conf and pd.notna(row.get("top_confidence")):
        low = " &middot; low confidence" if row.get("low_confidence") else ""
        conf = f"<span>Model score {row['top_confidence']:.2f}{low}</span>"
    dup = "<span class='rq-pill' style='background:#E1EBFA;color:#1F4E9C'>Possible duplicate</span>" if row.get("has_dup") else ""
    return (f"<div class='rq-card' style='border-left-color:{edge}'><div class='rq-card-top'><span class='rq-id'>{row['id']}</span>"
            f"<span>{fmt_time(row['timestamp'])}</span>{sim}{status_pill(row['review_status'])}{dup}</div>"
            f"<div class='rq-text'>{text}</div><div class='rq-meta'>{''.join(chip(l) for l in labels)}"
            f"{loc_badge(row['location_text'], row['location_status'])}{conf}</div></div>")


def highlight(text: str, entities: list[dict]) -> str:
    """Original text with evidence spans marked. Offsets refer to the original string."""
    cls = {"LOCATION": "loc", "RESOURCE": "res", "PEOPLE": "ppl"}
    spans = sorted([(e["start"], e["end"], cls.get(e["type"], "")) for e in entities
                    if e["type"] in cls and 0 <= e["start"] < e["end"] <= len(text)])
    out, pos = [], 0
    for s, e, c in spans:
        if s < pos:
            continue
        out += [html.escape(text[pos:s]), f"<mark class='{c}'>{html.escape(text[s:e])}</mark>"]
        pos = e
    out.append(html.escape(text[pos:]))
    return f"<div class='rq-ev rq-text'>{''.join(out)}</div>"


def score_bars(scores: dict[str, float], assigned: list[str]) -> str:
    rows = []
    for l, s in sorted(scores.items(), key=lambda kv: -kv[1]):
        on = l in assigned
        c = CAT_COLORS.get(l, "#6B7785")
        rows.append(f"<div style='display:grid;grid-template-columns:190px 1fr 46px;gap:.6rem;align-items:center;margin:.28rem 0;opacity:{1 if on else .55}'>"
                    f"<span style='font-size:.85rem;font-weight:{600 if on else 400}'>{html.escape(l)}</span>"
                    f"<div class='rq-bar'><div style='width:{s * 100:.0f}%;background:{c}'></div></div>"
                    f"<span style='font-size:.82rem;text-align:right'>{s:.2f}</span></div>")
    return "".join(rows)


def section(title: str) -> None:
    st.markdown(f"<div class='rq-section'>{html.escape(title)}</div>", unsafe_allow_html=True)


def note(text: str, warn: bool = False) -> None:
    st.markdown(f"<div class='{'rq-warn' if warn else 'rq-note'}'>{text}</div>", unsafe_allow_html=True)


def empty_state(message: str, hint: str = "") -> None:
    st.markdown(f"<div class='rq-note'><b>{html.escape(message)}</b><br>{html.escape(hint)}</div>", unsafe_allow_html=True)


# ---- real-world additions -----------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_service() -> IngestionService:
    p = get_pipeline()
    svc = IngestionService(p.db, p)
    if config.AUTOPOLL_IN_DASHBOARD:  # optional; the DB lease lock prevents duplicate polling across reruns/processes
        from ingestion import scheduler
        scheduler.start_background(svc)
    return svc


def safe_url(url: str | None) -> str | None:
    """Only http(s) links are rendered as links (blocks javascript: and similar)."""
    return url if url and re.match(r"^https?://", url.strip(), re.I) else None


def link(url: str | None, text: str) -> str:
    u = safe_url(url)
    return f"<a href='{html.escape(u, quote=True)}' target='_blank' rel='noopener noreferrer'>{html.escape(text)}</a>" if u else html.escape(text)


def fmt_utc(ts) -> str:
    try:
        return datetime.fromisoformat(str(ts)).strftime("%d %b %Y, %H:%M UTC")
    except ValueError:
        return "unknown"


def is_admin() -> bool:
    mode = security.dashboard_mode()
    return mode == "open_local" or (mode == "secret" and st.session_state.get("admin_ok", False))


def admin_panel() -> None:
    """Sidebar sign-in. Reading is public; reviewing, refreshing and deleting need this."""
    mode = security.dashboard_mode()
    st.sidebar.markdown("---")
    if mode == "open_local":
        st.sidebar.markdown("<div class='rq-side-note'><b>Local mode:</b> no login. Set RESQTEXT_ADMIN_PASSCODE before sharing this app.</div>", unsafe_allow_html=True)
    elif mode == "locked":
        st.sidebar.markdown("<div class='rq-side-note'><b>Read-only.</b> Review and refresh are disabled until RESQTEXT_ADMIN_PASSCODE is configured.</div>", unsafe_allow_html=True)
    elif st.session_state.get("admin_ok"):
        st.sidebar.markdown("<div class='rq-side-note'>Reviewer signed in.</div>", unsafe_allow_html=True)
        if st.sidebar.button("Sign out"):
            st.session_state["admin_ok"] = False
            st.rerun()
    else:
        with st.sidebar.form("admin_login"):
            code = st.text_input("Reviewer passcode", type="password")
            if st.form_submit_button("Sign in") and security.check_passcode(code):
                st.session_state["admin_ok"] = True
                st.rerun()
            elif code:
                st.error("Incorrect passcode.")


def require_admin(action: str = "This action") -> bool:
    if is_admin():
        return True
    note(f"<b>{html.escape(action)} needs a signed-in reviewer.</b> Use the sign-in box in the sidebar"
         + (" (read-only until a passcode is configured)." if security.dashboard_mode() == "locked" else "."), warn=True)
    return False


def origin_badge(origin: str) -> str:
    from ingestion.models import ORIGIN_LABELS
    c = ORIGIN_STYLE.get(origin, "#6B7785")
    return f"<span class='rq-chip' style='border-color:{c}55;color:{c}'>{html.escape(ORIGIN_LABELS.get(origin, origin))}</span>"


def verif_badge(key: str) -> str:
    label, fg, bg = VERIF_STYLE.get(key, (key, "#333", "#eee"))
    return f"<span class='rq-pill' style='color:{fg};background:{bg}'>{label}</span>"


def update_card(row) -> str:
    """Card for an imported item. Observed source data and model output are visually separated."""
    from reports import NLP_STATUS_LABELS, LOCATION_PRECISION_LABELS, excerpt
    title = html.escape(row["title"] or row["original_text"][:90])
    exp = " <span class='rq-pill' style='background:#E8ECF1;color:#4A5565'>Expired</span>" if row.get("is_expired") else ""
    cats = row["labels"] if isinstance(row["labels"], (list, tuple)) else []
    nlp_line = ""
    if cats:
        nlp_line = ("<div class='rq-meta'><span style='color:#5B6B82'>NLP-inferred (not confirmed):</span>"
                    + "".join(chip(l) for l in cats) + "</div>")
    summ = html.escape(excerpt(row.get("summary") or row["original_text"], 260))
    prec = LOCATION_PRECISION_LABELS.get(row.get("coord_precision"), "")
    loc = loc_badge(row["location_text"], row["location_status"]) + (f" <span style='color:#7A889C'>{prec}</span>" if row.get("coord_precision") else "")
    return (f"<div class='rq-card' style='border-left-color:{ORIGIN_STYLE.get(row['origin'], '#9AA8BA')}'>"
            f"<div class='rq-card-top'>{origin_badge(row['origin'])}{verif_badge(row['verification'])}{exp}"
            f"<span class='rq-id'>{row['id']}</span></div>"
            f"<div class='rq-text'><b>{title}</b></div><div class='rq-text' style='color:#44556E;margin-top:.2rem'>{summ}</div>"
            f"<div class='rq-meta'>{loc}"
            + (f"<span>{html.escape(row['event_type'])}</span>" if row.get("event_type") else "")
            + (f"<span>Severity: {html.escape(str(row['severity']))}</span>" if row.get("severity") else "") + "</div>"
            f"{nlp_line}"
            f"<div class='rq-meta' style='font-size:.76rem;color:#6B7B91'>Source: {link(row['source_url'], row['source_name'] or 'source')}"
            f" &middot; published {fmt_utc(row['published_at'])} &middot; retrieved {fmt_utc(row['retrieved_at'])}"
            f" &middot; processing: {html.escape(NLP_STATUS_LABELS.get(row['nlp_status'], row['nlp_status']))}</div></div>")
