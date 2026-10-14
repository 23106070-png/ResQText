import html

import pandas as pd
import streamlit as st

import config
import reports as rp
from dashboard import ui

HEALTH = {"ok": ("Current", "#1B6B47", "#DDF3E7"), "stale": ("Stale", "#8A5A00", "#FFF4DB"), "failing": ("Failing", "#9B1C1C", "#FBE4E4"),
          "untested": ("Not tested yet", "#4A5565", "#E8ECF1"), "not_configured": ("Not configured", "#4A5565", "#E8ECF1")}


def render():
    svc = ui.get_service()
    ui.setup("Data Sources and System Health", "Which integrations are configured, when they last worked, and whether the data on screen is current.")
    ui.note("A source is shown as <b>connected</b> only after at least one successful fetch recorded in this database. "
            "Sources refresh periodically, not continuously: this is <b>not</b> a real-time feed.")
    statuses = svc.source_status()
    admin = ui.is_admin()

    top = st.columns([1, 1, 3])
    if top[0].button("Refresh all sources", type="primary", disabled=not admin):
        with st.spinner("Fetching..."):
            res = svc.refresh(trigger="manual", force=True)
        for r in res:
            (st.success if r["status"] in ("success", "partial") else st.warning if r["status"] == "skipped" else st.error)(
                f"{r['source']}: {r['status']}. {r.get('message') or ''} (new {r.get('inserted', 0)}, updated {r.get('updated', 0)})")
    if not admin:
        top[2].caption("Refreshing needs a signed-in reviewer (sidebar).")

    for s in statuses:
        label, fg, bg = HEALTH[s["health"]]
        age = rp.freshness_text(s["age_minutes"])
        with st.container(border=True):
            h1, h2 = st.columns([4, 1])
            h1.markdown(f"<b>{html.escape(s['label'])}</b> &nbsp;<span class='rq-pill' style='color:{fg};background:{bg}'>{label}</span>"
                        f"<div style='color:#5B6B82;font-size:.85rem'>{html.escape(s['kind'])} &middot; {ui.link(s['homepage'], s['homepage'])}</div>", unsafe_allow_html=True)
            if h2.button("Refresh", key=f"r_{s['name']}", disabled=not (admin and s["configured"])):
                r = svc.run_source(s["name"], "manual", force=True)
                st.info(f"{r['status']}: {r.get('message') or ''} (new {r.get('inserted', 0)}, updated {r.get('updated', 0)})")
            m = st.columns(4)
            m[0].metric("Last successful fetch", age, help=s["last_success_at"] and f"{s['last_success_at']} UTC")
            m[1].metric("Items stored", s["items"])
            m[2].metric("Checks every", f"{s['interval_minutes']:.0f} min")
            m[3].metric("Calls today", s["api_calls_today"])
            if not s["configured"]:
                st.warning(s["config_note"] + (" " + s["credential_help"] if s["credential_help"] else ""))
            elif s["health"] == "untested":
                st.info("Configured, but no successful fetch has been recorded yet. Not shown as connected.")
            elif s["health"] == "stale":
                st.warning(f"Data is stale: last success {age}. Items below from this source may be out of date.")
            elif s["health"] == "failing":
                st.error("The latest attempt failed" + (f": {s['last_error']}" if admin and s["last_error"] else ". Sign in as a reviewer to see details.")
                         + (f" Showing data from {age}." if s["connected"] else ""))
            if s["backoff_until"]:
                st.caption(f"Backing off until {s['backoff_until']} UTC after repeated errors.")
            st.caption("Attribution: " + s["attribution"])

    ui.section("Recent fetch runs")
    runs = svc.db.recent_runs(limit=15)
    if runs:
        st.dataframe(pd.DataFrame(runs)[["started_at", "source", "trigger", "status", "fetched", "inserted", "updated", "unchanged", "held_back", "errors", "api_calls", "message"]]
                     .rename(columns={"started_at": "Started (UTC)"}), hide_index=True, use_container_width=True)
    else:
        st.caption("No runs yet.")
    ui.section("Latest errors")
    errs = svc.db.recent_errors(limit=15)
    if not errs:
        st.caption("No errors recorded.")
    elif admin:
        st.dataframe(pd.DataFrame(errs)[["occurred_at", "source", "kind", "message", "item_ref"]], hide_index=True, use_container_width=True)
    else:
        st.caption(f"{len(errs)} recent error(s) recorded. Details are visible to signed-in reviewers only.")

    with st.expander("Configuration (names only, never values)"):
        st.write({"RELIEFWEB_APPNAME": "set" if config.RELIEFWEB_APPNAME else "not set",
                  "RESQTEXT_ENV": config.APP_ENV, "Automatic polling in dashboard": config.AUTOPOLL_IN_DASHBOARD,
                  "Automatic polling in API": config.AUTOPOLL_IN_API,
                  "Separate poller": "python -m ingestion.scheduler (recommended)"})
