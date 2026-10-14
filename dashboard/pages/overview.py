import plotly.express as px
import streamlit as st

import analytics
import config
import reports as rp
from dashboard import ui


def render():
    p = ui.get_pipeline()
    svc = ui.get_service()
    ui.setup("Overview", "What has come in, where it came from, what needs a person's attention, and how current the data is.")
    df = rp.add_labels(p.db.messages_df())
    statuses = svc.source_status()
    if df.empty and not any(s["connected"] for s in statuses):
        ui.empty_state("No reports yet.", "Open Data Sources and System Health to refresh real-world sources, or load simulated demo reports in Data and Settings.")
        return
    real = df[df["origin"] != "user_report"] if not df.empty else df
    ok_times = [s["last_success_at"] for s in statuses if s["last_success_at"]]
    latest = max(ok_times) if ok_times else None
    last_age = min((s["age_minutes"] for s in statuses if s["age_minutes"] is not None), default=None)

    cols = st.columns(5)
    ui.kpi(cols[0], len(real), "Imported real-world items")
    ui.kpi(cols[1], int(real["origin"].isin(["official_event", "official_alert"]).sum()) if len(real) else 0, "Official events and alerts")
    ui.kpi(cols[2], int((real["nlp_status"] == "needs_review").sum()) if len(real) else 0, "Reports needing human review", alert=True)
    ui.kpi(cols[3], int((df["verification"] == "human_verified").sum()) if len(df) else 0, "Verified by a reviewer")
    ui.kpi(cols[4], rp.freshness_text(last_age), "Latest successful data refresh")
    if latest:
        st.caption(f"Last successful fetch: {ui.fmt_utc(latest)}. Sources refresh periodically; this is not a live feed.")
    stale = [s for s in statuses if s["health"] in ("stale", "failing")]
    for s in stale:
        ui.note(f"<b>{s['label']}</b> is {s['health']}. Items from this source may be out of date.", warn=True)

    left, right = st.columns([3, 2])
    with left:
        ui.section("Reports grouped by need category")
        needs = rp.filter_reports(df, origins=list(config.NEED_ORIGINS)) if len(df) else df
        cat = analytics.by_category(needs)
        if cat["count"].sum() == 0:
            st.caption("No classified reports yet. Official events and alerts are not classified into needs.")
        else:
            fig = px.bar(cat, x="count", y="label", orientation="h", color="label", color_discrete_map=ui.CAT_COLORS)
            fig.update_layout(showlegend=False, yaxis=dict(autorange="reversed", title=None), xaxis_title="Reports (a report can have several categories)")
            st.plotly_chart(ui.style_fig(fig, 300), use_container_width=True)
        st.caption("NLP-inferred categories on real reports are unconfirmed: the classifier was trained on synthetic data.")
    with right:
        ui.section("Verified vs unverified")
        v = df["verification"].value_counts().rename_axis("k").reset_index(name="count") if len(df) else None
        if v is not None and len(v):
            v["name"] = v["k"].map(lambda k: ui.VERIF_STYLE.get(k, (k,))[0])
            fig = px.pie(v, names="name", values="count", hole=0.6, color="name",
                         color_discrete_map={ui.VERIF_STYLE[k][0]: ui.VERIF_STYLE[k][1] for k in ui.VERIF_STYLE})
            fig.update_traces(textinfo="value", sort=False)
            st.plotly_chart(ui.style_fig(fig, 300), use_container_width=True)

    a, b = st.columns(2)
    with a:
        ui.section("Reports by source")
        if len(real) or len(df):
            bysrc = df.assign(src=df["source_name"].fillna("Submitted / simulated"))["src"].value_counts().rename_axis("source").reset_index(name="count")
            fig = px.bar(bysrc, x="count", y="source", orientation="h", color_discrete_sequence=["#1F4E9C"])
            fig.update_layout(yaxis=dict(autorange="reversed", title=None), xaxis_title="Items")
            st.plotly_chart(ui.style_fig(fig, 240), use_container_width=True)
    with b:
        ui.section("Source status")
        for s in statuses:
            colour = {"ok": "#2E7D5B", "stale": "#B7791F", "failing": "#B3261E"}.get(s["health"], "#8A97A8")
            st.markdown(f"<div style='margin:.3rem 0'><span class='rq-dot' style='background:{colour}'></span><b>{s['name']}</b> "
                        f"<span style='color:#5B6B82'>{s['health'].replace('_', ' ')} &middot; last success {rp.freshness_text(s['age_minutes'])}</span></div>", unsafe_allow_html=True)
        errs = svc.db.recent_errors(limit=3)
        if errs:
            if ui.is_admin():
                for e in errs:
                    st.caption(f"{e['occurred_at']} UTC &middot; {e['source']} &middot; {e['kind']}: {e['message']}")
            else:
                st.caption(f"{len(errs)} recent ingestion problem(s). Reviewers can see details.")

    ui.section("Recent official events")
    ev = rp.filter_reports(df, origins=["official_event", "official_alert"], include_expired=False) if len(df) else df
    if ev is not None and len(ev):
        for _, r in ev.sort_values(["is_india", "ts"], ascending=[False, False]).head(5).iterrows():
            st.markdown(ui.update_card(r), unsafe_allow_html=True)
    else:
        st.caption("No current official events.")

    ui.section("Review queue: submitted and situation reports awaiting a person")
    q = df[(df["review_status"] == "pending") & df["origin"].isin(config.NEED_ORIGINS)].copy() if len(df) else df
    if q is None or q.empty:
        ui.empty_state("Nothing is waiting for review.")
    else:
        q["priority"] = q["low_confidence"] * 2 + (q["origin"] == "situation_report").astype(int) * 2 + q["location_status"].isin(["unknown", "unverified", "ambiguous"]).astype(int)
        for _, r in q.sort_values(["priority", "ts"], ascending=[False, False]).head(6).iterrows():
            st.markdown(ui.update_card(r) if r["origin"] == "situation_report" else ui.report_card(r), unsafe_allow_html=True)
    ui.note(config.CONFIDENCE_NOTE)
