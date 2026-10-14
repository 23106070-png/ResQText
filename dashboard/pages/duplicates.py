import streamlit as st

import config
from dashboard import ui
from nlp import duplicate_detection as dd


def _side(title, mid, text, loc, ts):
    import html
    return (f"<div class='rq-card'><div class='rq-card-top'><span class='rq-id'>{title}: {mid}</span><span>{ui.fmt_time(ts)}</span></div>"
            f"<div class='rq-text'>{html.escape(text)}</div><div class='rq-meta'>Location: {html.escape(loc)}</div></div>")


def render():
    p = ui.get_pipeline()
    ui.setup("Duplicate Detection", "Reports that look alike. Similar wording can still be a different incident, so nothing is merged or deleted for you.")
    emb = dd.get_embedder()
    ui.note(f"Similarity method: <b>{emb.name}</b>." + (f" {dd.FALLBACK_REASON}" if dd.FALLBACK_REASON and "fallback" in emb.name else ""))
    thr = st.slider("Similarity threshold", 0.30, 0.95, float(st.session_state.get("sim_threshold", config.SIMILARITY_THRESHOLD)), 0.01,
                    help="Higher finds fewer, closer matches. Tune it on labelled pairs: see Model Evaluation.")
    st.session_state["sim_threshold"] = thr
    c1, c2 = st.columns([1, 3])
    if c1.button("Re-scan all reports"):
        df, n = p.db.messages_df(), 0
        bar = st.progress(0.0)
        for i, mid in enumerate(df["id"]):
            n += len(p.find_duplicates(mid, thr))
            bar.progress((i + 1) / len(df))
        bar.empty()
        st.success(f"Scan complete: {n} suggestion(s) at threshold {thr:.2f} (new ones are added; earlier decisions are kept).")
    c2.caption("Score = similarity, lowered when locations differ or the reports are far apart in time.")

    status = st.radio("Show", ["pending", "confirmed", "rejected"], horizontal=True,
                      format_func=lambda s: {"pending": "To review", "confirmed": "Confirmed duplicates", "rejected": "Different incidents"}[s])
    items = [d for d in p.db.list_duplicates(status=status) if d["adjusted_score"] >= thr or status != "pending"]
    if not items:
        ui.empty_state("Nothing here.", "Add more reports or lower the threshold." if status == "pending" else "")
        return
    reviewer = st.session_state.get("reviewer") or "reviewer"
    for d in items[:25]:
        rel = {"same": "Same location", "different": "Different locations", "unknown": "Location unknown"}[d["location_relation"] or "unknown"]
        gap = f"{d['time_gap_hours']:.1f} h apart" if d["time_gap_hours"] is not None else "time gap unknown"
        st.markdown(f"**Score {d['adjusted_score']:.2f}** (raw similarity {d['similarity']:.2f}) &nbsp;|&nbsp; {rel} &nbsp;|&nbsp; {gap}")
        a, b = st.columns(2)
        a.markdown(_side("Report", d["message_id"], d["text_a"], d["loc_a"], d["ts_a"]), unsafe_allow_html=True)
        b.markdown(_side("Looks like", d["other_id"], d["text_b"], d["loc_b"], d["ts_b"]), unsafe_allow_html=True)
        if status == "pending":
            x, y, _ = st.columns([1, 1, 3])
            if x.button("Same incident", key=f"c{d['id']}", type="primary"):
                p.db.decide_duplicate(d["id"], "confirmed", reviewer)
                st.rerun()
            if y.button("Different incident", key=f"r{d['id']}"):
                p.db.decide_duplicate(d["id"], "rejected", reviewer)
                st.rerun()
        else:
            st.caption(f"Decided by {d['decided_by']} at {d['decided_at']}")
        st.divider()
