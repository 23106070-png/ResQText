import pandas as pd
import streamlit as st

import config
import reports as rp
from dashboard import ui
from ingestion.models import ORIGIN_LABELS

PAGE_SIZES = [10, 25, 50]


def render():
    p = ui.get_pipeline()
    ui.setup("Real-World Updates", "Reports and official alerts fetched from connected sources. Each item links to its original.")
    ui.note("<b>How to read this page.</b> Titles, times, links and locations are <b>observed</b> source data. Category chips are "
            "<b>NLP-inferred</b> and not confirmed. An official alert of a hazard in an area does not mean any person there needs rescue or supplies.")
    df = rp.add_labels(p.db.messages_df())
    df = df[df["origin"] != "user_report"] if not df.empty else df
    if df.empty:
        ui.empty_state("Nothing imported yet.", "Open Data Sources and System Health to check which sources are configured, then refresh.")
        return
    c1, c2, c3 = st.columns(3)
    sources = c1.multiselect("Source", sorted(df["source_name"].dropna().unique()))
    etypes = c2.multiselect("Event type", sorted(df["event_type"].dropna().unique()))
    cats = c3.multiselect("Need category (NLP-inferred)", config.LABELS)
    c4, c5, c6 = st.columns(3)
    verif = c4.multiselect("Verification", list(rp.VERIFICATION_LABELS), format_func=rp.VERIFICATION_LABELS.get)
    locs = c5.multiselect("Location", sorted(df["location_text"].dropna().unique()))
    nlps = c6.multiselect("Processing status", list(rp.NLP_STATUS_LABELS), format_func=rp.NLP_STATUS_LABELS.get)
    c7, c8, c9 = st.columns([2, 1, 1])
    lo, hi = df["ts"].min(), df["ts"].max()
    dates = c7.date_input("Published between", (lo.date(), hi.date()), min_value=lo.date(), max_value=hi.date()) if pd.notna(lo) else ()
    india = c8.toggle("India only", value=False)
    hide_exp = c9.toggle("Hide expired", value=False)
    q = st.text_input("Search title or text", placeholder="e.g. flood, Bihar, cyclone")

    f = rp.filter_reports(df, source=sources or None, event_type=etypes or None, category=cats or None, verification=verif or None,
                          location=locs or None, nlp_status=nlps or None, india_only=india, include_expired=not hide_exp,
                          date_from=dates[0] if len(dates) == 2 else None, date_to=dates[1] if len(dates) == 2 else None, search=q or None)
    f = f.sort_values(["is_india", "ts"], ascending=[False, False]) if india is False else f.sort_values("ts", ascending=False)
    size = st.select_slider("Items per page", PAGE_SIZES, value=25)
    pages = max(1, -(-len(f) // size))
    page = st.number_input("Page", 1, pages, 1) if pages > 1 else 1
    total, view = rp.paginate(f, size, (int(page) - 1) * size)
    st.caption(f"{total} item(s). India-related items are listed first. Showing {len(view)} (page {int(page)} of {pages}).")
    if view.empty:
        ui.empty_state("No items match these filters.")
    for _, r in view.iterrows():
        st.markdown(ui.update_card(r), unsafe_allow_html=True)
    if not view.empty:
        srcs = view["attribution"].dropna().unique()
        with st.expander("Source attribution"):
            for a in srcs[:6]:
                st.write(a)
