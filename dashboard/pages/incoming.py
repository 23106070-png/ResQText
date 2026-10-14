import pandas as pd
import streamlit as st

import config
from dashboard import ui
from nlp.pipeline import CSVValidationError, parse_csv

TEMPLATE = "text,timestamp,source,location,message_id,is_simulated\n\"Five people need drinking water in Kalwa\",2024-07-26T10:15:00,sms,,R-001,1\n"


def _browse(p):
    df = p.db.messages_df()
    df = df[df["origin"] == "user_report"] if len(df) else df  # imported items are on Real-World Updates
    if df.empty:
        ui.empty_state("No submitted reports stored yet.", "Upload a CSV in the next tab. Imported real-world items are on Real-World Updates.")
        return
    f1, f2, f3 = st.columns(3)
    cats = f1.multiselect("Category", config.LABELS)
    locs = f2.multiselect("Location", sorted(df["location_text"].unique()))
    status = f3.multiselect("Review status", list(ui.STATUS_STYLE), format_func=lambda s: ui.STATUS_STYLE[s][0])
    f4, f5 = st.columns(2)
    lo, hi = df["ts"].min(), df["ts"].max()
    dates = f4.date_input("Date range", (lo.date(), hi.date()), min_value=lo.date(), max_value=hi.date()) if pd.notna(lo) else ()
    conf = f5.slider("Minimum model score", 0.0, 1.0, 0.0, 0.05)
    if cats:
        df = df[df["labels"].map(lambda l: any(c in l for c in cats))]
    if locs:
        df = df[df["location_text"].isin(locs)]
    if status:
        df = df[df["review_status"].isin(status)]
    if len(dates) == 2:
        df = df[(df["ts"].dt.date >= dates[0]) & (df["ts"].dt.date <= dates[1])]
    df = df[df["top_confidence"].fillna(0) >= conf]
    st.caption(f"{len(df)} report(s) match.")
    view = df.assign(labels=df["labels"].map(", ".join), resources=df["resources"].map(", ".join))
    st.dataframe(view[["id", "timestamp", "original_text", "labels", "location_text", "location_status", "people_count",
                       "resources", "top_confidence", "review_status", "is_simulated"]].rename(columns={
        "id": "ID", "original_text": "Report", "labels": "Categories", "location_text": "Location", "location_status": "Location check",
        "people_count": "People", "resources": "Resources", "top_confidence": "Model score", "review_status": "Status",
        "timestamp": "Time", "is_simulated": "Simulated"}), use_container_width=True, hide_index=True, height=420,
        column_config={"Model score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                       "Report": st.column_config.TextColumn(width="large")})


def _upload(p):
    st.download_button("Download CSV template", TEMPLATE, "resqtext_template.csv", "text/csv")
    ui.note("Required column: <b>text</b>. Optional: timestamp, source, location, message_id, is_simulated. "
            "Re-uploading the same file will not create duplicates; identical text at a different time is kept as a separate report.")
    up = st.file_uploader("Upload reports (CSV)", type=["csv"])
    if not up:
        return
    try:
        df, warnings = parse_csv(up)
    except CSVValidationError as exc:
        st.error(str(exc))
        return
    for w in warnings:
        st.warning(w)
    st.success(f"{len(df)} valid report(s) found.")
    st.dataframe(df[["text", "timestamp", "source"]].head(10), use_container_width=True, hide_index=True)
    if st.button(f"Analyse and store {len(df)} reports", type="primary"):
        bar, summary = st.progress(0.0, text="Processing..."), {"inserted": 0, "skipped_existing": 0, "failed": [], "duplicate_suggestions": 0}
        for i in range(len(df)):
            part = p.ingest_dataframe(df.iloc[i:i + 1])
            for k in ("inserted", "skipped_existing", "duplicate_suggestions"):
                summary[k] += part[k]
            summary["failed"] += part["failed"]
            bar.progress((i + 1) / len(df), text=f"Processing {i + 1} of {len(df)}")
        bar.empty()
        st.success(f"Stored {summary['inserted']} new report(s). Skipped {summary['skipped_existing']} already imported. "
                   f"{summary['duplicate_suggestions']} possible duplicate(s) flagged for review.")
        if summary["failed"]:
            st.error(f"{len(summary['failed'])} row(s) failed.")
            st.dataframe(pd.DataFrame(summary["failed"]), hide_index=True)


def render():
    p = ui.get_pipeline()
    ui.setup("Incoming Messages", "Browse stored reports or add new ones from a CSV file.")
    ui.require_model(p)
    t1, t2 = st.tabs(["Browse reports", "Upload CSV"])
    with t1:
        _browse(p)
    with t2:
        _upload(p)
