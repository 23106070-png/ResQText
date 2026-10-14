import pandas as pd
import plotly.express as px
import streamlit as st

import config
from dashboard import ui


def render():
    p = ui.get_pipeline()
    ui.setup("Location Map", "Where reports say help is needed. Coordinates are approximate area centres, never exact addresses.")
    df = p.db.messages_df()
    df = df[df["origin"] == "user_report"] if len(df) else df  # official events have their own map
    ui.note("This map shows <b>submitted/simulated reports</b>. Official hazard events are on Disaster Event Map.")
    if df.empty:
        ui.empty_state("No reports yet.", "Load demo reports in Data and Settings.")
        return
    cats = st.multiselect("Category", config.LABELS)
    if cats:
        df = df[df["labels"].map(lambda l: any(c in l for c in cats))]
    mapped = df.dropna(subset=["lat", "lon"]).copy()
    ui.section(f"{len(mapped)} of {len(df)} reports have map coordinates")
    if mapped.empty:
        ui.empty_state("No mappable reports.", "Locations must be recognised or corrected by a reviewer first.")
    else:
        mapped["top_label"] = mapped["labels"].map(lambda l: l[0] if l else "General information")
        g = (mapped.groupby(["location_text", "lat", "lon", "location_status"])
             .agg(reports=("id", "count"), top_label=("top_label", lambda s: s.value_counts().index[0]),
                  pending=("review_status", lambda s: int((s == "pending").sum()))).reset_index())
        fig = px.scatter_map(g, lat="lat", lon="lon", size="reports", size_max=34, color="top_label",
                             color_discrete_map=ui.CAT_COLORS, hover_name="location_text", opacity=0.82,
                             hover_data={"reports": True, "pending": True, "location_status": True, "lat": False, "lon": False},
                             zoom=9, center={"lat": float(g["lat"].mean()), "lon": float(g["lon"].mean())})
        fig.update_layout(map_style="open-street-map", legend_title_text="Most common need")
        st.plotly_chart(ui.style_fig(fig, 500), use_container_width=True)
        st.caption("Bubble size = number of reports. Colour = most common need there. Basemap tiles load from OpenStreetMap in your browser.")

    ui.section("Reports with no usable location")
    bad = df[df["lat"].isna()].copy()
    if bad.empty:
        st.success("Every report has a location.")
    else:
        st.dataframe(bad.assign(labels=bad["labels"].map(", ".join))[["id", "original_text", "location_text", "location_status", "labels"]]
                     .rename(columns={"id": "ID", "original_text": "Report", "location_text": "Extracted text", "location_status": "Check", "labels": "Categories"}),
                     hide_index=True, use_container_width=True, height=260)

    ui.section("Correct a location")
    with st.form("loc_fix"):
        mid = st.selectbox("Report", df["id"].tolist(), format_func=lambda i: f"{i}  -  {df.loc[df['id'] == i, 'original_text'].iloc[0][:70]}")
        name = st.text_input("Place name")
        c1, c2 = st.columns(2)
        lat, lon = c1.text_input("Latitude (optional)"), c2.text_input("Longitude (optional)")
        if st.form_submit_button("Save correction", type="primary"):
            try:
                la = float(lat) if lat.strip() else None
                lo = float(lon) if lon.strip() else None
                if not name.strip():
                    raise ValueError("Enter a place name.")
                p.db.add_review(mid, st.session_state.get("reviewer") or "location-editor", "corrected", None, name.strip(), la, lo, "Manual location correction")
                st.success("Saved. Without coordinates the report stays off the map and is marked unverified.")
            except ValueError as exc:
                st.error(str(exc) if "place" in str(exc) or "range" in str(exc) or "both" in str(exc) else "Latitude and longitude must be numbers, for example 19.2183 and 72.9781.")
