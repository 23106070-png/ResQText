import plotly.express as px
import streamlit as st

import reports as rp
from dashboard import ui

ALERT_COLORS = {"Red": "#B3261E", "Orange": "#D9822B", "Green": "#2E7D5B"}


def render():
    p = ui.get_pipeline()
    ui.setup("Disaster Event Map", "Where official sources say a hazard is occurring. Not where any person is.")
    ui.note("<b>Reading the map.</b> Markers are <b>approximate</b> centres of official events or alert areas, taken from the source or "
            "from a small place list. They never show an individual's location. Events without reliable coordinates are listed below, not guessed.", warn=True)
    df = rp.add_labels(p.db.messages_df())
    ev = df[df["origin"].isin(["official_event", "official_alert"])] if not df.empty else df
    if ev.empty:
        ui.empty_state("No official events imported yet.", "Refresh GDACS or NDMA Sachet in Data Sources and System Health.")
        return
    c1, c2, c3 = st.columns(3)
    src = c1.multiselect("Source", sorted(ev["source_name"].dropna().unique()))
    et = c2.multiselect("Event type", sorted(ev["event_type"].dropna().unique()))
    india = c3.toggle("India-related only", value=False)
    show_old = st.toggle("Include expired events", value=False)
    ev = rp.filter_reports(ev, source=src or None, event_type=et or None, india_only=india, include_expired=show_old)
    mapped = ev.dropna(subset=["lat", "lon"]).copy()
    ui.section(f"{len(mapped)} of {len(ev)} events have approximate coordinates")
    if mapped.empty:
        ui.empty_state("No mappable events with these filters.")
    else:
        mapped["level"] = mapped["extra_json"].map(lambda j: (__import__("json").loads(j or "{}")).get("alert_level") or "Alert")
        mapped["precision"] = mapped["coord_precision"].map(rp.LOCATION_PRECISION_LABELS)
        mapped["when"] = mapped["published_at"]
        fig = px.scatter_map(mapped, lat="lat", lon="lon", color="level", color_discrete_map={**ALERT_COLORS, "Alert": "#1F4E9C"},
                             hover_name="title", hover_data={"event_type": True, "source_name": True, "precision": True, "when": True,
                                                              "lat": False, "lon": False, "level": False}, zoom=1.2, opacity=0.8,
                             center={"lat": 22, "lon": 80} if india or mapped["is_india"].any() else None)
        fig.update_traces(marker=dict(size=13))
        fig.update_layout(map_style="open-street-map", legend_title_text="Alert level")
        st.plotly_chart(ui.style_fig(fig, 520), use_container_width=True)
        st.caption("All coordinates are approximate. Basemap tiles load from OpenStreetMap in your browser.")
    ui.section("Event details")
    t = ev.sort_values("ts", ascending=False).head(200)
    st.dataframe(t.assign(Mapped=t["lat"].notna().map({True: "Approximate", False: "Not mapped"}))[
        ["title", "event_type", "severity", "location_text", "Mapped", "source_name", "published_at", "expires_at", "source_url"]]
        .rename(columns={"title": "Event", "event_type": "Type", "severity": "Severity", "location_text": "Place", "source_name": "Source",
                         "published_at": "Published (UTC)", "expires_at": "Expires / ends (UTC)", "source_url": "Source link"}),
        hide_index=True, use_container_width=True, height=360,
        column_config={"Source link": st.column_config.LinkColumn("Source link", display_text="open")})
    srcs = ev["attribution"].dropna().unique()
    with st.expander("Source attribution"):
        for a in srcs[:5]:
            st.write(a)
