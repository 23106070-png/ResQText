import plotly.express as px
import streamlit as st

import analytics
import config
import reports as rp
from dashboard import ui


def render():
    p = ui.get_pipeline()
    ui.setup("Emerging Needs", "How the mix and volume of incoming reports are changing over time.")
    df = rp.add_labels(p.db.messages_df())
    choice = st.radio("Reports to analyse", ["All reports that can express needs", "Submitted / simulated reports", "Situation reports (ReliefWeb)"], horizontal=True)
    origins = {"All reports that can express needs": list(config.NEED_ORIGINS), "Submitted / simulated reports": ["user_report"],
               "Situation reports (ReliefWeb)": ["situation_report"]}[choice]
    df = rp.filter_reports(df, origins=origins) if len(df) else df
    if df.empty or df["ts"].isna().all():
        ui.empty_state("No timestamped reports yet.", "Load demo reports in Data and Settings.")
        return
    ui.note(analytics.CAVEAT, warn=True)
    c1, c2 = st.columns(2)
    hours = c1.select_slider("Comparison window", [3, 6, 12, 24, 48], value=12, format_func=lambda h: f"last {h} h vs the {h} h before")
    freq = c2.selectbox("Chart interval", ["1h", "3h", "6h", "12h"], index=2)

    ui.section("Change in reports by need")
    comp = analytics.window_comparison(df, hours).set_index("label")
    cols = st.columns(4)
    for col, label in zip(cols, ["Medical assistance", "Food and drinking water", "Shelter and evacuation", "Rescue assistance"]):
        r = comp.loc[label]
        delta = None if r["change_pct"] is None else f"{r['change_pct']:+.0f}%"
        col.metric(label, int(r["current"]), delta, delta_color="off", help=f"Previous window: {int(r['previous'])} reports")
    st.caption("Percent change is hidden when the earlier window had no reports. Small counts swing widely.")

    left, right = st.columns(2)
    with left:
        ui.section("Report volume")
        v = analytics.volume_over_time(df, freq)
        fig = px.bar(v, x="ts", y="count", color_discrete_sequence=["#1F4E9C"])
        fig.update_layout(xaxis_title=None, yaxis_title="Reports")
        st.plotly_chart(ui.style_fig(fig), use_container_width=True)
    with right:
        ui.section("Reports by need over time")
        t = analytics.category_trend(df, freq)
        fig = px.line(t, x="ts", y="count", color="label", color_discrete_map=ui.CAT_COLORS, markers=True)
        fig.update_layout(xaxis_title=None, yaxis_title="Reports", legend_title_text="")
        st.plotly_chart(ui.style_fig(fig), use_container_width=True)

    a, b = st.columns(2)
    with a:
        ui.section("Most requested resources")
        r = analytics.top_resources(df)
        if r.empty:
            st.caption("No resources extracted yet.")
        else:
            fig = px.bar(r, x="count", y="resource", orientation="h", color_discrete_sequence=["#2F7F8F"])
            fig.update_layout(yaxis=dict(autorange="reversed", title=None), xaxis_title="Reports")
            st.plotly_chart(ui.style_fig(fig), use_container_width=True)
    with b:
        ui.section("Requests by location")
        l = analytics.by_location(df)
        if l.empty:
            st.caption("No locations extracted yet.")
        else:
            fig = px.bar(l, x="count", y="location", orientation="h", color_discrete_sequence=["#5B6FD6"])
            fig.update_layout(yaxis=dict(autorange="reversed", title=None), xaxis_title="Reports")
            st.plotly_chart(ui.style_fig(fig), use_container_width=True)
    ui.section("Reports by place: latest window vs the one before")
    lw = analytics.location_window_comparison(df, hours)
    if lw.empty:
        st.caption("No located reports in these windows.")
    else:
        st.dataframe(lw.rename(columns={"location": "Place", "previous": "Previous", "current": "Latest", "change": "Change"}), hide_index=True, use_container_width=True)
    ui.note("These counts are <b>collected reports</b>, not the number of affected people or the amount of aid needed. Official alerts are excluded because they do not express needs.")
    with st.expander("Comparison table"):
        st.dataframe(comp.reset_index(), hide_index=True, use_container_width=True)
