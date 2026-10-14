"""ResQText dashboard.  Run: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="ResQText", page_icon=":material/emergency:", layout="wide", initial_sidebar_state="expanded")

from dashboard import ui  # noqa: E402
from dashboard.pages import (duplicates, emerging, evaluation, event_map, incoming, map_view, nlp_analysis, overview,  # noqa: E402
                             real_world, review, settings, sources_health)

nav = {
    "Monitor": [
        st.Page(overview.render, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
        st.Page(real_world.render, title="Real-World Updates", icon=":material/public:", url_path="updates"),
        st.Page(event_map.render, title="Disaster Event Map", icon=":material/crisis_alert:", url_path="events"),
        st.Page(emerging.render, title="Emerging Needs", icon=":material/trending_up:", url_path="trends"),
    ],
    "Analyse and review": [
        st.Page(nlp_analysis.render, title="NLP Analysis", icon=":material/psychology:", url_path="analysis"),
        st.Page(review.render, title="Human Review", icon=":material/rate_review:", url_path="review"),
        st.Page(duplicates.render, title="Duplicate Detection", icon=":material/content_copy:", url_path="duplicates"),
        st.Page(evaluation.render, title="Model Evaluation", icon=":material/fact_check:", url_path="evaluation"),
    ],
    "Submitted reports": [
        st.Page(incoming.render, title="Incoming Messages", icon=":material/inbox:", url_path="incoming"),
        st.Page(map_view.render, title="Report Location Map", icon=":material/map:", url_path="map"),
    ],
    "System": [
        st.Page(sources_health.render, title="Data Sources and Health", icon=":material/monitor_heart:", url_path="sources"),
        st.Page(settings.render, title="Data and Settings", icon=":material/settings:", url_path="settings"),
    ],
}
ui.sidebar_brand(ui.get_pipeline())
page = st.navigation(nav)
ui.admin_panel()
page.run()
