import html

import streamlit as st

import config
import reports as rp
from dashboard import ui

ACTIONS = {"Verify": "approved", "Save corrections": "corrected", "Incorrect": "rejected", "Duplicate": "duplicate", "Needs investigation": "needs_info"}


def render():
    p = ui.get_pipeline()
    ui.setup("Human Review", "Check each report against its original source, correct what the model got wrong, then record a decision.")
    ui.note("<b>What \"Verified\" means:</b> a named reviewer checked this record against its source. It does not mean the underlying event or "
            "need is officially confirmed. Official-source records keep their separate &quot;Official source&quot; label.")
    df = rp.add_labels(p.db.messages_df())
    if df.empty:
        ui.empty_state("No reports to review yet.")
        return
    signed_in = ui.is_admin()
    if not signed_in:
        ui.require_admin("Recording review decisions")
    st.text_input("Your name (recorded with every decision)", key="reviewer", placeholder="e.g. A. Sharma", disabled=not signed_in)

    c1, c2 = st.columns(2)
    only_pending = c1.toggle("Only reports awaiting review", value=True)
    scope = c2.selectbox("Show", ["Submitted and situation reports", "Official events and alerts", "Everything"])
    origins = {"Submitted and situation reports": list(config.NEED_ORIGINS), "Official events and alerts": ["official_event", "official_alert"],
               "Everything": None}[scope]
    pool = rp.filter_reports(df, origins=origins)
    pool = pool[(pool["review_status"] == "pending") | (pool["pending_update_json"].notna())] if only_pending else pool
    if pool.empty:
        st.success("Nothing to review here.")
    else:
        pool = pool.assign(prio=pool["low_confidence"] * 2 + pool["pending_update_json"].notna().astype(int) * 3
                           + (pool["nlp_status"] == "needs_review").astype(int) * 2
                           + pool["location_status"].isin(["unknown", "unverified", "ambiguous"]).astype(int)).sort_values(["prio", "ts"], ascending=[False, False])
        mid = st.selectbox("Report", pool["id"].tolist(),
                           format_func=lambda i: f"{i}  |  {(pool.loc[pool['id'] == i, 'title'].iloc[0] or pool.loc[pool['id'] == i, 'original_text'].iloc[0])[:80]}")
        m = p.db.get_message(mid)
        row = df[df["id"] == mid].iloc[0]
        st.markdown(ui.update_card(row) if row["origin"] != "user_report" else ui.report_card(row), unsafe_allow_html=True)
        if m.get("pending_update_json"):
            ui.note("<b>The source published a newer version of this item.</b> Your reviewed version has been kept. "
                    "Applying the update replaces the source text and marks the item as needing investigation.", warn=True)
            if st.button("Apply the newer source version", disabled=not signed_in):
                p.db.apply_pending_update(mid, st.session_state.get("reviewer") or "reviewer")
                st.rerun()
        ui.section("Original text")
        st.markdown(ui.highlight(m["original_text"], [{"type": e["type"], "start": e["start_char"], "end": e["end_char"]} for e in m["entities"]]), unsafe_allow_html=True)
        with st.expander("What the model extracted (model-generated, unconfirmed)"):
            st.write({"Predicted": m["predicted_labels"] or "none", "Location": m["location_text"], "People mentioned": m["people_count"],
                      "Resources": [e["value"] for e in m["entities"] if e["type"] == "RESOURCE"] or "Unknown"})
        dups = p.db.list_duplicates(message_id=mid)
        if dups:
            ui.note(f"{len(dups)} possible duplicate(s) of this report. Resolve them on the Duplicate Detection page.")

        needs = row["origin"] in config.NEED_ORIGINS
        with st.form("review_form"):
            labels = st.multiselect("Categories (edit if the model is wrong)", config.LABELS, default=m["final_labels"]) if needs else []
            c1, c2, c3 = st.columns([2, 1, 1])
            loc = c1.text_input("Location", value="" if m["location_status"] == "unknown" else m["location_text"])
            lat = c2.text_input("Latitude (optional)", value="" if m["lat"] is None else str(m["lat"]))
            lon = c3.text_input("Longitude (optional)", value="" if m["lon"] is None else str(m["lon"]))
            notes = st.text_area("Notes (why you decided this)", height=70)
            btns = st.columns(len(ACTIONS))
            action = None
            for col, (label, status) in zip(btns, ACTIONS.items()):
                if col.form_submit_button(label, type="primary" if status == "approved" else "secondary", disabled=not signed_in):
                    action = status
        if action and signed_in:
            try:
                changed_labels = needs and sorted(labels) != sorted(m["predicted_labels"])
                loc_changed = bool(loc.strip()) and loc.strip() != m["location_text"]
                la = float(lat) if lat.strip() else None
                lo = float(lon) if lon.strip() else None
                if (la is None) != (lo is None):
                    raise ValueError("Enter both latitude and longitude, or leave both blank.")
                if action == "corrected" and not (changed_labels or loc_changed or (la is not None and la != m["lat"])):
                    raise ValueError("Nothing was changed. Edit a field first, or choose another decision.")
                if needs and action in {"approved", "corrected"} and not labels:
                    raise ValueError("Pick at least one category.")
                if action in {"rejected", "duplicate", "needs_info"} and not notes.strip():
                    raise ValueError("Add a short note explaining this decision.")
                p.db.add_review(mid, st.session_state.get("reviewer", ""), action, labels if changed_labels else None,
                                loc.strip() if (loc_changed or la is not None) else None, la, lo, notes)
                st.success("Decision saved with your name and the current time.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc) if not str(exc).startswith("could not convert") else "Latitude and longitude must be numbers.")

    ui.section("Reviewed results")
    out = p.db.reviewed_export()
    if out.empty:
        st.caption("Decisions you save appear here.")
    else:
        st.dataframe(out, hide_index=True, use_container_width=True, height=240)
        st.download_button("Export reviewed results (CSV)", out.to_csv(index=False), "resqtext_reviewed.csv", "text/csv", type="primary")
