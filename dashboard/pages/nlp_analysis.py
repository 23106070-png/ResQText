import pandas as pd
import streamlit as st

import config
from dashboard import ui

EXAMPLES = {
    "Trapped, needs water (Kalwa)": "We are trapped due to flooding near the municipal school in Kalwa. Five people need drinking water.",
    "Medical emergency, typos": "pregnent woman in labour at Mulund, no ambulence can reach us. water is rising",
    "Several needs": "Bridge near Dadar has collapsed. 12 elderly residents have no food and need blankets, one has fever.",
    "Unknown location": "Need help urgently, my father needs insulin and we have no electricity.",
    "General update": "Heavy rain continues since morning. Is the road to Thane open for traffic?",
}


def _imported_tab(p):
    import html
    import reports as rp
    from nlp.summarize import summarize

    df = rp.add_labels(p.db.messages_df())
    df = df[df["origin"] != "user_report"] if len(df) else df
    if df.empty:
        ui.empty_state("No imported reports yet.", "Refresh sources in Data Sources and System Health.")
        return
    order = df.sort_values(["is_india", "ts"], ascending=[False, False])
    mid = st.selectbox("Imported report", order["id"].tolist(),
                       format_func=lambda i: f"{order.loc[order['id'] == i, 'source_name'].iloc[0]}  |  {(order.loc[order['id'] == i, 'title'].iloc[0] or '')[:90]}")
    m = p.db.get_message(mid)
    row = df[df["id"] == mid].iloc[0]
    ui.section("Observed from the source")
    st.markdown(ui.update_card(row), unsafe_allow_html=True)
    ui.section("What the NLP pipeline did")
    st.markdown(f"<div class='rq-note'><b>Processing status:</b> {html.escape(rp.NLP_STATUS_LABELS.get(row['nlp_status'], row['nlp_status']))}"
                f"{'. ' + html.escape(row['nlp_note']) if row['nlp_note'] else ''}</div>", unsafe_allow_html=True)
    if m["classifications"]:
        scores = {c["label"]: c["score"] for c in m["classifications"]}
        st.markdown("".join(ui.chip(l) for l in m["predicted_labels"]), unsafe_allow_html=True)
        st.markdown(ui.score_bars(scores, m["predicted_labels"]), unsafe_allow_html=True)
        st.caption(config.CONFIDENCE_NOTE + " Scores on real reports are unvalidated.")
    else:
        st.caption("No need categories were assigned to this item.")
    ui.section("Supporting text")
    st.markdown(ui.highlight(m["original_text"], [{"type": e["type"], "start": e["start_char"], "end": e["end_char"]} for e in m["entities"]]), unsafe_allow_html=True)
    summ = summarize(m["original_text"])
    if summ["text"]:
        st.markdown(f"<div class='rq-note'><b>Extractive summary</b> (sentences copied from the text, nothing generated): {html.escape(summ['text'])}</div>", unsafe_allow_html=True)
    ents = pd.DataFrame(m["entities"])
    if not ents.empty:
        st.dataframe(ents[["type", "value", "evidence", "source"]], hide_index=True, use_container_width=True)


def render():
    p = ui.get_pipeline()
    ui.setup("NLP Analysis", "Inspect how the system reads a report. Every field shows the text it came from.")
    t1, t2 = st.tabs(["Paste a message", "Imported report"])
    with t2:
        _imported_tab(p)
    with t1:
        _paste_tab(p)


def _paste_tab(p):
    ui.require_model(p)
    choice = st.selectbox("Try an example", ["(write your own)"] + list(EXAMPLES))
    default = EXAMPLES.get(choice, st.session_state.get("nlp_text", ""))
    text = st.text_area("Report text", value=default, height=120, key=f"text_{choice}",
                        placeholder="Paste a message exactly as it was received.")
    if st.button("Analyse report", type="primary"):
        try:
            st.session_state["nlp_result"] = p.analyze(text, label_threshold=st.session_state.get("label_threshold"))
            st.session_state["nlp_text"] = text
        except ValueError:
            st.session_state.pop("nlp_result", None)
            st.error("The report is empty. Paste some text to analyse.")
    r = st.session_state.get("nlp_result")
    if not r:
        return

    if r["needs_verification"]:
        reasons = []
        if r["low_confidence"]:
            reasons.append("the model score is low")
        if r["location"]["status"] in {"unknown", "unverified", "ambiguous"}:
            reasons.append(f"the location is {r['location']['status']}")
        ui.note("<b>Needs verification:</b> " + " and ".join(reasons) + ".", warn=True)

    ui.section("Predicted needs")
    st.markdown("".join(ui.chip(l) for l in r["labels"]), unsafe_allow_html=True)
    st.markdown(ui.score_bars(r["scores"], r["labels"]), unsafe_allow_html=True)
    st.caption(config.CONFIDENCE_NOTE)

    ui.section("Evidence in the original message")
    st.markdown(ui.highlight(r["original_text"], r["entities"]), unsafe_allow_html=True)
    st.caption("Highlighted: locations (blue), resources (green), people counts (purple).")

    ui.section("Structured result")
    loc = r["location"]
    ev = {e["type"]: e["evidence"] for e in r["entities"] if e["type"] in {"LOCATION", "PEOPLE"}}
    res_ev = [e["evidence"] for e in r["entities"] if e["type"] == "RESOURCE"]
    rows = [
        ("Category", ", ".join(r["labels"]), "Model prediction"),
        ("Location", loc["name"] if loc["status"] != "unknown" else "Unknown", ev.get("LOCATION", "Needs verification")),
        ("Location check", ui.LOC_STYLE.get(loc["status"], ("", loc["status"]))[1], loc["note"] or ""),
        ("Resources requested", ", ".join(r["resources"]), "; ".join(res_ev) or "Needs verification"),
        ("People mentioned", r["people_mentioned"], ev.get("PEOPLE", "Not stated") + (f" ({r['people_note']})" if r["people_note"] else "")),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Field", "Value", "Evidence / note"]), hide_index=True, use_container_width=True)
    if r["other_locations"]:
        st.caption("Other places mentioned: " + ", ".join(r["other_locations"]))

    with st.expander("All extracted entities"):
        ents = pd.DataFrame(r["entities"])
        if ents.empty:
            st.write("No entities found.")
        else:
            st.dataframe(ents[["type", "value", "evidence", "source"]], hide_index=True, use_container_width=True)
    st.caption(f"Entity extraction: {'spaCy NER + rules' if r['spacy_used'] else 'rules + gazetteer only (spaCy model not loaded)'}.")

    ui.section("Save this report")
    c1, c2 = st.columns([1, 2])
    sim = c1.checkbox("Mark as simulated", value=True)
    if c2.button("Store report and check for duplicates"):
        out = p.ingest(st.session_state["nlp_text"], source="dashboard", is_simulated=sim, location_hint=None)
        if not out["created"]:
            st.info(f"This exact report is already stored as {out['message_id']}.")
        else:
            st.success(f"Stored as {out['message_id']}. {len(out['duplicates'])} possible duplicate(s) flagged for review.")
