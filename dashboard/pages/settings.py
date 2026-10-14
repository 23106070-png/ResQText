import pandas as pd
import streamlit as st

import config
from dashboard import ui
from nlp import duplicate_detection as dd
from nlp.entity_extraction import spacy_available


def _load_demo(p, n: int) -> None:
    df = pd.read_csv(config.SAMPLE_CSV)
    df = df[df["split"] == "test"].sort_values("timestamp").head(n)  # test split: messages the model never trained on
    df = df.rename(columns={"message_id": "message_id"})[["text", "timestamp", "source", "message_id", "is_simulated"]].assign(location="")
    df["is_simulated"] = True
    bar = st.progress(0.0, text="Loading simulated reports...")
    done = 0
    for i in range(len(df)):
        done += p.ingest_dataframe(df.iloc[i:i + 1])["inserted"]
        bar.progress((i + 1) / len(df))
    bar.empty()
    st.success(f"Loaded {done} simulated reports.")


def render():
    p = ui.get_pipeline()
    ui.setup("Data and Settings", "Demo data, thresholds and the status of optional components.")
    ui.section("Thresholds (this session only)")
    c1, c2 = st.columns(2)
    st.session_state["label_threshold"] = c1.slider("Category threshold", 0.1, 0.9, float(st.session_state.get("label_threshold", config.LABEL_THRESHOLD)), 0.05,
                                                    help="Lower it to catch more needs (higher recall, more false alarms).")
    st.session_state["sim_threshold"] = c2.slider("Duplicate similarity threshold", 0.3, 0.95, float(st.session_state.get("sim_threshold", config.SIMILARITY_THRESHOLD)), 0.01)
    st.caption("To change the defaults permanently set RESQTEXT_LABEL_THRESHOLD and RESQTEXT_SIMILARITY_THRESHOLD in .env.")

    ui.section("Component status")
    rows = [("Classifier (TF-IDF + Logistic Regression)", "Ready" if p.model_ready() else "Not trained"),
            ("spaCy NER model", f"Loaded ({config.SPACY_MODEL})" if spacy_available() else "Not loaded: rules and gazetteer only"),
            ("Duplicate similarity", dd.get_embedder().name),
            ("Geocoding", "Enabled (online)" if p.resolver.enable_geocoding else "Off (offline gazetteer)"),
            ("Transformer classifier", "Enabled" if config.USE_TRANSFORMER else "Off (optional)")]
    st.dataframe(pd.DataFrame(rows, columns=["Component", "Status"]), hide_index=True, use_container_width=True)

    ui.section("Model")
    meta = p.classifier.meta if p.model_ready() else {}
    if meta:
        st.caption(f"Trained {meta.get('trained_at', '?')} on {meta.get('n_train', '?')} messages. Synthetic data: {meta.get('is_synthetic')}.")
    if st.button("Retrain on the sample dataset", disabled=not ui.is_admin()):
        with st.spinner("Training and evaluating..."):
            from training.train_classifier import train
            train(verbose=False)
            p.reload_model()
        st.success("Done. See Model Evaluation.")

    ui.section("Data")
    c = p.db.counts()
    st.write(f"{c['messages']} reports stored ({c['simulated']} simulated). Database file: `{config.DB_PATH}`")
    n = st.number_input("Demo reports to load", 10, 239, 120, 10)
    if st.button("Load simulated demo reports", type="primary", disabled=not ui.is_admin()):
        ui.require_model(p)
        _load_demo(p, int(n))
    ui.note("Demo reports are generated from templates (data/sample_messages.csv, test split) and are clearly marked SIMULATED. Locations are real place names used only as examples.")
    with st.expander("Danger zone"):
        sure = st.checkbox("I understand this deletes every stored report, review and duplicate decision.")
        if st.button("Delete all stored data", disabled=not (sure and ui.is_admin())):
            p.db.clear_all()
            st.success("Database cleared.")
