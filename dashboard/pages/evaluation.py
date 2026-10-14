import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import config
from dashboard import ui


def render():
    ui.setup("Model Evaluation", "Every number here is calculated from real predictions on a held-out test set.")
    f = Path(config.ARTIFACT_DIR) / "metrics.json"
    if not f.exists():
        ui.empty_state("No evaluation report yet.", "Train the model: python -m training.train_classifier (or use the button on Data and Settings).")
        return
    rep = json.loads(f.read_text(encoding="utf-8"))
    c = rep["classifier"]
    if c.get("is_synthetic"):
        ui.note("<b>Synthetic data.</b> These scores come from computer-generated messages built from templates. "
                "They show the pipeline works; they are <b>not</b> evidence of real-world accuracy. Evaluate on real, labelled reports before relying on any result.", warn=True)

    ui.section("Classifier: held-out test set")
    cols = st.columns(5)
    ui.kpi(cols[0], f"{c['macro_f1']:.2f}", "Macro F1")
    ui.kpi(cols[1], f"{c['macro_precision']:.2f}", "Macro precision")
    ui.kpi(cols[2], f"{c['macro_recall']:.2f}", "Macro recall")
    ui.kpi(cols[3], f"{c['exact_match_accuracy']:.2f}", "Exact-match accuracy")
    ui.kpi(cols[4], f"{c['hamming_accuracy']:.2f}", "Per-label accuracy")
    st.caption(f"Training: {c['n_train']} | validation: {c['n_val']} | test: {c['n_test']} messages. Label threshold {c['threshold']:.2f}. "
               "Exact-match counts a report as right only when its whole set of labels is right, so it is strict for multi-need reports.")

    pc = pd.DataFrame(c["per_class"])
    left, right = st.columns([3, 2])
    with left:
        ui.section("Per-class performance")
        m = pc.melt(id_vars="label", value_vars=["precision", "recall", "f1"], var_name="metric", value_name="score")
        fig = px.bar(m, x="label", y="score", color="metric", barmode="group",
                     color_discrete_sequence=["#1F4E9C", "#D9822B", "#2F7F8F"], range_y=[0, 1])
        fig.update_layout(xaxis_title=None, yaxis_title=None, legend_title_text="")
        st.plotly_chart(ui.style_fig(fig, 330), use_container_width=True)
    with right:
        ui.section("Class balance")
        st.dataframe(pc.rename(columns={"label": "Category", "support": "Test count", "prevalence": "Share of test set"})[["Category", "Test count", "Share of test set"]],
                     hide_index=True, use_container_width=True,
                     column_config={"Share of test set": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    st.dataframe(pc.rename(columns=str.title), hide_index=True, use_container_width=True)

    ui.section("Confusion matrices (one per category)")
    labels = list(c["confusion"])
    fig = make_subplots(rows=2, cols=3, subplot_titles=labels, vertical_spacing=0.18)
    for i, l in enumerate(labels):
        m = c["confusion"][l]
        z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        fig.add_trace(go.Heatmap(z=z, x=["Pred. no", "Pred. yes"], y=["Actual no", "Actual yes"], text=z, texttemplate="%{text}",
                                 colorscale=[[0, "#F4F6FA"], [1, "#1F4E9C"]], showscale=False), row=i // 3 + 1, col=i % 3 + 1)
        fig.update_yaxes(autorange="reversed", row=i // 3 + 1, col=i % 3 + 1)
    st.plotly_chart(ui.style_fig(fig, 460), use_container_width=True)
    ui.note("Reports can carry several categories, so each category has its own 2x2 matrix: true/false positives and negatives.")

    ui.section("Why imbalance and recall matter here")
    st.markdown("Some needs are far rarer than others, and a model can look accurate by ignoring them. A missed rescue or medical request "
                "(a false negative) can cost more than a false alarm, so **recall** on those categories matters more than overall accuracy. "
                "Lower the label threshold in Data and Settings to trade precision for recall, and always keep a person in the loop.")
    ui.note("Low-scoring categories are listed in the per-class table above. Read recall for Rescue and Medical first.")

    ui.section("Duplicate detection: labelled pairs")
    d = rep.get("duplicates", {})
    if "error" in d or not d:
        st.info(d.get("error", "Not evaluated."))
        return
    t = d["test_at_recommended"]
    st.caption(f"Method: {d['embedder']}. Threshold chosen on {d['n_val_pairs']} validation pairs, reported on {d['n_test_pairs']} unseen test pairs.")
    cols = st.columns(4)
    ui.kpi(cols[0], f"{d['recommended_threshold']:.2f}", "Chosen threshold (from validation)")
    ui.kpi(cols[1], f"{t['precision']:.2f}", "Test precision")
    ui.kpi(cols[2], f"{t['recall']:.2f}", "Test recall")
    ui.kpi(cols[3], f"{t['f1']:.2f}", "Test F1")
    sw = pd.DataFrame(d["val_sweep"]).melt(id_vars="threshold", value_vars=["precision", "recall", "f1"])
    fig = px.line(sw, x="threshold", y="value", color="variable", color_discrete_sequence=["#1F4E9C", "#D9822B", "#2F7F8F"])
    fig.update_layout(xaxis_title="Similarity threshold", yaxis_title=None, legend_title_text="")
    st.plotly_chart(ui.style_fig(fig, 280), use_container_width=True)
    bt = pd.DataFrame([{"Pair type": k, "Pairs": v["n"], "Accuracy": v["accuracy"]} for k, v in d["test_by_pair_type"].items()])
    st.dataframe(bt, hide_index=True, use_container_width=True, column_config={"Accuracy": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    ui.note("'Easy' pairs are near-copies and flatter the score. The <b>paraphrase</b> row is the honest test of semantic matching: "
            "the offline fallback is lexical, so install Sentence Transformers for real semantic similarity.")
