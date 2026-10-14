"""Train + evaluate the TF-IDF / Logistic Regression baseline.  Usage: python -m training.train_classifier [csv]"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import config
from nlp.classifier import DisasterClassifier, build_pipeline, decide_labels
from training.evaluate_model import evaluate_classifier, evaluate_duplicates, to_matrix
from training.prepare_dataset import load_labelled_csv, split_dataset


def train(csv_path=None, artifact_dir=None, pairs_path=None, is_synthetic: bool | None = None, verbose=True) -> dict:
    csv_path = csv_path or config.SAMPLE_CSV
    artifact_dir = artifact_dir or config.ARTIFACT_DIR
    df = load_labelled_csv(csv_path)
    if is_synthetic is None:
        is_synthetic = ("is_simulated" in df.columns and bool(df["is_simulated"].astype(int).all())) or "sample_messages" in str(csv_path)
    parts = split_dataset(df)
    train_df, val_df, test_df = parts["train"], parts["val"], parts["test"]
    labels = config.LABELS
    y_train = to_matrix(train_df["label_list"].tolist(), labels)

    # model selection on VALIDATION only
    best_C, best_f1 = None, -1.0
    for C in (1.0, 4.0, 10.0):
        pipe = build_pipeline(C).fit(train_df["norm"].tolist(), y_train)
        sc = pipe.predict_proba(val_df["norm"].tolist())
        pred = np.array([[1 if l in decide_labels(r, labels, config.LABEL_THRESHOLD) else 0 for l in labels] for r in sc])
        f1 = f1_score(to_matrix(val_df["label_list"].tolist(), labels), pred, average="macro", zero_division=0)
        if verbose:
            print(f"  C={C:<5} validation macro-F1={f1:.3f}")
        if f1 > best_f1:
            best_C, best_f1 = C, f1
    pipe = build_pipeline(best_C).fit(train_df["norm"].tolist(), y_train)  # trained on TRAIN only
    meta = {"dataset": str(csv_path), "is_synthetic": bool(is_synthetic), "C": best_C, "val_macro_f1": best_f1,
            "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
            "trained_at": datetime.now().isoformat(timespec="seconds")}
    clf = DisasterClassifier(pipe, labels, meta)
    clf.save(artifact_dir)

    # held-out TEST evaluation (touched only here)
    metrics = evaluate_classifier(clf, test_df)
    metrics.update(meta)
    metrics["label_counts_train"] = {l: int(y_train[:, i].sum()) for i, l in enumerate(labels)}
    report = {"classifier": metrics}
    pairs_path = pairs_path or config.PAIRS_CSV
    try:
        report["duplicates"] = evaluate_duplicates(pd.read_csv(pairs_path))
        report["duplicates"]["is_synthetic"] = True
    except Exception as exc:
        report["duplicates"] = {"error": f"Duplicate evaluation skipped: {exc}"}
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if verbose:
        c = metrics
        print(f"Test macro-F1={c['macro_f1']:.3f} exact-match={c['exact_match_accuracy']:.3f} "
              f"(train={c['n_train']}, test={c['n_test']}, synthetic={is_synthetic})")
        print(f"Saved model + metrics to {out}")
    return report


if __name__ == "__main__":
    print("Training TF-IDF + Logistic Regression ...")
    train(sys.argv[1] if len(sys.argv) > 1 else None)
