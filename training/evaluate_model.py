"""Metric calculation. Every number comes from real predictions; nothing is hardcoded."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, hamming_loss, multilabel_confusion_matrix,
                             precision_recall_fscore_support)

import config
from nlp.classifier import decide_labels
from nlp.duplicate_detection import adjust_score, cosine_matrix, get_embedder, location_relation


def to_matrix(label_lists: list[list[str]], labels: list[str]) -> np.ndarray:
    return np.array([[1 if l in ls else 0 for l in labels] for ls in label_lists])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    return {
        "exact_match_accuracy": float(accuracy_score(y_true, y_pred)),  # whole label set must match
        "hamming_accuracy": float(1 - hamming_loss(y_true, y_pred)),
        "macro_precision": float(np.mean(p)), "macro_recall": float(np.mean(r)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "per_class": [{"label": l, "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                       "support": int(s[i]), "prevalence": float(y_true[:, i].mean())} for i, l in enumerate(labels)],
        "confusion": {l: {"tn": int(m[0, 0]), "fp": int(m[0, 1]), "fn": int(m[1, 0]), "tp": int(m[1, 1])}
                      for l, m in zip(labels, mcm)},
    }


def evaluate_classifier(clf, test_df: pd.DataFrame, threshold: float | None = None) -> dict:
    threshold = config.LABEL_THRESHOLD if threshold is None else threshold
    scores = clf.predict_scores(test_df["text"].tolist())
    y_pred = np.array([[1 if l in decide_labels(row, clf.labels, threshold) else 0 for l in clf.labels] for row in scores])
    y_true = to_matrix(test_df["label_list"].tolist(), clf.labels)
    m = compute_metrics(y_true, y_pred, clf.labels)
    m["threshold"] = threshold
    return m


def pair_scores(pairs: pd.DataFrame, embedder=None) -> np.ndarray:
    emb = embedder or get_embedder()
    a, b = emb.encode(pairs["text_a"].tolist()), emb.encode(pairs["text_b"].tolist())
    raw = np.array([cosine_matrix(a[i:i + 1], b[i:i + 1])[0, 0] for i in range(len(pairs))])
    rel = [location_relation(x, y) for x, y in zip(pairs["location_a"], pairs["location_b"])]
    return np.array([adjust_score(float(s), r, None)[0] for s, r in zip(raw, rel)])


def prf_at(scores: np.ndarray, y: np.ndarray, thr: float) -> dict:
    pred = (scores >= thr).astype(int)
    tp, fp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": float(thr), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "tp": tp, "fp": fp, "fn": fn, "tn": int(((pred == 0) & (y == 0)).sum())}


def evaluate_duplicates(pairs: pd.DataFrame, embedder=None) -> dict:
    """Tune the threshold on VAL pairs, then report once on TEST pairs at that threshold."""
    emb = embedder or get_embedder()
    grid = np.round(np.arange(0.30, 0.96, 0.05), 2)
    val, test = pairs[pairs["split"] == "val"], pairs[pairs["split"] == "test"]
    if val.empty or test.empty:
        raise ValueError("duplicate_pairs.csv needs both 'val' and 'test' rows.")
    sv, st = pair_scores(val, emb), pair_scores(test, emb)
    sweep_val = [prf_at(sv, val["is_duplicate"].values, t) for t in grid]
    best = max(sweep_val, key=lambda d: (d["f1"], d["threshold"]))
    by_type = {}
    for ptype, grp in test.groupby("pair_type"):
        pred = (st[grp.index.map(lambda i: test.index.get_loc(i))] >= best["threshold"]).astype(int)
        by_type[ptype] = {"n": int(len(grp)), "accuracy": float((pred == grp["is_duplicate"].values).mean())}
    return {"embedder": emb.name, "n_val_pairs": int(len(val)), "n_test_pairs": int(len(test)),
            "recommended_threshold": best["threshold"], "val_sweep": sweep_val,
            "test_at_recommended": prf_at(st, test["is_duplicate"].values, best["threshold"]),
            "test_by_pair_type": by_type, "created_at": datetime.now().isoformat(timespec="seconds")}
