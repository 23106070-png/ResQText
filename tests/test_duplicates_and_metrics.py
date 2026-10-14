import numpy as np
import pandas as pd
import pytest

import config
from nlp import duplicate_detection as dd
from training.evaluate_model import compute_metrics, evaluate_duplicates, prf_at
from training.prepare_dataset import DatasetError, assert_no_leakage, load_labelled_csv, normalize_labels, split_dataset

EMB = dd.HashingEmbedder()
A = "Five people stuck on a rooftop in Kalwa, need a boat urgently."


def test_same_text_different_location_not_suggested():
    q = {"id": "q", "text": A, "location": "Kalwa", "timestamp": "2024-07-26T10:00:00"}
    other = {"id": "o", "text": A.replace("Kalwa", "Mumbra"), "location": "Mumbra", "timestamp": "2024-07-26T10:05:00"}
    assert dd.find_duplicates(q, [other], 0.7, EMB) == []
    sim_only = dd.find_duplicates(q, [other], 0.3, EMB)  # raw similarity is high, adjusted score is lower
    assert sim_only[0].location_relation == "different" and sim_only[0].adjusted_score < sim_only[0].similarity


def test_identical_text_different_timestamps_still_suggested():
    q = {"id": "q", "text": A, "location": "Kalwa", "timestamp": "2024-07-26T10:00:00"}
    o = {"id": "o", "text": A, "location": "Kalwa", "timestamp": "2024-07-26T20:00:00"}
    s = dd.find_duplicates(q, [o], 0.7, EMB)[0]
    assert s.time_gap_hours == 10 and s.similarity > 0.99


def test_far_apart_in_time_is_penalised():
    q = {"id": "q", "text": A, "location": "Kalwa", "timestamp": "2024-07-26T10:00:00"}
    o = {"id": "o", "text": A, "location": "Kalwa", "timestamp": "2024-08-26T10:00:00"}
    assert dd.find_duplicates(q, [o], 0.7, EMB)[0].adjusted_score < 0.96


def test_no_candidates_or_blank_query():
    assert dd.find_duplicates({"id": "q", "text": A}, [], 0.7, EMB) == []
    assert dd.find_duplicates({"id": "q", "text": " "}, [{"id": "o", "text": A}], 0.7, EMB) == []


def test_metrics_are_calculated_from_predictions():
    y_true = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    y_pred = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    m = compute_metrics(y_true, y_pred, ["a", "b"])
    pa, pb = m["per_class"]
    assert pa["precision"] == 0.5 and pa["recall"] == 0.5 and pb["precision"] == 1.0 and pb["recall"] == 1.0
    assert m["exact_match_accuracy"] == 0.5 and m["hamming_accuracy"] == 0.75
    assert m["macro_f1"] == pytest.approx((0.5 + 1.0) / 2)
    assert m["confusion"]["a"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_prf_at_threshold():
    r = prf_at(np.array([0.9, 0.8, 0.4, 0.2]), np.array([1, 0, 1, 0]), 0.5)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 1, 1) and r["precision"] == 0.5


def test_duplicate_evaluation_runs_on_pairs():
    d = evaluate_duplicates(pd.read_csv(config.PAIRS_CSV), EMB)
    assert 0.3 <= d["recommended_threshold"] <= 0.95 and d["n_test_pairs"] > 0


def test_leakage_guard_and_splits():
    parts = split_dataset(load_labelled_csv(config.SAMPLE_CSV))
    assert not (set(parts["train"]["norm"]) & set(parts["test"]["norm"]))
    bad = {"train": pd.DataFrame({"norm": ["a", "b"]}), "test": pd.DataFrame({"norm": ["b"]})}
    with pytest.raises(DatasetError):
        assert_no_leakage(bad)


def test_label_normalisation_and_bad_dataset(tmp_path):
    assert normalize_labels("rescue|Food and water;nonsense") == ["Rescue assistance", "Food and drinking water"]
    f = tmp_path / "d.csv"
    f.write_text("msg,lab\nx,y\n")
    with pytest.raises(DatasetError):
        load_labelled_csv(f)
    with pytest.raises(DatasetError):
        load_labelled_csv(tmp_path / "missing.csv")
