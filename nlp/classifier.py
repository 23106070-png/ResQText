"""Multi-label TF-IDF + Logistic Regression classifier (the required baseline)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

import config
from nlp.preprocessing import normalize_text

MODEL_FILE = "classifier.joblib"


class ModelNotFoundError(RuntimeError):
    """Raised when no usable trained classifier artifact exists."""


@dataclass
class Prediction:
    labels: list[str]
    scores: dict[str, float]  # per-label model score (NOT probability that a report is true)
    confidence: float  # score of the top label
    low_confidence: bool = False
    model_name: str = "tfidf-logreg"
    extras: dict = field(default_factory=dict)


def build_pipeline(C: float = 4.0) -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            # character n-grams keep the model robust to spelling mistakes
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ]
    )
    clf = OneVsRestClassifier(LogisticRegression(C=C, max_iter=1000, class_weight="balanced"))
    return Pipeline([("features", features), ("clf", clf)])


def decide_labels(scores: np.ndarray, labels: list[str], threshold: float) -> list[str]:
    chosen = [labels[i] for i, s in enumerate(scores) if s >= threshold]
    return chosen or [labels[int(np.argmax(scores))]]


class DisasterClassifier:
    def __init__(self, pipeline: Pipeline, labels: list[str], meta: dict | None = None):
        self.pipeline = pipeline
        self.labels = labels
        self.meta = meta or {}

    @classmethod
    def load(cls, artifact_dir: Path | str | None = None) -> "DisasterClassifier":
        path = Path(artifact_dir or config.ARTIFACT_DIR) / MODEL_FILE
        if not path.exists():
            raise ModelNotFoundError(f"No trained model at {path}. Run: python -m training.train_classifier")
        try:
            blob = joblib.load(path)
            return cls(blob["pipeline"], blob["labels"], blob.get("meta", {}))
        except Exception as exc:  # corrupted file or incompatible scikit-learn version
            raise ModelNotFoundError(f"Model artifact at {path} could not be loaded ({exc}). Retrain it.") from exc

    def save(self, artifact_dir: Path | str | None = None) -> Path:
        d = Path(artifact_dir or config.ARTIFACT_DIR)
        d.mkdir(parents=True, exist_ok=True)
        path = d / MODEL_FILE
        joblib.dump({"pipeline": self.pipeline, "labels": self.labels, "meta": self.meta}, path)
        return path

    def predict_scores(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba([normalize_text(t) for t in texts])

    def predict(self, text: str, threshold: float | None = None) -> Prediction:
        threshold = config.LABEL_THRESHOLD if threshold is None else threshold
        scores = self.predict_scores([text])[0]
        top = float(np.max(scores))
        return Prediction(
            labels=decide_labels(scores, self.labels, threshold),
            scores={l: round(float(s), 4) for l, s in zip(self.labels, scores)},
            confidence=round(top, 4),
            low_confidence=top < config.LOW_CONFIDENCE,
        )
