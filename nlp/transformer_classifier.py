"""OPTIONAL transformer classifier (zero-shot). Not imported by the core app unless enabled.

Enable with RESQTEXT_USE_TRANSFORMER=1 and `pip install transformers torch`. The first call downloads a
model from Hugging Face (internet needed once). If anything is missing, is_available() is False and the
TF-IDF baseline keeps working.
"""
from __future__ import annotations

import config


def is_available() -> bool:
    if not config.USE_TRANSFORMER:
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


class ZeroShotClassifier:
    def __init__(self, model: str = "facebook/bart-large-mnli"):
        from transformers import pipeline

        self.pipe = pipeline("zero-shot-classification", model=model)

    def predict(self, text: str, labels: list[str]) -> dict[str, float]:
        out = self.pipe(text, candidate_labels=labels, multi_label=True)
        return {l: round(float(s), 4) for l, s in zip(out["labels"], out["scores"])}
