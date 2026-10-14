"""Semantic duplicate *suggestions*. Nothing is ever merged or deleted automatically."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

import config
from nlp.location_processing import is_unknown, norm_name
from nlp.preprocessing import normalize_text


class HashingEmbedder:
    """Offline fallback: stateless hashed word + character n-grams (lexical, not truly semantic)."""

    name = "hashing-ngrams (offline fallback)"

    def __init__(self):
        self.word = HashingVectorizer(ngram_range=(1, 2), n_features=2**15, alternate_sign=False,
                                      norm=None, stop_words="english")
        self.char = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2**15,
                                      alternate_sign=False, norm=None)

    def encode(self, texts: list[str]):
        texts = [normalize_text(t) for t in texts]
        m = sparse.hstack([self.word.transform(texts), 0.7 * self.char.transform(texts)]).tocsr()
        return normalize(m)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers: {model_name}"

    def encode(self, texts: list[str]):
        return np.asarray(self.model.encode([normalize_text(t) for t in texts], normalize_embeddings=True))


_EMBEDDER = None
FALLBACK_REASON = ""


def get_embedder(force_fallback: bool = False):
    """Sentence Transformers when enabled and loadable, otherwise the offline fallback."""
    global _EMBEDDER, FALLBACK_REASON
    if _EMBEDDER is not None and not force_fallback:
        return _EMBEDDER
    if config.USE_SENTENCE_TRANSFORMERS and not force_fallback:
        try:
            _EMBEDDER = SentenceTransformerEmbedder(config.SENTENCE_MODEL)
            return _EMBEDDER
        except Exception as exc:
            FALLBACK_REASON = f"Sentence Transformers unavailable ({type(exc).__name__}); using offline fallback."
    else:
        FALLBACK_REASON = "Sentence Transformers not enabled (set RESQTEXT_USE_SENTENCE_TRANSFORMERS=1)."
    _EMBEDDER = HashingEmbedder()
    return _EMBEDDER


def cosine_matrix(a, b) -> np.ndarray:
    prod = a @ b.T
    return prod.toarray() if sparse.issparse(prod) else np.asarray(prod)


def location_relation(loc_a: str | None, loc_b: str | None) -> str:
    if is_unknown(loc_a) or is_unknown(loc_b):
        return "unknown"
    return "same" if norm_name(loc_a) == norm_name(loc_b) else "different"


def parse_ts(value) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


@dataclass
class DuplicateSuggestion:
    other_id: str
    similarity: float  # raw cosine similarity
    adjusted_score: float  # after location / time adjustments (this is what the threshold is applied to)
    location_relation: str  # same | different | unknown
    time_gap_hours: float | None
    reason: str


def adjust_score(sim: float, rel: str, gap_hours: float | None) -> tuple[float, str]:
    score, notes = sim, []
    if rel == "different":
        score -= config.DIFFERENT_LOCATION_PENALTY
        notes.append("different locations: likely a separate incident")
    elif rel == "same":
        notes.append("same location")
    else:
        notes.append("location unknown on one or both reports")
    if gap_hours is not None and gap_hours > config.TIME_WINDOW_HOURS:
        score -= 0.05
        notes.append(f"reports are {gap_hours:.0f}h apart")
    return max(score, 0.0), "; ".join(notes)


def find_duplicates(query: dict, candidates: list[dict], threshold: float | None = None,
                    embedder=None, top_k: int = 5) -> list[DuplicateSuggestion]:
    """query/candidates: dicts with id, text, location (optional), timestamp (optional)."""
    threshold = config.SIMILARITY_THRESHOLD if threshold is None else threshold
    candidates = [c for c in candidates if c.get("id") != query.get("id") and (c.get("text") or "").strip()]
    if not candidates or not (query.get("text") or "").strip():
        return []
    emb = embedder or get_embedder()
    sims = cosine_matrix(emb.encode([query["text"]]), emb.encode([c["text"] for c in candidates]))[0]
    q_ts, out = parse_ts(query.get("timestamp")), []
    for c, sim in zip(candidates, sims):
        rel = location_relation(query.get("location"), c.get("location"))
        c_ts = parse_ts(c.get("timestamp"))
        gap = abs((q_ts - c_ts).total_seconds()) / 3600 if q_ts and c_ts else None
        score, reason = adjust_score(float(sim), rel, gap)
        if score >= threshold:
            out.append(DuplicateSuggestion(c["id"], round(float(sim), 4), round(score, 4), rel,
                                           None if gap is None else round(gap, 2), reason))
    return sorted(out, key=lambda d: d.adjusted_score, reverse=True)[:top_k]
