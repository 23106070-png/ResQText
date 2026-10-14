"""Filtering, pagination and labelling of stored reports. Shared by the API and the dashboard so both always agree."""
from __future__ import annotations

import json

import pandas as pd

import config
from ingestion import models as _models
from ingestion.models import ORIGIN_LABELS, iso

VERIFICATION_LABELS = {
    "human_verified": "Verified by a reviewer",
    "incorrect": "Marked incorrect by a reviewer",
    "duplicate": "Marked duplicate by a reviewer",
    "investigating": "Needs further investigation",
    "official_source": "Official source (as published; not checked here)",
    "source_reported": "Reported by a humanitarian publisher (not verified here)",
    "unverified": "Unverified",
}
NLP_STATUS_LABELS = {
    "processed": "Processed", "needs_review": "Needs human review", "skipped_official": "Official record (not classified)",
    "skipped_language": "Not English (not classified)", "failed": "Processing failed",
}
LOCATION_PRECISION_LABELS = {"event_centroid": "Approximate (event centroid)", "area_centroid": "Approximate (area centre)",
                             "precise": "Precise", None: "Not mapped"}


def utc_now():  # indirection so tests can freeze the clock
    return _models.utc_now()


def verification_key(row) -> str:
    rs = row.get("review_status", "pending")
    return {"approved": "human_verified", "corrected": "human_verified", "rejected": "incorrect", "duplicate": "duplicate",
            "needs_info": "investigating"}.get(rs) or row.get("verification_status") or "unverified"


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        for c in ("verification", "verification_label", "origin_label", "is_expired"):
            df[c] = []
        return df
    df = df.copy()
    df["verification"] = df.apply(verification_key, axis=1)
    df["verification_label"] = df["verification"].map(VERIFICATION_LABELS)
    df["origin_label"] = df["origin"].map(ORIGIN_LABELS)
    now = iso(utc_now())
    df["is_expired"] = df["expires_at"].notna() & (df["expires_at"] < now)
    return df


def filter_reports(df: pd.DataFrame, *, origins=None, source=None, event_type=None, category=None, location=None,
                   country=None, verification=None, review_status=None, nlp_status=None, india_only=False,
                   date_from=None, date_to=None, min_confidence: float = 0.0, include_expired: bool = True,
                   search: str | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    m = pd.Series(True, index=df.index)
    if origins:
        m &= df["origin"].isin(origins)
    if source:
        m &= df["source_name"].isin([source] if isinstance(source, str) else source)
    if event_type:
        m &= df["event_type"].fillna("").str.lower().isin([e.lower() for e in ([event_type] if isinstance(event_type, str) else event_type)])
    if category:
        cats = [category] if isinstance(category, str) else category
        m &= df["labels"].map(lambda l: any(c in l for c in cats))
    if location:
        loc = [location] if isinstance(location, str) else location
        m &= df["location_text"].str.lower().isin([x.lower() for x in loc])
    if country:
        m &= df["country"].fillna("").str.contains(country, case=False, regex=False)
    if verification:
        v = [verification] if isinstance(verification, str) else verification
        m &= df["verification"].isin(v)
    if review_status:
        r = [review_status] if isinstance(review_status, str) else review_status
        m &= df["review_status"].isin(r)
    if nlp_status:
        n = [nlp_status] if isinstance(nlp_status, str) else nlp_status
        m &= df["nlp_status"].isin(n)
    if india_only:
        m &= df["is_india"] == 1
    if date_from is not None:
        m &= df["ts"] >= pd.Timestamp(date_from)
    if date_to is not None:
        m &= df["ts"] < pd.Timestamp(date_to) + pd.Timedelta(days=1)
    if min_confidence:
        m &= df["top_confidence"].fillna(0) >= min_confidence
    if not include_expired:
        m &= ~df["is_expired"]
    if search:
        s = search.lower()
        m &= df["title"].fillna("").str.lower().str.contains(s, regex=False) | df["original_text"].str.lower().str.contains(s, regex=False)
    return df[m]


def paginate(df: pd.DataFrame, limit: int = 50, offset: int = 0) -> tuple[int, pd.DataFrame]:
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    return len(df), df.iloc[offset:offset + limit]


PUBLIC_FIELDS = ["id", "origin", "origin_label", "title", "source_name", "source_url", "published_at", "retrieved_at",
                 "event_type", "severity", "country", "location_text", "location_status", "lat", "lon", "coord_precision",
                 "labels", "top_confidence", "nlp_status", "nlp_note", "verification", "verification_label", "review_status",
                 "is_india", "expires_at", "is_expired", "attribution", "summary"]


def excerpt(text: str | None, n: int = 400) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def to_public_records(df: pd.DataFrame, with_excerpt: bool = True) -> list[dict]:
    if df.empty:
        return []
    cols = [c for c in PUBLIC_FIELDS if c in df.columns]
    recs = json.loads(df[cols].to_json(orient="records", force_ascii=False))
    if with_excerpt:
        for rec, text in zip(recs, df["original_text"]):
            rec["excerpt"] = excerpt(text)
    return recs


def freshness_text(minutes: float | None) -> str:
    if minutes is None:
        return "never"
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{int(minutes)} min ago"
    if minutes < 48 * 60:
        return f"{minutes / 60:.0f} h ago"
    return f"{minutes / 1440:.0f} days ago"
