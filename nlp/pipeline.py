"""Glue: classification + extraction + location resolution + duplicate detection + storage."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd

import config
from database.db import Database
from nlp import duplicate_detection as dd
from nlp.classifier import DisasterClassifier, ModelNotFoundError
from nlp.entity_extraction import extract
from nlp.location_processing import UNKNOWN, LocationResolver, get_resolver, is_unknown
from nlp.preprocessing import is_blank, normalize_text
from nlp.summarize import summarize


class CSVValidationError(ValueError):
    pass


def parse_csv(source, max_rows: int = 5000) -> tuple[pd.DataFrame, list[str]]:
    """Read + validate an uploaded CSV. Required column: text. Optional: timestamp, source, location,
    message_id, is_simulated. Returns (clean_df, warnings)."""
    try:
        df = pd.read_csv(source if isinstance(source, (str, Path)) else io.BytesIO(source.read()
                         if hasattr(source, "read") else source), dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise CSVValidationError("The file is empty.") from exc
    except Exception as exc:
        raise CSVValidationError(f"Could not read this file as CSV ({type(exc).__name__}).") from exc
    df.columns = [c.strip().lower() for c in df.columns]
    df["_row"] = df.index + 2  # spreadsheet row number (header is row 1), kept for error messages
    if "text" not in df.columns:
        raise CSVValidationError(f"Missing required column 'text'. Found: {list(df.columns)}")
    warnings = []
    if len(df) > max_rows:
        warnings.append(f"Only the first {max_rows} rows were kept.")
        df = df.head(max_rows)
    blank = df["text"].map(is_blank)
    if blank.any():
        warnings.append(f"{int(blank.sum())} row(s) with empty text were skipped.")
    df = df[~blank].copy()
    if df.empty:
        raise CSVValidationError("No rows with message text were found.")
    if "timestamp" in df.columns:
        parsed = pd.to_datetime(df["timestamp"].replace("", None), errors="coerce")
        bad = parsed.isna() & (df["timestamp"].str.strip() != "")
        if bad.any():
            warnings.append(f"{int(bad.sum())} timestamp(s) could not be read; import time was used for them.")
        df["timestamp"] = [t.strftime("%Y-%m-%dT%H:%M:%S") if pd.notna(t) else None for t in parsed]
    else:
        warnings.append("No 'timestamp' column: import time was used.")
        df["timestamp"] = None
    for col, default in (("source", "csv_upload"), ("location", ""), ("message_id", ""), ("is_simulated", "0")):
        if col not in df.columns:
            df[col] = default
    df["is_simulated"] = df["is_simulated"].str.strip().str.lower().isin({"1", "true", "yes"})
    return df.reset_index(drop=True), warnings


def _clean(value):
    """Turn NaN / empty strings from pandas into None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    value = str(value).strip()
    return value or None


class Pipeline:
    def __init__(self, db: Database | None = None, classifier: DisasterClassifier | None = None,
                 resolver: LocationResolver | None = None):
        self.db = db or Database()
        self._classifier = classifier
        self.resolver = resolver or get_resolver()

    @property
    def classifier(self) -> DisasterClassifier:
        if self._classifier is None:
            self._classifier = DisasterClassifier.load()  # raises ModelNotFoundError with instructions
        return self._classifier

    def reload_model(self) -> None:
        self._classifier = None

    def model_ready(self) -> bool:
        try:
            self.classifier
            return True
        except ModelNotFoundError:
            return False

    def analyze(self, text: str, location_hint: str | None = None, label_threshold: float | None = None) -> dict:
        """Pure analysis (nothing is stored)."""
        if is_blank(text):
            raise ValueError("Message text is empty.")
        pred = self.classifier.predict(text, label_threshold)
        ext = extract(text)
        place = ext.primary_location
        if is_unknown(place) and location_hint and location_hint.strip():
            place = location_hint.strip()
        loc = self.resolver.resolve(place)
        other = [e.value for e in ext.entities if e.type == "LOCATION" and e.value != ext.primary_location]
        return {
            "original_text": text,
            "normalized_text": normalize_text(text),
            "labels": pred.labels, "scores": pred.scores, "confidence": pred.confidence,
            "low_confidence": pred.low_confidence, "model_name": pred.model_name,
            "confidence_note": config.CONFIDENCE_NOTE,
            "location": loc, "other_locations": other,
            "people_mentioned": ext.people_count, "people_note": ext.people_note,
            "resources": ext.resources or ["Unknown"],
            "entities": [e for e in ext.to_dict()["entities"]],
            "spacy_used": ext.spacy_used,
            "needs_verification": bool(pred.low_confidence or loc["status"] in {"unknown", "unverified", "ambiguous"}),
        }

    def ingest(self, text: str, timestamp: str | None = None, source: str = "manual", is_simulated: bool = False,
               external_id: str | None = None, location_hint: str | None = None,
               similarity_threshold: float | None = None) -> dict:
        analysis = self.analyze(text, location_hint)
        mid, created = self.db.add_message(text, timestamp, source, is_simulated, external_id)
        dups = []
        if created:
            self.db.save_analysis(mid, analysis["labels"], analysis["scores"], analysis["confidence"],
                                  analysis["low_confidence"], analysis["model_name"], analysis["entities"],
                                  analysis["location"], analysis["people_mentioned"])
            dups = self.find_duplicates(mid, similarity_threshold, origin="user_report")
        return {"message_id": mid, "created": created, "analysis": analysis, "duplicates": dups}

    def find_duplicates(self, message_id: str, threshold: float | None = None, limit: int = 2000,
                        origin: str | None = None) -> list[dict]:
        """Duplicate suggestions among reports of the SAME origin (a ReliefWeb report is never compared with a
        submitted message, and so on). origin defaults to the message's own origin."""
        df = self.db.messages_df()
        if df.empty:
            return []
        me = df[df["id"] == message_id]
        if me.empty:
            return []
        me = me.iloc[0]
        origin = origin or me["origin"]
        df = df[df["origin"] == origin]
        others = df[df["id"] != message_id].head(limit)
        cands = [{"id": r.id, "text": r.original_text, "location": r.location_text, "timestamp": r.timestamp}
                 for r in others.itertuples()]
        sugg = dd.find_duplicates({"id": message_id, "text": me.original_text, "location": me.location_text,
                                   "timestamp": me.timestamp}, cands, threshold)
        self.db.add_duplicate_suggestions(message_id, sugg, dd.get_embedder().name)
        return [s.__dict__ for s in sugg]

    def ingest_dataframe(self, df: pd.DataFrame, similarity_threshold: float | None = None) -> dict:
        summary = {"inserted": 0, "skipped_existing": 0, "failed": [], "duplicate_suggestions": 0}
        for i, row in df.iterrows():
            try:
                r = self.ingest(row["text"], _clean(row.get("timestamp")), _clean(row.get("source")) or "csv_upload",
                                bool(row.get("is_simulated", False)), _clean(row.get("message_id")),
                                _clean(row.get("location")), similarity_threshold)
                if r["created"]:
                    summary["inserted"] += 1
                    summary["duplicate_suggestions"] += len(r["duplicates"])
                else:
                    summary["skipped_existing"] += 1
            except Exception as exc:
                summary["failed"].append({"row": int(row.get("_row", i + 2)), "reason": str(exc)})
        return summary


# ======================================================================================================
# Imported (real-world) items
# ======================================================================================================
def _compose_text(item) -> str:
    """The text we keep and analyse: title + body (excerpt). Entity offsets refer to this exact string."""
    body = (item.body or "").strip()
    if not body or body.startswith(item.title[:40]):
        return body or item.title
    return f"{item.title}\n{body}"


def _location_for(item, resolver) -> dict:
    """Location facts come from the SOURCE. Nothing is guessed: coordinates exist only when the source supplied
    them (event centroid) or the area text matches exactly one gazetteer place (approximate area centre)."""
    name = (item.location_text or item.country or "").strip() or UNKNOWN
    if item.lat is not None and item.lon is not None:
        return {"name": name, "status": "source_centroid", "lat": item.lat, "lon": item.lon, "source": item.source_name,
                "note": "Event centroid supplied by the source: approximate, not a person's location."}
    if item.origin == "official_alert" and item.location_text:
        found = {n for n, _, _ in resolver.find_in_text(item.location_text)}
        if len(found) == 1:
            r = resolver.resolve(next(iter(found)))
            if r["status"] == "verified":
                item.lat, item.lon, item.coord_precision = r["lat"], r["lon"], "area_centroid"
                return {"name": name, "status": "verified", "lat": r["lat"], "lon": r["lon"], "source": "gazetteer",
                        "note": f"Approximate centre of {r['name']}; the alert may cover a wider area."}
        return {"name": name, "status": "source_area_text", "lat": None, "lon": None, "source": item.source_name,
                "note": "Area text from the source; not mapped."}
    if item.location_text or item.country:
        return {"name": name, "status": "source_country", "lat": None, "lon": None, "source": item.source_name,
                "note": "Country-level location from the source; not mapped."}
    return {"name": UNKNOWN, "status": "unknown", "lat": None, "lon": None, "source": None, "note": "No location in the source data."}


def process_external(self: "Pipeline", item, similarity_threshold: float | None = None, retrieved_at: str | None = None) -> dict:
    """Store one imported item and (where meaningful) run the NLP pipeline on it. Never raises for NLP problems:
    the item is stored with nlp_status='failed' so a person can still see it."""
    from ingestion.models import iso, utc_now

    retrieved = retrieved_at or iso(utc_now())
    text = _compose_text(item)
    loc = _location_for(item, self.resolver)
    status, mid = self.db.upsert_external(item, text, retrieved, loc)
    if status not in ("inserted", "updated"):
        return {"status": status, "message_id": mid}
    ext = extract(text)
    ents = ext.to_dict()["entities"]
    summary = summarize(text)["text"]
    nlp_status, note = "processed", None
    try:
        if item.origin in ("official_event", "official_alert"):
            # An official alert says a hazard exists in an area. It does NOT say any person needs rescue or supplies,
            # so it is never classified into need categories.
            self.db.save_entities_only(mid, ents)
            nlp_status, note = "skipped_official", "Official source record: not classified into individual needs."
        elif item.language == "non-en":
            self.db.save_entities_only(mid, ents)
            nlp_status, note = "skipped_language", "Text is not English; the English classifier was not applied."
        else:
            pred = self.classifier.predict(text)
            self.db.save_analysis(mid, pred.labels, pred.scores, pred.confidence, pred.low_confidence, pred.model_name,
                                  ents, loc, ext.people_count)
            if not config.NLP_VALIDATED_ON_REAL:
                nlp_status, note = "needs_review", "NLP-inferred. The classifier was trained on synthetic messages only and is not validated on real reports."
            elif pred.low_confidence:
                nlp_status, note = "needs_review", "Low model score."
    except ModelNotFoundError:
        self.db.save_entities_only(mid, ents)
        nlp_status, note = "failed", "Classifier is not trained yet (python -m training.train_classifier)."
    except Exception as exc:  # keep the record even if NLP breaks
        nlp_status, note = "failed", f"NLP step failed ({type(exc).__name__})."
    self.db.set_nlp_result(mid, nlp_status, note, summary)
    if item.origin in ("situation_report", "official_alert"):
        try:
            self.find_duplicates(mid, similarity_threshold, limit=300, origin=item.origin)
        except Exception:
            pass
    return {"status": status, "message_id": mid, "nlp_status": nlp_status}


Pipeline.process_external = process_external
