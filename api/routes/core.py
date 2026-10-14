from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request

import security

import analytics
import config
from api import schemas
from nlp import duplicate_detection as dd
from nlp.classifier import ModelNotFoundError
from nlp.entity_extraction import spacy_available
from nlp.pipeline import Pipeline

router = APIRouter()


def get_pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline


def require_write(x_api_key: str | None = Header(default=None)) -> None:
    """Dependency for every endpoint that changes data. Fails closed in production without a configured key."""
    ok, status, msg = security.api_write_decision(x_api_key)
    if not ok:
        raise HTTPException(status, detail=msg)


def _model_guard(fn):
    try:
        return fn()
    except ModelNotFoundError as exc:
        raise HTTPException(503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc))


@router.get("/health", tags=["system"])
def health(p: Pipeline = Depends(get_pipeline)):
    return {"status": "ok", "model_loaded": p.model_ready(), "spacy_loaded": spacy_available(),
            "embedder": dd.get_embedder().name, "geocoding_enabled": p.resolver.enable_geocoding,
            "writes": {"open_local": "open (local demo mode)", "secret": "protected by API key",
                       "locked": "disabled (no API key configured)"}[security.api_mode()],
            "disclaimer": config.DISCLAIMER}


@router.post("/messages/analyze", response_model=schemas.AnalyzeResponse, tags=["messages"])
def analyze(body: schemas.AnalyzeRequest, p: Pipeline = Depends(get_pipeline), x_api_key: str | None = Header(default=None)):
    if body.save:
        require_write(x_api_key)
        r = _model_guard(lambda: p.ingest(body.text, body.timestamp, body.source, body.is_simulated,
                                          None, body.location_hint))
        return {"analysis": r["analysis"], "message_id": r["message_id"], "created": r["created"],
                "duplicates": r["duplicates"]}
    return {"analysis": _model_guard(lambda: p.analyze(body.text, body.location_hint))}


@router.post("/messages/batch", response_model=schemas.BatchResponse, tags=["messages"], dependencies=[Depends(require_write)])
def batch(body: schemas.BatchRequest, p: Pipeline = Depends(get_pipeline)):
    import pandas as pd

    df = pd.DataFrame([m.model_dump() for m in body.messages]).rename(columns={"external_id": "message_id", "location_hint": "location"})
    df["location"], df["message_id"] = df["location"].fillna(""), df["message_id"].fillna("")
    _model_guard(lambda: p.classifier)
    return p.ingest_dataframe(df)


@router.get("/messages", tags=["messages"])
def list_messages(p: Pipeline = Depends(get_pipeline), category: Optional[str] = None,
                  location: Optional[str] = None, review_status: Optional[str] = None,
                  min_confidence: float = Query(0, ge=0, le=1), limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    df = p.db.messages_df()
    if df.empty:
        return {"total": 0, "items": []}
    if category:
        if category not in config.LABELS:
            raise HTTPException(422, detail=f"category must be one of {config.LABELS}")
        df = df[df["labels"].map(lambda l: category in l)]
    if location:
        df = df[df["location_text"].str.lower() == location.lower()]
    if review_status:
        df = df[df["review_status"] == review_status]
    df = df[df["top_confidence"].fillna(0) >= min_confidence]
    cols = ["id", "timestamp", "original_text", "labels", "top_confidence", "location_text", "location_status",
            "review_status", "is_simulated", "resources"]
    return {"total": len(df), "items": json.loads(df.iloc[offset:offset + limit][cols].to_json(orient="records"))}


@router.get("/messages/{message_id}", tags=["messages"])
def get_message(message_id: str, p: Pipeline = Depends(get_pipeline)):
    m = p.db.get_message(message_id)
    if not m:
        raise HTTPException(404, detail=f"Message {message_id} not found")
    return m


@router.get("/messages/{message_id}/duplicates", tags=["duplicates"])
def message_duplicates(message_id: str, p: Pipeline = Depends(get_pipeline)):
    if not p.db.get_message(message_id):
        raise HTTPException(404, detail=f"Message {message_id} not found")
    return {"message_id": message_id, "suggestions": p.db.list_duplicates(message_id=message_id),
            "note": "Suggestions only. Similar reports can describe different incidents; a human must decide."}


@router.post("/duplicates/{suggestion_id}/decision", tags=["duplicates"], dependencies=[Depends(require_write)])
def decide_duplicate(suggestion_id: int, body: schemas.DuplicateDecision, p: Pipeline = Depends(get_pipeline)):
    try:
        p.db.decide_duplicate(suggestion_id, body.decision, body.reviewer)
    except KeyError as exc:
        raise HTTPException(404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc))
    return {"ok": True}


@router.patch("/messages/{message_id}/location", tags=["messages"], dependencies=[Depends(require_write)])
def patch_location(message_id: str, body: schemas.LocationPatch, p: Pipeline = Depends(get_pipeline)):
    try:
        p.db.set_location(message_id, body.name, body.lat, body.lon)
    except KeyError as exc:
        raise HTTPException(404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc))
    return p.db.get_message(message_id)


@router.get("/analytics/summary", tags=["analytics"])
def analytics_summary(p: Pipeline = Depends(get_pipeline)):
    return analytics.summary(p.db.messages_df(), p.db.counts())


@router.post("/reviews", status_code=201, tags=["review"], dependencies=[Depends(require_write)])
def create_review(body: schemas.ReviewIn, p: Pipeline = Depends(get_pipeline)):
    try:
        rid = p.db.add_review(body.message_id, body.reviewer, body.status, body.corrected_labels,
                              body.corrected_location, body.corrected_lat, body.corrected_lon, body.notes)
    except KeyError as exc:
        raise HTTPException(404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc))
    return {"review_id": rid}


@router.get("/evaluation", tags=["evaluation"])
def evaluation():
    f = Path(config.ARTIFACT_DIR) / "metrics.json"
    if not f.exists():
        raise HTTPException(404, detail="No evaluation report yet. Run: python -m training.train_classifier")
    return json.loads(f.read_text(encoding="utf-8"))
