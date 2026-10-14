"""Endpoints for imported real-world data: reports, official events, source health, refresh."""
from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request

import analytics
import config
import reports as rp
import security
from api import schemas
from api.routes.core import get_pipeline, require_write
from ingestion.service import IngestionService
from nlp.pipeline import Pipeline

router = APIRouter()
EVENT_ORIGINS = ["official_event", "official_alert"]


def get_service(request: Request) -> IngestionService:
    return request.app.state.service


def _authorised(x_api_key: str | None) -> bool:
    ok, _, _ = security.api_write_decision(x_api_key)
    return ok and security.api_mode() != "locked"


def _frame(p: Pipeline):
    return rp.add_labels(p.db.messages_df())


@router.get("/reports", tags=["real-world"], summary="List imported and submitted reports (filter + paginate)")
def list_reports(p: Pipeline = Depends(get_pipeline),
                 origin: Optional[list[str]] = Query(None, description="user_report | situation_report | official_event | official_alert"),
                 source: Optional[str] = None, event_type: Optional[str] = None, category: Optional[str] = None,
                 location: Optional[str] = None, country: Optional[str] = None, verification: Optional[str] = None,
                 review_status: Optional[str] = None, nlp_status: Optional[str] = None, india_only: bool = False,
                 date_from: Optional[date] = None, date_to: Optional[date] = None, include_expired: bool = True,
                 q: Optional[str] = Query(None, max_length=200), limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    if category and category not in config.LABELS:
        raise HTTPException(422, detail=f"category must be one of {config.LABELS}")
    if date_from and date_to and date_from > date_to:
        raise HTTPException(422, detail="date_from must not be after date_to")
    df = rp.filter_reports(_frame(p), origins=origin, source=source, event_type=event_type, category=category, location=location,
                           country=country, verification=verification, review_status=review_status, nlp_status=nlp_status,
                           india_only=india_only, date_from=date_from, date_to=date_to, include_expired=include_expired, search=q)
    total, page = rp.paginate(df.sort_values("ts", ascending=False), limit, offset)
    return {"total": total, "limit": limit, "offset": offset, "items": rp.to_public_records(page)}


@router.get("/reports/{message_id}", tags=["real-world"])
def get_report(message_id: str, p: Pipeline = Depends(get_pipeline)):
    m = p.db.get_message(message_id)
    if not m:
        raise HTTPException(404, detail=f"Report {message_id} not found")
    for k in ("content_hash", "pending_update_json"):
        m.pop(k, None)
    m["observed_vs_inferred"] = {
        "observed_from_source": ["title", "original_text", "published_at", "source_url", "event_type", "severity", "country", "location_text"],
        "model_generated": ["predicted_labels", "classifications", "entities", "summary"],
    }
    return m


@router.get("/events", tags=["real-world"], summary="Official disaster events and alerts (with map coordinates when the source gave them)")
def list_events(p: Pipeline = Depends(get_pipeline), source: Optional[str] = None, event_type: Optional[str] = None,
                india_only: bool = False, include_expired: bool = False, mapped_only: bool = False,
                limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
    df = rp.filter_reports(_frame(p), origins=EVENT_ORIGINS, source=source, event_type=event_type, india_only=india_only,
                           include_expired=include_expired)
    if mapped_only and not df.empty:
        df = df.dropna(subset=["lat", "lon"])
    total, page = rp.paginate(df.sort_values("ts", ascending=False), limit, offset)
    return {"total": total, "limit": limit, "offset": offset, "items": rp.to_public_records(page),
            "note": "Map coordinates of events are approximate centroids; they never identify a person's location."}


@router.get("/sources/status", tags=["sources"])
def sources_status(service: IngestionService = Depends(get_service), x_api_key: Optional[str] = Header(default=None)):
    detail = _authorised(x_api_key)
    out = []
    for s in service.source_status():
        if not detail:
            s = {**s, "last_error": "hidden" if s["last_error"] else None, "config_note": s["config_note"] if not s["needs_credentials"] else "Credentials required"}
        out.append(s)
    return {"sources": out, "detail_level": "full" if detail else "public"}


@router.get("/ingestion/runs", tags=["sources"], dependencies=[Depends(require_write)])
def ingestion_runs(service: IngestionService = Depends(get_service), source: Optional[str] = None, limit: int = Query(20, ge=1, le=200)):
    return {"runs": service.db.recent_runs(source, limit), "errors": service.db.recent_errors(source, limit)}


@router.post("/ingestion/refresh", tags=["sources"], dependencies=[Depends(require_write)],
             summary="Fetch new data now (respects each source's minimum interval and quotas)")
def refresh(body: schemas.RefreshRequest, service: IngestionService = Depends(get_service)):
    if body.sources:
        bad = [s for s in body.sources if s not in service.adapters]
        if bad:
            raise HTTPException(422, detail=f"Unknown source(s): {bad}. Known: {list(service.adapters)}")
    return {"results": service.refresh(body.sources, trigger="manual", force=body.force)}


@router.post("/reports/{message_id}/apply-update", tags=["real-world"], dependencies=[Depends(require_write)])
def apply_update(message_id: str, body: schemas.ApplyUpdateRequest, p: Pipeline = Depends(get_pipeline)):
    if not p.db.get_message(message_id):
        raise HTTPException(404, detail=f"Report {message_id} not found")
    if not p.db.apply_pending_update(message_id, body.reviewer):
        raise HTTPException(409, detail="This report has no newer source version waiting.")
    return {"ok": True}


@router.get("/analytics/real-world", tags=["analytics"])
def realworld_analytics(p: Pipeline = Depends(get_pipeline), service: IngestionService = Depends(get_service)):
    df = _frame(p)
    return {"by_origin": df["origin"].value_counts().to_dict() if not df.empty else {},
            "by_source": df["source_name"].dropna().value_counts().to_dict() if not df.empty else {},
            "by_verification": df["verification"].value_counts().to_dict() if not df.empty else {},
            "needs_by_category": analytics.by_category(rp.filter_reports(df, origins=list(config.NEED_ORIGINS))).to_dict("records") if not df.empty else [],
            "caveat": analytics.CAVEAT}
