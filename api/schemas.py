"""Pydantic request/response models."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Raw report text. Never modified.")
    save: bool = Field(False, description="Also store the report and run duplicate detection.")
    timestamp: Optional[str] = Field(None, description="ISO 8601 time the report was sent.")
    source: str = "api"
    is_simulated: bool = False
    location_hint: Optional[str] = None

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        if len(v) > 5000:
            raise ValueError("text must be at most 5000 characters")
        return v


class LocationOut(BaseModel):
    name: str
    status: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    source: Optional[str] = None
    note: Optional[str] = None


class EntityOut(BaseModel):
    type: str
    value: str
    evidence: str
    start: int
    end: int
    source: str


class AnalysisOut(BaseModel):
    original_text: str
    normalized_text: str
    labels: list[str]
    scores: dict[str, float]
    confidence: float
    low_confidence: bool
    confidence_note: str
    location: LocationOut
    other_locations: list[str]
    people_mentioned: str
    people_note: str
    resources: list[str]
    entities: list[EntityOut]
    needs_verification: bool
    model_name: str


class AnalyzeResponse(BaseModel):
    analysis: AnalysisOut
    message_id: Optional[str] = None
    created: Optional[bool] = None
    duplicates: list[dict] = []


class BatchItem(BaseModel):
    text: str
    timestamp: Optional[str] = None
    source: str = "api_batch"
    external_id: Optional[str] = None
    is_simulated: bool = False
    location_hint: Optional[str] = None


class BatchRequest(BaseModel):
    messages: list[BatchItem] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    inserted: int
    skipped_existing: int
    duplicate_suggestions: int
    failed: list[dict]


class ReviewIn(BaseModel):
    message_id: str
    reviewer: str = Field(..., min_length=1)
    status: str = Field(..., description="approved (verified) | corrected | rejected (incorrect) | needs_info | duplicate")
    corrected_labels: Optional[list[str]] = None
    corrected_location: Optional[str] = None
    corrected_lat: Optional[float] = None
    corrected_lon: Optional[float] = None
    notes: str = ""


class LocationPatch(BaseModel):
    name: str = Field(..., min_length=1)
    lat: Optional[float] = None
    lon: Optional[float] = None


class DuplicateDecision(BaseModel):
    decision: str = Field(..., description="confirmed | rejected | pending")
    reviewer: str = Field(..., min_length=1)


class RefreshRequest(BaseModel):
    sources: Optional[list[str]] = Field(None, description="Source names, e.g. ['GDACS']. Omit for all configured sources.")
    force: bool = Field(False, description="Ignore the normal interval (a short minimum still applies).")


class ApplyUpdateRequest(BaseModel):
    reviewer: str = Field(..., min_length=1)
