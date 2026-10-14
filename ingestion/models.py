"""Common internal format for everything we import, plus the error type shared by all adapters."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

ORIGINS = ("user_report", "situation_report", "official_event", "official_alert")
ORIGIN_LABELS = {
    "user_report": "Submitted report",
    "situation_report": "Situation report (ReliefWeb)",
    "official_event": "Official disaster event",
    "official_alert": "Official alert",
}


class SourceError(Exception):
    """A classified failure from a data source. `kind` is one of:
    not_configured, auth, rate_limited, timeout, network, http_error, invalid_response, parse."""

    def __init__(self, kind: str, message: str, status: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.kind, self.message, self.status, self.retry_after = kind, message, status, retry_after


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def iso(dt: datetime | None) -> str | None:
    return dt.replace(microsecond=0).isoformat() if dt else None


def parse_dt(value) -> datetime | None:
    """Parse ISO-8601 (with or without offset/Z) or RFC-822 into a naive UTC datetime. None if unreadable.
    Naive inputs are assumed to already be UTC (GDACS documents UTC times)."""
    if value in (None, ""):
        return None
    text = str(value).strip()
    dt = None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


@dataclass
class NormalizedItem:
    source_name: str            # e.g. "GDACS"
    source_id: str              # stable id within that source
    origin: str                 # see ORIGINS
    title: str
    body: str                   # source text kept for analysis (may be an excerpt)
    url: str | None
    published_at: str | None    # UTC, ISO
    event_type: str | None = None
    severity: str | None = None
    country: str | None = None
    location_text: str | None = None
    lat: float | None = None
    lon: float | None = None
    coord_precision: str | None = None   # event_centroid | area_centroid | precise
    language: str | None = None
    verification_status: str = "unverified"   # official_source | source_reported | unverified
    is_india: bool = False
    expires_at: str | None = None
    source_updated_at: str | None = None
    attribution: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def dedupe_key(self) -> str:
        return f"{self.source_name}:{self.source_id}"

    def fingerprint(self) -> str:
        """Hash of the source-owned content; used to tell 'unchanged' from 'source updated this item'."""
        parts = [self.title, self.body, self.severity, self.expires_at, self.source_updated_at, self.published_at,
                 self.lat, self.lon]
        return hashlib.sha256(json.dumps(parts, default=str, ensure_ascii=False).encode("utf-8")).hexdigest()


@dataclass
class FetchResult:
    items: list[NormalizedItem] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)   # {"kind","message","item_ref"} non-fatal, per-item problems
    api_calls: int = 0
    not_modified: bool = False
    notes: list[str] = field(default_factory=list)
    skipped: int = 0                                    # items deliberately ignored (tests, expired, filtered)
