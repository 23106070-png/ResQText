"""GDACS adapter. Official docs: GDACS API quick start (https://www.gdacs.org/Documents/2025/GDACS_API_quickstart_v2.pdf).
Endpoint: {base}/events/geteventlist/SEARCH  (GeoJSON, max 100 per page, `pagenumber` for more).
Terms: free; acknowledge the source as "Global Disaster Awareness and Coordination System, GDACS"."""
from __future__ import annotations

import re
import urllib.parse
from datetime import timedelta

import config
from ingestion.base import IngestContext, SourceAdapter
from ingestion.http import parse_json
from ingestion.models import FetchResult, NormalizedItem, SourceError, iso, parse_dt

EVENT_TYPES = {"FL": "Flood", "TC": "Tropical cyclone", "EQ": "Earthquake", "VO": "Volcano", "DR": "Drought",
               "WF": "Wildfire", "TS": "Tsunami"}
ATTRIBUTION = "Global Disaster Awareness and Coordination System, GDACS"
PAGE_SIZE = 100


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def parse_feature(feature: dict) -> NormalizedItem:
    """One GeoJSON feature -> NormalizedItem. Raises ValueError when essential fields are missing."""
    if not isinstance(feature, dict) or not isinstance(feature.get("properties"), dict):
        raise ValueError("feature has no properties")
    p = feature["properties"]
    etype, eid = p.get("eventtype"), p.get("eventid")
    if not etype or eid in (None, ""):
        raise ValueError("feature is missing eventtype/eventid")
    name = (p.get("name") or p.get("eventname") or "").strip() or f"{EVENT_TYPES.get(etype, etype)} event"
    html_desc = _clean(p.get("htmldescription") or "")
    sev = p.get("severitydata") or {}
    sev_text = (sev.get("severitytext") or "").strip()
    body = " ".join(x for x in [html_desc or _clean(p.get("description") or ""), f"Severity: {sev_text}." if sev_text else ""] if x)

    lat = lon = None
    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates") if isinstance(geom, dict) else None
    if geom.get("type") == "Point" and isinstance(coords, (list, tuple)) and len(coords) >= 2:
        try:
            lon, lat = float(coords[0]), float(coords[1])
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                lat = lon = None
        except (TypeError, ValueError):
            lat = lon = None
    precision = None
    if lat is not None:
        precision = "event_centroid" if "centroid" in str(p.get("Class", "Point_Centroid")).lower() else "precise"

    affected = p.get("affectedcountries") or []
    iso3s = {p.get("iso3")} | {a.get("iso3") for a in affected if isinstance(a, dict)}
    url = p.get("url") if isinstance(p.get("url"), dict) else {}
    level = p.get("alertlevel")
    country = (p.get("country") or "").strip() or None
    return NormalizedItem(
        source_name="GDACS", source_id=f"{etype}-{eid}", origin="official_event", title=name, body=body,
        url=url.get("report") or url.get("details"),
        published_at=iso(parse_dt(p.get("datemodified")) or parse_dt(p.get("fromdate"))),
        event_type=EVENT_TYPES.get(etype, str(etype)),
        severity=f"{level} (score {p.get('alertscore')})" if level else None,
        country=country, location_text=country, lat=lat, lon=lon, coord_precision=precision,
        language="en", verification_status="official_source", is_india="IND" in iso3s,
        expires_at=iso(parse_dt(p.get("todate"))), source_updated_at=iso(parse_dt(p.get("datemodified"))),
        attribution=ATTRIBUTION,
        extra={"episode_id": p.get("episodeid"), "event_from": p.get("fromdate"), "alert_level": level,
               "alert_score": p.get("alertscore"), "is_current": p.get("iscurrent"), "glide": p.get("glide") or None,
               "upstream_source": p.get("source"), "iso3": p.get("iso3") or None,
               "affected_countries": [a.get("countryname") for a in affected if isinstance(a, dict)],
               "details_url": url.get("details")})


class GdacsAdapter(SourceAdapter):
    name = "GDACS"
    label = "GDACS (global disaster alerts)"
    kind = "Official disaster events"
    homepage = "https://www.gdacs.org"
    attribution = ATTRIBUTION
    needs_credentials = False

    def interval_minutes(self) -> float:
        return max(config.GDACS_INTERVAL_MIN, 15)

    def configured(self) -> tuple[bool, str]:
        return (True, "") if config.GDACS_ENABLED else (False, "Disabled (GDACS_ENABLED=0).")

    QUERIES = (  # (event types, alert levels): significant events worldwide + low-level floods/cyclones for context
        ("EQ;TC;FL;VO;DR;WF;TS", "orange;red"),
        ("FL;TC;DR", "green"),
    )

    def _url(self, eventlist: str, levels: str, page: int, fromdate: str, todate: str) -> str:
        q = {"eventlist": eventlist, "fromdate": fromdate, "todate": todate, "alertlevel": levels,
             "pagenumber": page, "pagesize": PAGE_SIZE}
        return f"{config.GDACS_BASE_URL}/events/geteventlist/SEARCH?{urllib.parse.urlencode(q, safe=';')}"

    def fetch(self, ctx: IngestContext) -> FetchResult:
        res = FetchResult()
        frm = (ctx.now - timedelta(days=config.GDACS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        to = (ctx.now + timedelta(days=1)).strftime("%Y-%m-%d")
        seen: dict[str, NormalizedItem] = {}
        failures = 0
        for eventlist, levels in self.QUERIES:
            for page in range(1, max(config.GDACS_MAX_PAGES, 1) + 1):
                try:
                    resp = ctx.request(self.name, self._url(eventlist, levels, page, frm, to), what="GDACS search")
                    res.api_calls += 1
                    data = parse_json(resp, "GDACS search")
                except SourceError as exc:
                    failures += 1
                    res.errors.append({"kind": exc.kind, "message": exc.message, "item_ref": f"{eventlist}/{levels}/p{page}"})
                    if exc.kind in ("rate_limited", "auth"):
                        raise  # stop everything: retrying other queries would only hammer the service
                    break
                feats = data.get("features") if isinstance(data, dict) else None
                if not isinstance(feats, list):
                    failures += 1
                    res.errors.append({"kind": "invalid_response", "message": "GDACS response has no 'features' list.", "item_ref": f"p{page}"})
                    break
                for f in feats:
                    try:
                        item = parse_feature(f)
                    except ValueError as exc:
                        res.errors.append({"kind": "parse", "message": str(exc), "item_ref": None})
                        continue
                    old = seen.get(item.source_id)
                    if not old or (item.source_updated_at or "") >= (old.source_updated_at or ""):
                        seen[item.source_id] = item
                if len(feats) < PAGE_SIZE:
                    break
        if failures and not seen:
            raise SourceError("http_error", "GDACS returned no usable data (see logged errors).")
        res.items = list(seen.values())
        return res
