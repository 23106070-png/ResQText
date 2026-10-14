"""ReliefWeb API v2 adapter. Official docs: https://apidoc.reliefweb.int
* Requires a PRE-APPROVED `appname` (since 1 Nov 2025): https://apidoc.reliefweb.int/parameters#appname
* Quotas: max 1000 entries per call, max 1000 calls per day. No fees.
* Reports can contain copyrighted material from the original source: we store a short excerpt and link back."""
from __future__ import annotations

import json
import urllib.parse
from datetime import timedelta

import config
from ingestion.base import IngestContext, SourceAdapter, looks_non_english, strip_markdown
from ingestion.http import parse_json
from ingestion.models import FetchResult, NormalizedItem, SourceError, iso, parse_dt

FIELDS = ["title", "body", "url", "date.created", "date.original", "date.changed", "source.name", "source.shortname",
          "country.name", "country.iso3", "primary_country.name", "primary_country.iso3", "disaster.name", "disaster.type.name",
          "disaster_type.name", "format.name", "language.code"]


def _names(value, key="name") -> list[str]:
    if isinstance(value, dict):
        value = [value]
    return [str(v.get(key)) for v in value or [] if isinstance(v, dict) and v.get(key)]


def parse_report(entry: dict) -> NormalizedItem:
    if not isinstance(entry, dict) or "id" not in entry or not isinstance(entry.get("fields"), dict):
        raise ValueError("entry has no id/fields")
    f = entry["fields"]
    title = (f.get("title") or "").strip()
    if not title:
        raise ValueError("report has no title")
    text = strip_markdown(f.get("body") or "")[: config.RELIEFWEB_MAX_STORED_CHARS]
    dates = f.get("date") if isinstance(f.get("date"), dict) else {}
    countries = f.get("country") or []
    iso3s = {c.get("iso3") for c in countries if isinstance(c, dict)}
    primary = _names(f.get("primary_country")) or _names(countries)
    disasters = f.get("disaster") or []
    dtypes = _names(f.get("disaster_type")) or [n for d in disasters if isinstance(d, dict) for n in _names(d.get("type"))]
    sources = _names(f.get("source"))
    langs = _names(f.get("language"), "code")
    lang = langs[0] if langs else None
    if lang == "en" and looks_non_english(title + " " + text):
        lang = "non-en"
    return NormalizedItem(
        source_name="ReliefWeb", source_id=str(entry["id"]), origin="situation_report", title=title, body=text,
        url=f.get("url"), published_at=iso(parse_dt(dates.get("original")) or parse_dt(dates.get("created"))),
        event_type=dtypes[0] if dtypes else None, severity=None,
        country=", ".join(dict.fromkeys(_names(countries))) or None, location_text=", ".join(primary) or None,
        language=lang, verification_status="source_reported", is_india="IND" in iso3s or "India" in primary,
        source_updated_at=iso(parse_dt(dates.get("changed"))),
        attribution="ReliefWeb (UN OCHA). Original publisher: " + (", ".join(sources) or "unknown") +
                    ". Content may be copyrighted by the original publisher.",
        extra={"publishers": sources, "disasters": _names(disasters), "formats": _names(f.get("format")), "reliefweb_id": entry["id"]})


class ReliefWebAdapter(SourceAdapter):
    name = "ReliefWeb"
    label = "ReliefWeb (humanitarian situation reports)"
    kind = "Situation reports"
    homepage = "https://reliefweb.int"
    attribution = "ReliefWeb (UN OCHA)"
    needs_credentials = True
    credential_help = "Request a pre-approved appname at https://apidoc.reliefweb.int/parameters#appname, then set RELIEFWEB_APPNAME."

    def interval_minutes(self) -> float:
        return max(config.RELIEFWEB_INTERVAL_MIN, 30)

    def configured(self) -> tuple[bool, str]:
        if not config.RELIEFWEB_ENABLED:
            return False, "Disabled (RELIEFWEB_ENABLED=0)."
        if not config.RELIEFWEB_APPNAME:
            return False, "Needs an approved appname (RELIEFWEB_APPNAME is not set)."
        return True, ""

    def _body(self, ctx: IngestContext, offset: int) -> dict:
        conds = [{"field": "date.created", "value": {"from": iso(ctx.now - timedelta(days=config.RELIEFWEB_LOOKBACK_DAYS)) + "+00:00"}}]
        if config.RELIEFWEB_COUNTRY:
            conds.append({"field": "country", "value": config.RELIEFWEB_COUNTRY})
        return {"preset": "latest", "limit": config.RELIEFWEB_PAGE_SIZE, "offset": offset,
                "filter": {"operator": "AND", "conditions": conds}, "fields": {"include": FIELDS}}

    def fetch(self, ctx: IngestContext) -> FetchResult:
        ok, why = self.configured()
        if not ok:
            raise SourceError("not_configured", why)
        res = FetchResult()
        url = f"{config.RELIEFWEB_BASE_URL}/reports?appname={urllib.parse.quote(config.RELIEFWEB_APPNAME, safe='')}"
        offset = 0
        for _ in range(max(config.RELIEFWEB_MAX_PAGES, 1)):
            if ctx.db.api_calls_today(self.name) >= config.RELIEFWEB_DAILY_CALL_BUDGET:
                res.notes.append(f"Daily call budget reached ({config.RELIEFWEB_DAILY_CALL_BUDGET}); stopping to respect the 1000/day quota.")
                break
            resp = ctx.request(self.name, url, method="POST", headers={"Content-Type": "application/json"},
                               body=json.dumps(self._body(ctx, offset)).encode("utf-8"), what="ReliefWeb reports")
            res.api_calls += 1
            data = parse_json(resp, "ReliefWeb reports")
            entries = data.get("data") if isinstance(data, dict) else None
            if not isinstance(entries, list):
                raise SourceError("invalid_response", "ReliefWeb response has no 'data' list.")
            for e in entries:
                try:
                    res.items.append(parse_report(e))
                except ValueError as exc:
                    res.errors.append({"kind": "parse", "message": str(exc), "item_ref": str(e.get("id")) if isinstance(e, dict) else None})
            if len(entries) < config.RELIEFWEB_PAGE_SIZE:
                break
            offset += len(entries)
        return res
