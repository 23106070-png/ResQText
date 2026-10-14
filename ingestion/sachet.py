"""NDMA Sachet adapter (India). Public RSS feeds + CAP 1.2 XML per alert.
Official integration guide (https://sachet.ndma.gov.in/docs/Integration_Guide_For_Agencies.pdf) REQUIRES ETag
client-side caching: send If-None-Match, treat 304 as 'unchanged', replace the cached ETag on 200. This adapter does."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import config
from ingestion.base import IngestContext, SourceAdapter, looks_non_english
from ingestion.models import FetchResult, NormalizedItem, SourceError, iso, parse_dt
from nlp.location_processing import get_resolver

CAP_NS = {"cap": "urn:oasis:names:tc:emergency:cap:1.2"}
ATTRIBUTION = "National Disaster Management Authority (NDMA), India: SACHET / Common Alerting Protocol feeds"


def safe_xml(body: bytes, what: str) -> ET.Element:
    """Parse XML while refusing DTD/entity tricks (billion-laughs); no extra dependency needed."""
    head = body[:4096].upper()
    if b"<!DOCTYPE" in head or b"<!ENTITY" in body.upper():
        raise SourceError("invalid_response", f"{what}: XML with a DTD/entity declaration was rejected.")
    try:
        return ET.fromstring(body)
    except ET.ParseError as exc:
        raise SourceError("invalid_response", f"{what}: the XML could not be parsed.") from exc


def parse_rss(body: bytes) -> list[dict]:
    root = safe_xml(body, "Sachet RSS")
    channel = root.find("channel")
    if channel is None:
        raise SourceError("invalid_response", "Sachet RSS: no <channel> element.")
    out = []
    for it in channel.findall("item"):
        g = lambda tag: (it.findtext(tag) or "").strip()  # noqa: E731
        author = g("author")
        m = re.search(r"\(([^)]+)\)", author)
        out.append({"guid": g("guid"), "title": g("title"), "link": g("link"), "category": g("category"),
                    "pubdate": g("pubDate"), "issuer": m.group(1) if m else author or None})
    return out


def parse_cap(body: bytes) -> dict:
    """CAP 1.2 alert -> dict with the English <info> block preferred."""
    root = safe_xml(body, "Sachet CAP")
    if not root.tag.endswith("alert"):
        raise SourceError("invalid_response", "Sachet CAP: root element is not <alert>.")
    t = lambda el, tag: (el.findtext(f"cap:{tag}", default="", namespaces=CAP_NS) or "").strip()  # noqa: E731
    infos = root.findall("cap:info", CAP_NS)
    chosen = next((i for i in infos if t(i, "language").lower().startswith("en")), infos[0] if infos else None)
    d = {"identifier": t(root, "identifier"), "sender": t(root, "sender"), "sent": t(root, "sent"),
         "status": t(root, "status"), "msg_type": t(root, "msgType"), "scope": t(root, "scope"), "info": None}
    if chosen is not None:
        areas = [a.findtext("cap:areaDesc", default="", namespaces=CAP_NS).strip() for a in chosen.findall("cap:area", CAP_NS)]
        d["info"] = {k: t(chosen, k) for k in ("language", "category", "event", "urgency", "severity", "certainty", "effective",
                                              "onset", "expires", "headline", "description", "instruction", "web")}
        d["info"]["areas"] = [a for a in areas if a]
    return d


class SachetAdapter(SourceAdapter):
    name = "NDMA Sachet"
    label = "NDMA Sachet (India official alerts)"
    kind = "Official alerts"
    homepage = "https://sachet.ndma.gov.in"
    attribution = ATTRIBUTION
    needs_credentials = False

    def interval_minutes(self) -> float:
        return max(config.SACHET_INTERVAL_MIN, 5)

    def configured(self) -> tuple[bool, str]:
        if not config.SACHET_ENABLED:
            return False, "Disabled (SACHET_ENABLED=0)."
        return (True, "") if config.SACHET_FEEDS else (False, "No feed URLs in SACHET_FEEDS.")

    def fetch(self, ctx: IngestContext) -> FetchResult:
        res = FetchResult()
        budget = max(config.SACHET_MAX_DETAIL_PER_RUN, 0)
        deferred = 0
        all_unchanged = True
        pending_cache: list[tuple[str, object]] = []
        for feed in config.SACHET_FEEDS:
            resp = ctx.request(self.name, feed, conditional=True, what="Sachet RSS")
            res.api_calls += 1
            if resp.status == 304:
                continue  # feed unchanged since last full, successful pass
            all_unchanged = False
            entries = parse_rss(resp.body)
            fully_handled = True
            for e in entries:
                if not e["guid"]:
                    res.errors.append({"kind": "parse", "message": "RSS item without guid skipped.", "item_ref": e["link"] or None})
                    continue
                if ctx.db.source_item_exists(self.name, e["guid"]):
                    continue  # already stored: don't re-download (reduces load on NDMA servers)
                if budget <= 0:
                    deferred += 1
                    fully_handled = False
                    continue
                budget -= 1
                try:
                    cap_resp = ctx.request(self.name, e["link"], conditional=True, what="Sachet CAP")
                    res.api_calls += 1
                    if cap_resp.status == 304:
                        continue
                    cap = parse_cap(cap_resp.body)
                    item = self._to_item(e, cap)
                    ctx.commit_cache(e["link"], cap_resp)
                except SourceError as exc:
                    if exc.kind == "rate_limited":
                        raise
                    res.errors.append({"kind": exc.kind, "message": exc.message, "item_ref": e["guid"]})
                    fully_handled = False
                    continue
                if item is None:
                    res.skipped += 1  # non-Actual status (Test/Exercise/Draft...)
                else:
                    res.items.append(item)
            if fully_handled and not res.errors:
                pending_cache.append((feed, resp))
        for feed, resp in pending_cache:
            ctx.commit_cache(feed, resp)
        if deferred:
            res.notes.append(f"{deferred} newer alert(s) deferred to the next run (SACHET_MAX_DETAIL_PER_RUN={config.SACHET_MAX_DETAIL_PER_RUN}).")
        res.not_modified = all_unchanged
        return res

    def _to_item(self, rss: dict, cap: dict) -> NormalizedItem | None:
        if cap["status"] and cap["status"].lower() != "actual":
            return None
        info = cap["info"] or {}
        headline = info.get("headline") or rss["title"]
        description = info.get("description") or ""
        body = "\n".join(x for x in [headline, description if description != headline else ""] if x).strip() or rss["title"]
        areas = info.get("areas") or []
        area_text = "; ".join(dict.fromkeys(areas)) or None
        lang = "non-en" if looks_non_english(body) else (info.get("language") or "en")
        sev = info.get("severity") or None
        return NormalizedItem(
            source_name=self.name, source_id=rss["guid"], origin="official_alert", title=headline, body=body, url=rss["link"],
            published_at=iso(parse_dt(rss["pubdate"]) or parse_dt(cap["sent"])),
            event_type=info.get("event") or rss.get("category") or None,
            severity=sev, country="India", location_text=area_text, language=lang,
            verification_status="official_source", is_india=True,
            expires_at=iso(parse_dt(info.get("expires"))), source_updated_at=iso(parse_dt(cap["sent"])),
            attribution=f"{ATTRIBUTION}. Issued by: {rss.get('issuer') or cap['sender'] or 'unknown'}.",
            extra={"cap_identifier": cap["identifier"], "sender": cap["sender"], "issuer": rss.get("issuer"),
                   "msg_type": cap["msg_type"], "urgency": info.get("urgency"), "certainty": info.get("certainty"),
                   "effective": info.get("effective"), "onset": info.get("onset"), "instruction": info.get("instruction"),
                   "web": info.get("web"), "areas": areas, "rss_category": rss.get("category")})
