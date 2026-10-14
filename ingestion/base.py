"""Adapter plumbing: the context every adapter receives, plus small shared helpers."""
from __future__ import annotations

import re
from datetime import datetime

import config
from database.db import Database
from ingestion.http import Fetcher, HttpRequest, HttpResponse, default_fetcher, check_status
from ingestion.models import FetchResult, SourceError, utc_now


class IngestContext:
    def __init__(self, db: Database, fetch: Fetcher | None = None, now: datetime | None = None):
        self.db = db
        self.fetch = fetch or default_fetcher
        self.now = now or utc_now()

    def request(self, source: str, url: str, *, method: str = "GET", headers: dict | None = None, body: bytes | None = None,
                conditional: bool = False, what: str = "") -> HttpResponse:
        """One network call. With conditional=True, sends If-None-Match / If-Modified-Since from the cache and treats
        304 as success. Every call is counted in api_usage."""
        hdrs = dict(headers or {})
        if conditional:
            cached = self.db.get_http_cache(url)
            if cached:
                if cached.get("etag"):
                    hdrs["If-None-Match"] = cached["etag"]
                if cached.get("last_modified"):
                    hdrs["If-Modified-Since"] = cached["last_modified"]
        self.db.add_api_calls(source, 1)
        resp = self.fetch(HttpRequest(url, method, hdrs, body, config.HTTP_TIMEOUT))
        if resp.status == 304 and conditional:
            return resp
        check_status(resp, what or source)
        return resp

    def commit_cache(self, url: str, resp: HttpResponse) -> None:
        """Remember ETag/Last-Modified. Adapters call this only once everything in the response was handled, so a
        half-processed feed is fetched again next time instead of being skipped as 'unchanged'."""
        etag, lm = resp.headers.get("etag"), resp.headers.get("last-modified")
        if etag or lm:
            self.db.set_http_cache(url, etag, lm)


class SourceAdapter:
    name = ""
    label = ""
    kind = ""            # shown in the UI
    homepage = ""
    attribution = ""
    needs_credentials = False
    credential_help = ""

    def interval_minutes(self) -> float:
        raise NotImplementedError

    def configured(self) -> tuple[bool, str]:
        raise NotImplementedError

    def fetch(self, ctx: IngestContext) -> FetchResult:
        raise NotImplementedError


_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MD_NOISE = re.compile(r"[#*_`>]+")


def strip_markdown(text: str) -> str:
    text = _MD_LINK.sub(r"\1", text or "")
    text = _MD_NOISE.sub(" ", text)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def looks_non_english(text: str) -> bool:
    """Cheap script check: many Sachet alerts are in Hindi, Telugu, Bengali, Punjabi... which the English NLP can't read."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    return sum(ord(ch) > 0x024F for ch in letters) / len(letters) > 0.3
