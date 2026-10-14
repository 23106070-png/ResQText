"""Source parsing + adapters, with mocked HTTP. Real captured samples are used for GDACS and Sachet;
ReliefWeb fixtures follow the documented field names with invented values."""
import json
from datetime import datetime

import pytest

import config
from database.db import Database
from ingestion.base import IngestContext, looks_non_english, strip_markdown
from ingestion.gdacs import GdacsAdapter, parse_feature
from ingestion.http import HttpResponse
from ingestion.models import SourceError, parse_dt
from ingestion.reliefweb import ReliefWebAdapter, parse_report
from ingestion.sachet import SachetAdapter, parse_cap, parse_rss, safe_xml
from tests.netmock import FakeNet, R, fixture, sachet_net

NOW = datetime(2026, 10, 13, 12, 0, 0)


@pytest.fixture()
def ctx_factory(db):
    def make(net):
        return IngestContext(db, net, NOW)
    return make


# ------------------------------------------------------------------ time helper
def test_parse_dt_variants():
    assert parse_dt("2026-10-11T20:43:05+05:30").isoformat() == "2026-10-11T15:13:05"
    assert parse_dt("Sun, 11 Oct 2026 15:13:07 GMT").isoformat() == "2026-10-11T15:13:07"
    assert parse_dt("2026-10-11T12:05:31").isoformat() == "2026-10-11T12:05:31"
    assert parse_dt("2026-10-11T12:05:31Z").isoformat() == "2026-10-11T12:05:31"
    assert parse_dt("garbage") is None and parse_dt("") is None and parse_dt(None) is None


# ------------------------------------------------------------------ GDACS
def test_gdacs_parse_real_feature():
    feats = json.loads(fixture("gdacs_search.json"))["features"]
    tc = parse_feature(feats[1])
    assert tc.source_id == "TC-1001334" and tc.origin == "official_event" and tc.event_type == "Tropical cyclone"
    assert tc.attribution == "Global Disaster Awareness and Coordination System, GDACS"
    assert tc.url.startswith("https://www.gdacs.org/report.aspx")
    assert (tc.lat, tc.lon) == (37.5, -84.0) and tc.coord_precision == "event_centroid"
    assert tc.published_at == "2026-10-11T20:13:33" and tc.verification_status == "official_source"
    assert "Orange" in tc.severity and not tc.is_india


def test_gdacs_india_flag_and_missing_fields():
    feats = json.loads(fixture("gdacs_search.json"))["features"]
    assert parse_feature(feats[3]).is_india is True
    with pytest.raises(ValueError):
        parse_feature({"type": "Feature"})
    with pytest.raises(ValueError):
        parse_feature({"properties": {"name": "no id"}})
    f = json.loads(fixture("gdacs_malformed.json"))["features"][2]
    item = parse_feature(f)  # out-of-range coordinates must be dropped, never kept or "fixed"
    assert item.lat is None and item.lon is None and item.coord_precision is None
    f["geometry"] = None
    assert parse_feature(f).lat is None


def test_gdacs_fetch_dedupes_and_records_parse_errors(ctx_factory):
    net = FakeNet().add("geteventlist/SEARCH", R(200, fixture("gdacs_search.json")))
    res = GdacsAdapter().fetch(ctx_factory(net))
    ids = sorted(i.source_id for i in res.items)
    assert ids == ["EQ-1570859", "FL-1104220", "FL-9999001", "TC-1001334"]  # same event in both queries -> once
    assert len(net.calls) == 2  # one page per query: fewer than 100 results means no next page
    assert "pagenumber=1" in net.calls[0].url and "eventlist=EQ;TC;FL;VO;DR;WF;TS" in net.calls[0].url


def test_gdacs_pagination_requests_next_page(ctx_factory):
    base = json.loads(fixture("gdacs_search.json"))["features"][0]
    def page(n):
        feats = []
        for i in range(100):
            f = json.loads(json.dumps(base)); f["properties"]["eventid"] = 5000 + n * 100 + i; feats.append(f)
        return {"type": "FeatureCollection", "features": feats}
    net = FakeNet().add("pagenumber=1&", R(200, page(1))).add("pagenumber=2&", R(200, {"features": []}))
    res = GdacsAdapter().fetch(ctx_factory(net))
    assert len(res.items) == 100 and any("pagenumber=2" in u for u in net.urls())


def test_gdacs_malformed_and_http_failures(ctx_factory):
    net = FakeNet().add("SEARCH", R(200, "not json {"))
    with pytest.raises(SourceError) as e:
        GdacsAdapter().fetch(ctx_factory(net))
    assert e.value.kind in ("http_error", "invalid_response")
    net = FakeNet().add("SEARCH", R(200, {"unexpected": 1}))
    with pytest.raises(SourceError):
        GdacsAdapter().fetch(ctx_factory(net))
    net = FakeNet().add("SEARCH", R(429, b"", retry_after="120"))
    with pytest.raises(SourceError) as e:
        GdacsAdapter().fetch(ctx_factory(net))
    assert e.value.kind == "rate_limited" and e.value.retry_after == 120


def test_gdacs_timeout(ctx_factory):
    net = FakeNet().add("SEARCH", SourceError("timeout", "The request timed out."))
    with pytest.raises(SourceError) as e:
        GdacsAdapter().fetch(ctx_factory(net))
    assert e.value.kind in ("timeout", "http_error")


# ------------------------------------------------------------------ Sachet
def test_sachet_parse_rss_and_cap_real_samples():
    items = parse_rss(fixture("sachet_rss.xml"))
    assert len(items) == 4 and items[0]["guid"] == "1791731455698013" and items[0]["issuer"] == "CWC"
    cap = parse_cap(fixture("sachet_cap_flood.xml"))
    info = cap["info"]
    assert info["language"] == "en-IN" and info["event"] == "Flood" and info["severity"] == "Moderate"  # English block preferred
    assert info["areas"] == ["SAI, Rae-Bareli, Rae-Bareli, Uttar Pradesh"] and cap["status"] == "Actual"


def test_sachet_rejects_malformed_and_dtd_xml():
    with pytest.raises(SourceError):
        safe_xml(b"<rss><channel>", "x")
    with pytest.raises(SourceError):
        safe_xml(b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "boom">]><rss>&a;</rss>', "x")
    with pytest.raises(SourceError):
        parse_rss(b"<rss></rss>")
    with pytest.raises(SourceError):
        parse_cap(b"<notcap/>")
    assert parse_cap(b'<cap:alert xmlns:cap="urn:oasis:names:tc:emergency:cap:1.2"/>')["info"] is None  # missing info tolerated


def test_sachet_fetch_normalises_and_skips_tests(ctx_factory):
    res = SachetAdapter().fetch(ctx_factory(sachet_net()))
    by = {i.source_id: i for i in res.items}
    assert set(by) == {"1791731455698013", "1791732541708028", "1791720898008028"} and res.skipped == 1  # 'Test' status skipped
    f = by["1791731455698013"]
    assert f.event_type == "Flood" and f.severity == "Moderate" and f.is_india and f.origin == "official_alert"
    assert f.published_at == "2026-10-11T15:13:07" and f.expires_at == "2026-10-12T03:30:00"
    assert "NDMA" in f.attribution and "CWC" in f.attribution and f.url.endswith("1791731455698013")
    assert f.language == "en-IN" and by["1791720898008028"].language == "non-en"


def test_sachet_uses_etag_and_304(db, ctx_factory):
    ctx = ctx_factory(sachet_net())
    SachetAdapter().fetch(ctx)
    assert db.get_http_cache(config.SACHET_FEEDS[0])["etag"] == '"feed-1"'
    net2 = FakeNet().add("rss_india.xml", R(304))
    res = SachetAdapter().fetch(ctx_factory(net2))
    assert res.not_modified and res.items == []
    assert net2.calls[0].headers["If-None-Match"] == '"feed-1"'  # the official guide's mandatory behaviour


def test_sachet_does_not_refetch_stored_alerts(db, pipe, ctx_factory):
    ad, net = SachetAdapter(), sachet_net()
    for it in ad.fetch(ctx_factory(net)).items:
        pipe.process_external(it)
    net2 = FakeNet().add("rss_india.xml", R(200, fixture("sachet_rss.xml"), etag='"feed-2"'))
    net2.add("identifier=1000000000000001", R(200, fixture("sachet_cap_test.xml")))  # only the skipped Test alert is re-checked
    res = ad.fetch(ctx_factory(net2))
    assert res.items == [] and len([u for u in net2.urls() if "FetchXMLFile" in u]) == 1


def test_sachet_detail_cap_defers_and_keeps_feed_uncached(db, monkeypatch, ctx_factory):
    monkeypatch.setattr(config, "SACHET_MAX_DETAIL_PER_RUN", 1)
    res = SachetAdapter().fetch(ctx_factory(sachet_net()))
    assert len(res.items) == 1 and any("deferred" in n for n in res.notes)
    assert db.get_http_cache(config.SACHET_FEEDS[0]) is None  # unfinished feed must not be marked 'unchanged'


def test_sachet_detail_error_is_isolated(db, ctx_factory):
    net = sachet_net()
    net.routes.insert(0, ["identifier=1791732541708028", None, [R(500)]])
    res = SachetAdapter().fetch(ctx_factory(net))
    assert len(res.items) == 2 and any(e["kind"] == "http_error" for e in res.errors)
    assert db.get_http_cache(config.SACHET_FEEDS[0]) is None


def test_non_english_detector():
    assert looks_non_english("अगले 3 घंटों में बारिश") and not looks_non_english("Heavy rain expected") and not looks_non_english("")


# ------------------------------------------------------------------ ReliefWeb
def test_reliefweb_needs_appname_and_makes_no_call(monkeypatch, ctx_factory):
    monkeypatch.setattr(config, "RELIEFWEB_APPNAME", "")
    ok, why = ReliefWebAdapter().configured()
    assert not ok and "RELIEFWEB_APPNAME" in why
    net = FakeNet()
    with pytest.raises(SourceError) as e:
        ReliefWebAdapter().fetch(ctx_factory(net))
    assert e.value.kind == "not_configured" and net.calls == []


def test_reliefweb_request_shape_and_parsing(monkeypatch, ctx_factory):
    monkeypatch.setattr(config, "RELIEFWEB_APPNAME", "acme-test-abc123")
    net = FakeNet().add("/reports", R(200, fixture("reliefweb_reports.json")), method="POST")
    res = ReliefWebAdapter().fetch(ctx_factory(net))
    req = net.calls[0]
    body = json.loads(req.body)
    assert req.url == "https://api.reliefweb.int/v2/reports?appname=acme-test-abc123" and req.method == "POST"
    assert body["limit"] == config.RELIEFWEB_PAGE_SIZE and body["offset"] == 0
    assert {"field": "country", "value": "India"} in body["filter"]["conditions"]
    assert len(res.items) == 2 and len(res.errors) == 1  # title-less entry reported, not crashed on
    it = res.items[0]
    assert it.source_id == "4000001" and it.origin == "situation_report" and it.is_india
    assert it.published_at == "2026-10-11T18:30:00" and it.event_type == "Flood" and it.verification_status == "source_reported"
    assert "Test Relief Org" in it.attribution and "copyrighted" in it.attribution
    assert "**" not in it.body and "example.org" not in it.body and "Read more" in it.body  # markdown stripped


def test_reliefweb_truncates_long_text_and_flags_language(monkeypatch):
    monkeypatch.setattr(config, "RELIEFWEB_MAX_STORED_CHARS", 50)
    it = parse_report({"id": 1, "fields": {"title": "T", "body": "word " * 100, "language": [{"code": "en"}]}})
    assert len(it.body) <= 50
    with pytest.raises(ValueError):
        parse_report({"id": 1})
    with pytest.raises(ValueError):
        parse_report({"id": 1, "fields": {"title": " "}})
    assert strip_markdown("# H\n[a](http://x) **b**") == "H\na b"


def test_reliefweb_pagination_budget_and_errors(monkeypatch, db, ctx_factory):
    monkeypatch.setattr(config, "RELIEFWEB_APPNAME", "acme-test-abc123")
    monkeypatch.setattr(config, "RELIEFWEB_PAGE_SIZE", 2)
    page = lambda a: {"data": [{"id": a, "fields": {"title": f"R{a}"}}, {"id": a + 1, "fields": {"title": f"R{a + 1}"}}]}
    net = FakeNet().add("/reports", R(200, page(1)), R(200, page(3)), R(200, {"data": []}), method="POST")
    res = ReliefWebAdapter().fetch(ctx_factory(net))
    assert len(res.items) == 4 and [json.loads(c.body)["offset"] for c in net.calls] == [0, 2, 4]
    monkeypatch.setattr(config, "RELIEFWEB_DAILY_CALL_BUDGET", 0)
    net = FakeNet()
    res = ReliefWebAdapter().fetch(ctx_factory(net))
    assert net.calls == [] and any("budget" in n for n in res.notes)
    monkeypatch.setattr(config, "RELIEFWEB_DAILY_CALL_BUDGET", 800)
    for status, kind in [(403, "auth"), (429, "rate_limited"), (503, "http_error")]:
        with pytest.raises(SourceError) as e:
            ReliefWebAdapter().fetch(ctx_factory(FakeNet().add("/reports", R(status))))
        assert e.value.kind == kind and "acme-test-abc123" not in e.value.message  # secret never appears in errors
    with pytest.raises(SourceError) as e:
        ReliefWebAdapter().fetch(ctx_factory(FakeNet().add("/reports", R(200, "<html>"))))
    assert e.value.kind == "invalid_response"
