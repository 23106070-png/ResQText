from datetime import datetime

import pytest
from fastapi.testclient import TestClient

import config
import reports as rp
import security
from api.main import create_app
from ingestion.service import IngestionService
from tests.netmock import Clock, FakeNet, R, fixture, sachet_net


@pytest.fixture()
def seeded(pipe, monkeypatch):
    monkeypatch.setattr(rp, "utc_now", lambda: datetime(2026, 10, 13, 12, 0, 0))  # freeze 'now' so expiry results are stable
    monkeypatch.setattr(config, "RELIEFWEB_APPNAME", "acme-test-abc123")
    net = sachet_net()
    net.add("SEARCH", R(200, fixture("gdacs_search.json")))
    net.add("/reports", R(200, fixture("reliefweb_reports.json")), method="POST")
    svc = IngestionService(pipe.db, pipe, net, now_fn=Clock(datetime(2026, 10, 13, 12, 0, 0)))
    svc.refresh(force=True)
    pipe.ingest("Five people need drinking water in Kalwa", "2026-10-12T10:00:00", is_simulated=True)
    return pipe, svc


@pytest.fixture()
def client(seeded):
    pipe, svc = seeded
    return TestClient(create_app(pipe, svc))


# ------------------------------------------------------------------ listing, filtering, pagination
def test_reports_listing_and_source_attribution_fields(client):
    j = client.get("/reports").json()
    assert j["total"] == 10  # 4 GDACS + 3 Sachet + 2 ReliefWeb + 1 submitted
    item = next(i for i in j["items"] if i["id"] and i["source_name"] == "GDACS" and i["title"].startswith("Tropical"))
    for k in ("source_url", "published_at", "retrieved_at", "attribution", "verification", "nlp_status", "origin"):
        assert item[k]
    assert item["verification"] == "official_source" and item["coord_precision"] == "event_centroid"


@pytest.mark.parametrize("params,expected", [
    ({"origin": "official_alert"}, 3), ({"origin": ["official_event", "official_alert"]}, 7), ({"source": "ReliefWeb"}, 2),
    ({"event_type": "flood"}, 4), ({"india_only": True}, 6), ({"verification": "source_reported"}, 2),
    ({"nlp_status": "skipped_official"}, 7), ({"q": "thunderstorm"}, 1), ({"country": "Panama"}, 1),
    ({"date_from": "2026-10-12", "date_to": "2026-10-12"}, 3), ({"review_status": "pending"}, 10), ({"include_expired": False, "origin": "official_alert"}, 1),
])
def test_report_filters(client, params, expected):
    assert client.get("/reports", params=params).json()["total"] == expected


def test_pagination_is_consistent(client):
    a = client.get("/reports", params={"limit": 4, "offset": 0}).json()
    b = client.get("/reports", params={"limit": 4, "offset": 4}).json()
    c = client.get("/reports", params={"limit": 4, "offset": 8}).json()
    ids = [i["id"] for p in (a, b, c) for i in p["items"]]
    assert len(ids) == 10 and len(set(ids)) == 10 and a["total"] == b["total"] == 10
    assert client.get("/reports", params={"limit": 4, "offset": 50}).json()["items"] == []
    assert client.get("/reports", params={"limit": 0}).status_code == 422 and client.get("/reports", params={"offset": -1}).status_code == 422


def test_report_filter_validation(client):
    assert client.get("/reports", params={"category": "Nope"}).status_code == 422
    assert client.get("/reports", params={"date_from": "2026-10-12", "date_to": "2026-10-01"}).status_code == 422
    assert client.get("/reports", params={"date_from": "not-a-date"}).status_code == 422


def test_events_endpoint_marks_coordinates_as_approximate(client):
    j = client.get("/events", params={"mapped_only": True, "include_expired": True}).json()
    assert j["total"] == 4 and "never identify" in j["note"]
    assert all(i["coord_precision"] == "event_centroid" and i["lat"] is not None for i in j["items"])
    unmapped = [i for i in client.get("/events", params={"include_expired": True}).json()["items"] if i["lat"] is None]
    assert unmapped and all(i["coord_precision"] is None for i in unmapped)  # Sachet areas without a single known place are NOT mapped


def test_report_detail_separates_observed_from_model_output(client):
    rid = client.get("/reports", params={"source": "ReliefWeb"}).json()["items"][0]["id"]
    d = client.get(f"/reports/{rid}").json()
    assert "observed_vs_inferred" in d and "content_hash" not in d and d["verification_status"] == "source_reported"
    assert client.get("/reports/MSG-NOPE").status_code == 404


def test_expired_alerts_are_flagged(client):
    items = {i["title"][:20]: i for i in client.get("/reports", params={"origin": "official_alert"}).json()["items"]}
    flags = sorted(i["is_expired"] for i in items.values())
    assert flags == [False, True, True]  # two alerts carry a past 'expires'; the Hindi one has no expiry, so it is not called expired


# ------------------------------------------------------------------ human review statuses
def test_verification_label_follows_review_not_source(seeded, client):
    pipe, _ = seeded
    gd = client.get("/reports", params={"source": "GDACS"}).json()["items"][0]
    assert gd["verification"] == "official_source"
    for status, expected in [("approved", "human_verified"), ("rejected", "incorrect"), ("duplicate", "duplicate"), ("needs_info", "investigating")]:
        assert client.post("/reviews", json={"message_id": gd["id"], "reviewer": "Asha", "status": status, "notes": "x"}).status_code == 201
        assert client.get(f"/reports/{gd['id']}").status_code == 200
        item = next(i for i in client.get("/reports", params={"source": "GDACS", "limit": 50}).json()["items"] if i["id"] == gd["id"])
        assert item["verification"] == expected
    row = pipe.db.reviewed_export()
    assert row.iloc[0]["reviewer"] == "Asha" and row.iloc[0]["reviewed_at"]  # who + when recorded


# ------------------------------------------------------------------ refresh + status endpoints
def test_refresh_endpoint_and_unknown_source(client):
    assert client.post("/ingestion/refresh", json={"sources": ["Nope"]}).status_code == 422
    r = client.post("/ingestion/refresh", json={"sources": ["GDACS"]}).json()
    assert r["results"][0]["source"] == "GDACS" and r["results"][0]["status"] in ("skipped", "success")  # interval protection


def test_sources_status_never_claims_untested_or_unconfigured_sources_are_connected(pipe, monkeypatch):
    monkeypatch.setattr(config, "RELIEFWEB_APPNAME", "")
    c = TestClient(create_app(pipe, IngestionService(pipe.db, pipe, FakeNet())))
    s = {x["name"]: x for x in c.get("/sources/status").json()["sources"]}
    assert not any(x["connected"] for x in s.values())
    assert s["ReliefWeb"]["health"] == "not_configured" and s["GDACS"]["health"] == "untested"


# ------------------------------------------------------------------ security
def test_production_without_key_disables_all_writes(seeded, monkeypatch):
    pipe, svc = seeded
    monkeypatch.setattr(config, "APP_ENV", "production")
    monkeypatch.setattr(config, "API_KEY", "")
    c = TestClient(create_app(pipe, svc))
    mid = pipe.db.messages_df().iloc[0]["id"]
    for call in (lambda: c.post("/reviews", json={"message_id": mid, "reviewer": "A", "status": "approved"}),
                 lambda: c.post("/ingestion/refresh", json={}),
                 lambda: c.post("/messages/batch", json={"messages": [{"text": "x"}]}),
                 lambda: c.patch(f"/messages/{mid}/location", json={"name": "X"}),
                 lambda: c.post("/messages/analyze", json={"text": "need water", "save": True}),
                 lambda: c.get("/ingestion/runs")):
        assert call().status_code == 503
    assert c.get("/reports").status_code == 200 and c.get("/health").json()["writes"].startswith("disabled")
    assert c.post("/messages/analyze", json={"text": "need water"}).status_code == 200  # read-only analysis stays available


def test_api_key_required_when_configured(seeded, monkeypatch):
    pipe, svc = seeded
    monkeypatch.setattr(config, "API_KEY", "s3cret-key")
    c = TestClient(create_app(pipe, svc))
    mid = pipe.db.messages_df().iloc[0]["id"]
    body = {"message_id": mid, "reviewer": "A", "status": "approved"}
    assert c.post("/reviews", json=body).status_code == 401
    assert c.post("/reviews", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.post("/reviews", json=body, headers={"X-API-Key": "s3cret-key"}).status_code == 201
    pub = c.get("/sources/status").json()
    assert pub["detail_level"] == "public" and all(s["last_error"] in (None, "hidden") for s in pub["sources"])
    assert c.get("/sources/status", headers={"X-API-Key": "s3cret-key"}).json()["detail_level"] == "full"
    assert "s3cret-key" not in c.get("/health").text and "acme-test" not in c.get("/sources/status").text


def test_dashboard_security_modes(monkeypatch):
    monkeypatch.setattr(config, "APP_ENV", "local"); monkeypatch.setattr(config, "ADMIN_PASSCODE", "")
    assert security.dashboard_mode() == "open_local"
    monkeypatch.setattr(config, "APP_ENV", "production")
    assert security.dashboard_mode() == "locked"
    monkeypatch.setattr(config, "ADMIN_PASSCODE", "pw")
    assert security.dashboard_mode() == "secret" and security.check_passcode("pw") and not security.check_passcode("nope") and not security.check_passcode("")


def test_realworld_analytics_endpoint(client):
    j = client.get("/analytics/real-world").json()
    assert j["by_origin"]["official_event"] == 4 and j["by_source"]["NDMA Sachet"] == 3 and "caveat" in j
    assert all("official" not in str(r) for r in j["needs_by_category"])  # official records never counted as needs
    assert sum(r["count"] for r in j["needs_by_category"]) >= 2


def test_helpers_pagination_and_labels():
    assert rp.freshness_text(None) == "never" and rp.freshness_text(30) == "30 min ago" and rp.freshness_text(180) == "3 h ago"
    assert rp.excerpt("a" * 500, 50).endswith("…") and len(rp.excerpt("a" * 500, 50)) == 50
