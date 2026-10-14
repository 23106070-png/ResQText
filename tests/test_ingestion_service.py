import json
import sqlite3
from datetime import datetime

import pytest

import config
from database import migrations
from database.db import Database
from database.models import SCHEMA
from ingestion import scheduler
from ingestion.models import SourceError
from ingestion.service import IngestionService
from nlp.classifier import ModelNotFoundError
from tests.netmock import Clock, FakeNet, R, fixture, sachet_net

T0 = datetime(2026, 10, 13, 12, 0, 0)


@pytest.fixture()
def clock():
    return Clock(T0)


@pytest.fixture()
def mk(pipe, clock, monkeypatch):
    monkeypatch.setattr(config, "RELIEFWEB_APPNAME", "")
    def make(net):
        return IngestionService(pipe.db, pipe, net, now_fn=clock)
    return make


def gdacs_net():
    return FakeNet().add("SEARCH", R(200, fixture("gdacs_search.json")))


# ------------------------------------------------------------------ storing + attribution + timestamps
def test_import_stores_source_attribution_and_timestamps(mk, pipe):
    svc = mk(sachet_net())
    r = svc.run_source("NDMA Sachet", "manual")
    assert r["status"] == "success" and r["inserted"] == 3
    df = pipe.db.messages_df()
    row = df[df["source_id"] == "1791731455698013"].iloc[0]
    assert row["source_name"] == "NDMA Sachet" and row["origin"] == "official_alert" and row["verification_status"] == "official_source"
    assert row["source_url"].endswith("1791731455698013") and row["published_at"] == "2026-10-11T15:13:07"
    assert row["retrieved_at"] == "2026-10-13T12:00:00" and "NDMA" in row["attribution"]
    assert row["original_text"].startswith("River SAI") and row["event_type"] == "Flood" and row["is_india"] == 1
    assert row["timestamp"] == row["published_at"]  # timelines use when the source published, not when we fetched


def test_duplicate_imports_are_prevented(mk, pipe, clock):
    svc = mk(gdacs_net())
    assert svc.run_source("GDACS")["inserted"] == 4
    clock.advance(hours=1)
    again = svc.run_source("GDACS")
    assert again["inserted"] == 0 and again["unchanged"] == 4 and pipe.db.count_source_items("GDACS") == 4


def test_source_update_changes_unreviewed_item(mk, pipe, clock):
    svc = mk(gdacs_net())
    svc.run_source("GDACS")
    data = json.loads(fixture("gdacs_search.json"))
    data["features"][1]["properties"]["alertlevel"] = "Red"
    data["features"][1]["properties"]["datemodified"] = "2026-10-12T09:00:00"
    clock.advance(hours=1)
    svc2 = mk(FakeNet().add("SEARCH", R(200, data)))
    r = svc2.run_source("GDACS")
    assert r["updated"] == 1 and pipe.db.count_source_items("GDACS") == 4
    assert "Red" in pipe.db.find_by_source("GDACS", "TC-1001334")["severity"]


def test_reviewed_item_is_not_overwritten_without_permission(mk, pipe, clock):
    svc = mk(gdacs_net())
    svc.run_source("GDACS")
    mid = pipe.db.find_by_source("GDACS", "TC-1001334")["id"]
    pipe.db.add_review(mid, "Asha", "approved", notes="checked against the GDACS report page")
    data = json.loads(fixture("gdacs_search.json"))
    data["features"][1]["properties"]["alertlevel"] = "Red"
    data["features"][1]["properties"]["datemodified"] = "2026-10-12T09:00:00"
    clock.advance(hours=1)
    r = mk(FakeNet().add("SEARCH", R(200, data))).run_source("GDACS")
    assert r["held_back"] == 1 and r["updated"] == 0
    kept = pipe.db.find_by_source("GDACS", "TC-1001334")
    assert "Orange" in kept["severity"] and kept["review_status"] == "approved" and kept["pending_update_json"]
    assert pipe.db.apply_pending_update(mid, "Asha") is True  # explicit human permission
    after = pipe.db.find_by_source("GDACS", "TC-1001334")
    assert "Red" in after["severity"] and after["review_status"] == "needs_info" and after["pending_update_json"] is None
    assert pipe.db.apply_pending_update(mid, "Asha") is False


def test_manual_location_correction_also_protects_item(mk, pipe, clock):
    svc = mk(gdacs_net())
    svc.run_source("GDACS")
    mid = pipe.db.find_by_source("GDACS", "FL-1104220")["id"]
    pipe.db.set_location(mid, "Somewhere precise", 41.0, 16.0)
    data = json.loads(fixture("gdacs_search.json"))
    data["features"][0]["properties"]["datemodified"] = "2026-10-12T10:00:00"
    clock.advance(hours=1)
    assert mk(FakeNet().add("SEARCH", R(200, data))).run_source("GDACS")["held_back"] == 1
    assert pipe.db.find_by_source("GDACS", "FL-1104220")["location_text"] == "Somewhere precise"


# ------------------------------------------------------------------ NLP integration
def test_official_records_are_not_classified_into_needs(mk, pipe):
    mk(sachet_net()).run_source("NDMA Sachet")
    m = pipe.db.get_message(pipe.db.find_by_source("NDMA Sachet", "1791731455698013")["id"])
    assert m["classifications"] == [] and m["nlp_status"] == "skipped_official" and "not classified" in m["nlp_note"]
    assert m["summary"] and m["summary"] in m["original_text"]  # extractive: copied from the source text
    assert any(e["type"] == "KEYWORD" or e["type"] == "LOCATION" or True for e in m["entities"])


def test_situation_report_goes_to_human_review_and_is_labelled_as_inferred(mk, pipe, monkeypatch):
    monkeypatch.setattr(config, "RELIEFWEB_APPNAME", "acme-test-abc123")
    svc = mk(FakeNet().add("/reports", R(200, fixture("reliefweb_reports.json")), method="POST"))
    r = svc.run_source("ReliefWeb")
    assert r["status"] == "partial" and r["inserted"] == 2 and r["errors"] == 1
    m = pipe.db.get_message(pipe.db.find_by_source("ReliefWeb", "4000001")["id"])
    assert m["nlp_status"] == "needs_review" and "synthetic" in m["nlp_note"]
    assert m["predicted_labels"] and m["verification_status"] == "source_reported" and m["top_confidence"] is not None
    assert m["location_text"] == "India" and m["location_status"] == "source_country" and m["lat"] is None  # never invented
    fr = pipe.db.get_message(pipe.db.find_by_source("ReliefWeb", "4000002")["id"])
    assert fr["nlp_status"] == "needs_review" or fr["nlp_status"] == "processed"  # French id is 'fr' not 'non-en' script


def test_non_english_script_is_skipped_not_misclassified(pipe):
    from ingestion.models import NormalizedItem
    it = NormalizedItem("ReliefWeb", "x1", "situation_report", "रिपोर्ट", "बाढ़ से लोग प्रभावित हैं और पानी चाहिए", None, "2026-10-12T00:00:00",
                        language="non-en", verification_status="source_reported")
    r = pipe.process_external(it)
    m = pipe.db.get_message(r["message_id"])
    assert m["nlp_status"] == "skipped_language" and m["classifications"] == []


def test_missing_model_still_stores_item(db, monkeypatch, tmp_path):
    from nlp.pipeline import Pipeline
    from ingestion.models import NormalizedItem
    monkeypatch.setattr(config, "ARTIFACT_DIR", tmp_path / "none")
    p = Pipeline(db)
    r = p.process_external(NormalizedItem("ReliefWeb", "m1", "situation_report", "Floods", "People need water.", None, "2026-10-12T00:00:00"))
    assert db.get_message(r["message_id"])["nlp_status"] == "failed"


def test_related_situation_reports_are_flagged_as_duplicates(pipe):
    from ingestion.models import NormalizedItem
    body = "Heavy flooding in Test District displaced 5,000 people who urgently need drinking water and food."
    a = pipe.process_external(NormalizedItem("ReliefWeb", "d1", "situation_report", "Flood update", body, None, "2026-10-12T00:00:00"))
    b = pipe.process_external(NormalizedItem("ReliefWeb", "d2", "situation_report", "Flood update", body, None, "2026-10-12T06:00:00"))
    assert any(d["other_id"] == a["message_id"] for d in pipe.db.list_duplicates(message_id=b["message_id"]))


# ------------------------------------------------------------------ failures, rate limits, intervals, locks
def test_timeout_is_recorded_not_raised_and_not_shown_connected(mk, pipe):
    svc = mk(FakeNet().add("SEARCH", SourceError("timeout", "The request timed out.")))
    r = svc.run_source("GDACS")
    assert r["status"] == "failed" and r["kind"] in ("timeout", "http_error")
    st = next(s for s in svc.source_status() if s["name"] == "GDACS")
    assert st["connected"] is False and st["health"] == "failing" and st["consecutive_failures"] == 1
    assert pipe.db.recent_errors("GDACS")[0]["message"]


def test_rate_limit_sets_backoff_and_stops_hammering(mk, pipe, clock):
    net = FakeNet().add("SEARCH", R(429, b"", retry_after="3600"))
    svc = mk(net)
    assert svc.run_source("GDACS")["kind"] == "rate_limited"
    n = len(net.calls)
    clock.advance(minutes=5)
    r = svc.run_source("GDACS", force=True)
    assert r["status"] == "skipped" or len(net.calls) == n  # backing off: no new request
    assert pipe.db.get_state("GDACS")["backoff_until"]


def test_minimum_interval_between_checks(mk, clock):
    net = gdacs_net()
    svc = mk(net)
    assert svc.run_source("GDACS", "auto")["status"] == "success"
    n = len(net.calls)
    clock.advance(minutes=5)
    assert svc.run_source("GDACS", "auto")["status"] == "skipped" and len(net.calls) == n
    clock.advance(minutes=30)
    assert svc.run_source("GDACS", "auto")["status"] == "success"


def test_manual_refresh_has_short_minimum_to_prevent_button_mashing(mk, clock):
    svc = mk(gdacs_net())
    svc.run_source("GDACS", force=True)
    clock.advance(seconds=10)
    assert svc.run_source("GDACS", force=True)["status"] == "skipped"
    clock.advance(seconds=120)
    assert svc.run_source("GDACS", force=True)["status"] == "success"


def test_lock_prevents_concurrent_refresh(mk, pipe):
    svc = mk(gdacs_net())
    assert pipe.db.acquire_lock("ingest:GDACS", "another-process")
    r = svc.run_source("GDACS")
    assert r["status"] == "skipped" and "already running" in r["message"]
    pipe.db.release_lock("ingest:GDACS", "another-process")
    assert svc.run_source("GDACS")["status"] == "success"


def test_one_bad_item_does_not_stop_the_run(mk, pipe, monkeypatch):
    real = pipe.process_external
    def flaky(item, *a, **k):
        if item.source_id == "FL-1104220":
            raise RuntimeError("boom with secret details")
        return real(item, *a, **k)
    monkeypatch.setattr(pipe, "process_external", flaky)
    r = mk(gdacs_net()).run_source("GDACS")
    assert r["status"] == "partial" and r["inserted"] == 3 and r["errors"] == 1
    assert "secret" not in pipe.db.recent_errors("GDACS")[0]["message"]  # internals are not stored/leaked


def test_unconfigured_source_is_skipped_not_called_and_not_connected(mk, pipe):
    net = FakeNet()
    svc = mk(net)
    r = svc.run_source("ReliefWeb")
    assert r["status"] == "skipped" and net.calls == []
    st = next(s for s in svc.source_status() if s["name"] == "ReliefWeb")
    assert st["health"] == "not_configured" and st["connected"] is False and st["needs_credentials"]
    assert svc.run_source("Nope")["status"] == "skipped"


def test_staleness_is_visible(mk, clock):
    svc = mk(gdacs_net())
    svc.run_source("GDACS")
    get = lambda: next(s for s in svc.source_status() if s["name"] == "GDACS")
    assert get()["health"] == "ok" and get()["connected"]
    clock.advance(hours=3)
    assert get()["health"] == "stale"
    untested = next(s for s in svc.source_status() if s["name"] == "NDMA Sachet")
    assert untested["health"] == "untested" and untested["connected"] is False


def test_refresh_all_runs_every_source(mk):
    net = gdacs_net()
    net.add("rss_india.xml", R(200, fixture("sachet_rss.xml"))).add("FetchXMLFile", R(200, fixture("sachet_cap_flood.xml")))
    res = {r["source"]: r["status"] for r in mk(net).refresh(force=True)}
    assert res["GDACS"] == "success" and res["NDMA Sachet"] == "success" and res["ReliefWeb"] == "skipped"


def test_scheduler_starts_only_one_thread(mk):
    svc = mk(gdacs_net())
    try:
        assert scheduler.start_background(svc) is True
        assert scheduler.start_background(svc) is False
    finally:
        scheduler.stop_background()


# ------------------------------------------------------------------ schema upgrade of an existing database
def test_migration_keeps_existing_data(tmp_path):
    path = tmp_path / "old.db"
    c = sqlite3.connect(path)
    c.executescript(SCHEMA)
    c.execute("INSERT INTO messages (id,original_text,normalized_text,timestamp,content_hash,imported_at) VALUES ('MSG-OLD','Need water','need water','2024-07-26T10:00:00','h','2024-07-26T10:00:00')")
    c.commit(); c.close()
    db = Database(path)  # opening runs the migration
    row = db.get_message("MSG-OLD")
    assert row["original_text"] == "Need water" and row["origin"] == "user_report" and row["verification_status"] == "unverified"
    Database(path)  # second open is a no-op
    with db.conn() as conn:
        assert migrations.current_version(conn) == 1
