"""Headless runs of the new/changed pages with imported data present, plus the sign-in gate."""
from datetime import datetime

import pytest
from streamlit.testing.v1 import AppTest

import config
from database.db import Database
from ingestion.service import IngestionService
from nlp.pipeline import Pipeline
from tests.netmock import Clock, R, fixture, sachet_net

PAGES = ["overview", "real_world", "event_map", "emerging", "nlp_analysis", "review", "sources_health", "settings", "incoming", "map_view", "duplicates", "evaluation"]


@pytest.fixture(scope="module")
def imported(trained):
    monkey = pytest.MonkeyPatch()
    monkey.setattr(config, "RELIEFWEB_APPNAME", "acme-test-abc123")
    p = Pipeline(Database())  # same DB the AppTests use
    net = sachet_net()
    net.add("SEARCH", R(200, fixture("gdacs_search.json"))).add("/reports", R(200, fixture("reliefweb_reports.json")), method="POST")
    IngestionService(p.db, p, net, now_fn=Clock(datetime(2026, 10, 13, 12))).refresh(force=True)
    p.ingest("Five people need drinking water in Kalwa", "2026-10-12T10:00:00", is_simulated=True)
    yield p
    monkey.undo()


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_with_imported_data(page, imported):
    at = AppTest.from_string(f"from dashboard.pages import {page}\n{page}.render()", default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]


def test_full_app_navigation_renders(imported):
    at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app.py"), default_timeout=90).run()
    assert not at.exception


def test_real_world_page_shows_attribution_and_safe_links(imported):
    at = AppTest.from_string("from dashboard.pages import real_world\nreal_world.render()", default_timeout=90).run()
    html = " ".join(m.value for m in at.markdown)
    assert "sachet.ndma.gov.in" in html and "rel='noopener noreferrer'" in html and "UTC" in html
    assert "NLP-inferred" in html and "Official source" in html


def test_javascript_links_are_never_rendered():
    from dashboard import ui
    assert ui.safe_url("javascript:alert(1)") is None and ui.safe_url("https://reliefweb.int/x") and ui.safe_url(None) is None
    assert "href" not in ui.link("javascript:alert(1)", "x")


def test_sources_page_does_not_claim_connection_for_untested_source(imported):
    at = AppTest.from_string("from dashboard.pages import sources_health\nsources_health.render()", default_timeout=90).run()
    text = " ".join(m.value for m in at.markdown) + " ".join(i.value for i in at.info) + " ".join(w.value for w in at.warning)
    assert "not a real-time feed" in text.lower() or "not</b> a real-time feed" in text.lower()


def test_review_is_locked_without_sign_in_in_production(imported, monkeypatch):
    monkeypatch.setattr(config, "APP_ENV", "production"); monkeypatch.setattr(config, "ADMIN_PASSCODE", "")
    at = AppTest.from_string("from dashboard.pages import review\nreview.render()", default_timeout=90).run()
    assert not at.exception and any("signed-in reviewer" in m.value for m in at.markdown)
    assert all(b.disabled for b in at.button if b.label in ("Verify", "Incorrect", "Duplicate", "Needs investigation"))


def test_review_open_in_local_mode(imported, monkeypatch):
    monkeypatch.setattr(config, "APP_ENV", "local"); monkeypatch.setattr(config, "ADMIN_PASSCODE", "")
    at = AppTest.from_string("from dashboard.pages import review\nreview.render()", default_timeout=90).run()
    assert not at.exception and not any("signed-in reviewer" in m.value for m in at.markdown)
