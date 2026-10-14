"""Run every dashboard page headlessly and fail on any Streamlit exception."""
import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["overview", "incoming", "nlp_analysis", "duplicates", "map_view", "emerging", "evaluation", "review", "settings"]


@pytest.fixture(scope="module")
def seeded(trained):
    from dashboard import ui
    from nlp.pipeline import Pipeline  # noqa: F401
    import pandas as pd, config

    p = ui.get_pipeline.__wrapped__() if hasattr(ui.get_pipeline, "__wrapped__") else None
    from database.db import Database
    from nlp.pipeline import Pipeline
    p = Pipeline(Database())  # same DB path the AppTests use (env var set in conftest)
    df = pd.read_csv(config.SAMPLE_CSV)
    df = df[df["split"] == "test"].head(25)
    for r in df.itertuples():
        p.ingest(r.text, r.timestamp, "test", True, r.message_id)
    p.ingest("Trapped in Zzyzxville, need a boat", "2024-07-26T12:00:00", is_simulated=True)
    return p


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page, seeded):
    at = AppTest.from_string(f"from dashboard.pages import {page}\n{page}.render()", default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]


def test_nlp_analysis_button(seeded):
    at = AppTest.from_string("from dashboard.pages import nlp_analysis\nnlp_analysis.render()", default_timeout=60).run()
    at.text_area[0].set_value("Five people need drinking water in Kalwa").run()
    at.button[0].click().run()
    assert not at.exception and any("Kalwa" in m.value for m in at.markdown)


def test_empty_message_shows_error(seeded):
    at = AppTest.from_string("from dashboard.pages import nlp_analysis\nnlp_analysis.render()", default_timeout=60).run()
    at.text_area[0].set_value("   ").run()
    at.button[0].click().run()
    assert not at.exception and at.error
