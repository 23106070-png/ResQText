import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="resqtext_test_"))
os.environ["RESQTEXT_DB_PATH"] = str(_TMP / "app.db")  # must be set BEFORE config is imported
os.environ["RESQTEXT_ARTIFACT_DIR"] = str(_TMP / "artifacts")

import pytest  # noqa: E402

import config  # noqa: E402
from database.db import Database  # noqa: E402
from nlp.pipeline import Pipeline  # noqa: E402


@pytest.fixture(scope="session")
def trained():
    from training.train_classifier import train

    return train(artifact_dir=config.ARTIFACT_DIR, verbose=False)


@pytest.fixture()
def db(tmp_path):
    return Database(tmp_path / "t.db")


@pytest.fixture()
def pipe(db, trained):
    return Pipeline(db)
