"""Central configuration. Every value can be overridden with an environment variable (see .env.example)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

try:  # optional: load a local .env file if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:  # pragma: no cover
    pass


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(int(default))).strip().lower() in {"1", "true", "yes", "on"}


LABELS = [
    "Rescue assistance",
    "Medical assistance",
    "Food and drinking water",
    "Shelter and evacuation",
    "Infrastructure damage",
    "General information",
]

DB_PATH = Path(os.getenv("RESQTEXT_DB_PATH", ROOT / "database" / "resqtext.db"))
ARTIFACT_DIR = Path(os.getenv("RESQTEXT_ARTIFACT_DIR", ROOT / "artifacts"))
DATA_DIR = ROOT / "data"
SAMPLE_CSV = DATA_DIR / "sample_messages.csv"
PAIRS_CSV = DATA_DIR / "duplicate_pairs.csv"
GAZETTEER_CSV = DATA_DIR / "gazetteer.csv"

# Classification: a label is assigned when its probability >= LABEL_THRESHOLD.
# If nothing passes, the single most likely label is used so every message gets a label.
LABEL_THRESHOLD = _float("RESQTEXT_LABEL_THRESHOLD", 0.40)
LOW_CONFIDENCE = _float("RESQTEXT_LOW_CONFIDENCE", 0.50)

# Duplicate detection
SIMILARITY_THRESHOLD = _float("RESQTEXT_SIMILARITY_THRESHOLD", 0.70)
DIFFERENT_LOCATION_PENALTY = _float("RESQTEXT_DIFF_LOCATION_PENALTY", 0.25)
TIME_WINDOW_HOURS = _float("RESQTEXT_TIME_WINDOW_HOURS", 72)
USE_SENTENCE_TRANSFORMERS = _flag("RESQTEXT_USE_SENTENCE_TRANSFORMERS", False)
SENTENCE_MODEL = os.getenv("RESQTEXT_SENTENCE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Optional integrations (all OFF by default so the core app runs fully offline)
ENABLE_GEOCODING = _flag("RESQTEXT_ENABLE_GEOCODING", False)
GEOCODER_USER_AGENT = os.getenv("RESQTEXT_GEOCODER_UA", "ResQText-hackathon-demo (set-your-contact-email)")
USE_TRANSFORMER = _flag("RESQTEXT_USE_TRANSFORMER", False)
SPACY_MODEL = os.getenv("RESQTEXT_SPACY_MODEL", "en_core_web_sm")

API_URL = os.getenv("RESQTEXT_API_URL", "http://127.0.0.1:8000")
DISCLAIMER = (
    "Decision-support prototype. Not connected to emergency services and not a replacement for "
    "official emergency response systems. Predictions can be wrong; a human must verify every report."
)
CONFIDENCE_NOTE = (
    "Confidence is the model's score for a label. It is not the probability that the report is true "
    "or that the situation is urgent."
)


# ======================================================================================================
# Real-world data ingestion (all optional; the app works fully offline without any of this)
# ======================================================================================================
APP_ENV = os.getenv("RESQTEXT_ENV", "local").strip().lower()          # local | production
ADMIN_PASSCODE = os.getenv("RESQTEXT_ADMIN_PASSCODE", "")              # protects review / refresh in the dashboard
API_KEY = os.getenv("RESQTEXT_API_KEY", "")                            # protects write endpoints of the API

HTTP_TIMEOUT = _float("RESQTEXT_HTTP_TIMEOUT", 20)
USER_AGENT = os.getenv("RESQTEXT_USER_AGENT", "ResQText-prototype/2.0 (disaster decision-support research)")
STALE_FACTOR = _float("RESQTEXT_STALE_FACTOR", 3)                       # stale after 3 missed intervals
MIN_MANUAL_REFRESH_SECONDS = _float("RESQTEXT_MIN_MANUAL_REFRESH_SECONDS", 60)

# GDACS (free, no key). Attribution required: "Global Disaster Awareness and Coordination System, GDACS".
GDACS_ENABLED = _flag("GDACS_ENABLED", True)
GDACS_BASE_URL = os.getenv("GDACS_BASE_URL", "https://www.gdacs.org/gdacsapi/api")
GDACS_LOOKBACK_DAYS = int(_float("GDACS_LOOKBACK_DAYS", 14))
GDACS_MAX_PAGES = int(_float("GDACS_MAX_PAGES", 3))
GDACS_INTERVAL_MIN = _float("GDACS_INTERVAL_MIN", 30)

# NDMA Sachet public CAP feeds (no key). Consumers must use ETag caching (official integration guide).
SACHET_ENABLED = _flag("SACHET_ENABLED", True)
SACHET_FEEDS = [u.strip() for u in os.getenv(
    "SACHET_FEEDS", "https://sachet.ndma.gov.in/cap_public_website/rss/rss_india.xml").split(",") if u.strip()]
SACHET_MAX_DETAIL_PER_RUN = int(_float("SACHET_MAX_DETAIL_PER_RUN", 25))
SACHET_INTERVAL_MIN = _float("SACHET_INTERVAL_MIN", 10)

# ReliefWeb API v2 (REQUIRES a pre-approved appname, quota 1000 calls/day).
RELIEFWEB_APPNAME = os.getenv("RELIEFWEB_APPNAME", "").strip()
RELIEFWEB_ENABLED = _flag("RELIEFWEB_ENABLED", True)
RELIEFWEB_BASE_URL = os.getenv("RELIEFWEB_BASE_URL", "https://api.reliefweb.int/v2")
RELIEFWEB_COUNTRY = os.getenv("RELIEFWEB_COUNTRY", "India")             # empty = worldwide
RELIEFWEB_LOOKBACK_DAYS = int(_float("RELIEFWEB_LOOKBACK_DAYS", 30))
RELIEFWEB_PAGE_SIZE = min(int(_float("RELIEFWEB_PAGE_SIZE", 50)), 1000)
RELIEFWEB_MAX_PAGES = int(_float("RELIEFWEB_MAX_PAGES", 3))
RELIEFWEB_DAILY_CALL_BUDGET = int(_float("RELIEFWEB_DAILY_CALL_BUDGET", 800))  # below the documented 1000/day
RELIEFWEB_INTERVAL_MIN = _float("RELIEFWEB_INTERVAL_MIN", 60)
RELIEFWEB_MAX_STORED_CHARS = int(_float("RELIEFWEB_MAX_STORED_CHARS", 3000))   # respect partners' copyright: store a short excerpt

# Automatic polling (off by default). Prefer the separate process: python -m ingestion.scheduler
AUTOPOLL_IN_API = _flag("RESQTEXT_AUTOPOLL", False)
AUTOPOLL_IN_DASHBOARD = _flag("RESQTEXT_AUTOPOLL_IN_DASHBOARD", False)
SCHEDULER_TICK_SECONDS = _float("RESQTEXT_SCHEDULER_TICK_SECONDS", 60)

# The classifier was trained only on synthetic first-person messages. Until you validate it on labelled real
# reports and set this to 1, every real report is routed to human review.
NLP_VALIDATED_ON_REAL = _flag("RESQTEXT_NLP_VALIDATED_ON_REAL", False)
NEED_ORIGINS = ("user_report", "situation_report")                      # origins that can express needs
