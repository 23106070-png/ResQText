"""SQLite schema. Original text and normalised text are stored separately; original is never rewritten."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id               TEXT PRIMARY KEY,
    original_text    TEXT NOT NULL,
    normalized_text  TEXT NOT NULL,
    timestamp        TEXT NOT NULL,
    timestamp_source TEXT NOT NULL DEFAULT 'provided',   -- provided | import_time
    source           TEXT NOT NULL DEFAULT 'manual',
    is_simulated     INTEGER NOT NULL DEFAULT 0,
    content_hash     TEXT NOT NULL UNIQUE,                -- blocks re-importing the exact same report
    imported_at      TEXT NOT NULL,
    location_text    TEXT NOT NULL DEFAULT 'Unknown',
    location_status  TEXT NOT NULL DEFAULT 'unknown',
    lat              REAL,
    lon              REAL,
    location_source  TEXT,
    people_count     TEXT NOT NULL DEFAULT 'Unknown',
    top_confidence   REAL,
    low_confidence   INTEGER NOT NULL DEFAULT 0,
    model_name       TEXT,
    review_status    TEXT NOT NULL DEFAULT 'pending'      -- pending | approved(=verified by reviewer) | corrected | rejected(=incorrect) | needs_info | duplicate
);
CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(timestamp);

CREATE TABLE IF NOT EXISTS entities (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    type       TEXT NOT NULL,
    value      TEXT NOT NULL,
    evidence   TEXT,
    start_char INTEGER,
    end_char   INTEGER,
    source     TEXT
);
CREATE INDEX IF NOT EXISTS idx_entities_msg ON entities(message_id);

CREATE TABLE IF NOT EXISTS classifications (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    label      TEXT NOT NULL,
    score      REAL NOT NULL,
    assigned   INTEGER NOT NULL,                          -- 1 when the label was predicted
    model_name TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_class_msg ON classifications(message_id);

CREATE TABLE IF NOT EXISTS duplicate_suggestions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id        TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    other_id          TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    similarity        REAL NOT NULL,
    adjusted_score    REAL NOT NULL,
    location_relation TEXT,
    time_gap_hours    REAL,
    reason            TEXT,
    backend           TEXT,
    status            TEXT NOT NULL DEFAULT 'pending',     -- pending | confirmed | rejected
    decided_by        TEXT,
    decided_at        TEXT,
    created_at        TEXT NOT NULL,
    UNIQUE(message_id, other_id)
);

CREATE TABLE IF NOT EXISTS reviews (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id          TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    reviewer            TEXT NOT NULL,
    status              TEXT NOT NULL,
    corrected_labels    TEXT,                              -- JSON list, NULL when labels were accepted as-is
    corrected_location  TEXT,
    corrected_lat       REAL,
    corrected_lon       REAL,
    notes               TEXT,
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reviews_msg ON reviews(message_id);

CREATE TABLE IF NOT EXISTS model_evaluations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT NOT NULL,                            -- classifier | duplicates
    dataset      TEXT,
    n_train      INTEGER,
    n_test       INTEGER,
    is_synthetic INTEGER NOT NULL DEFAULT 1,
    metrics_json TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
"""
REVIEW_STATUSES = ("pending", "approved", "corrected", "rejected", "needs_info", "duplicate")
