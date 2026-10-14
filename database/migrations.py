"""Safe, idempotent schema upgrades. Existing databases keep all their data.

Each migration is a function(conn). They only ADD columns/tables/indexes, never drop or rewrite data, and they
check PRAGMA table_info first so running them twice (or on a fresh database) is harmless.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime

NEW_MESSAGE_COLUMNS = [  # (name, SQL type + default)
    ("origin", "TEXT NOT NULL DEFAULT 'user_report'"),       # user_report | situation_report | official_event | official_alert
    ("source_name", "TEXT"),
    ("source_id", "TEXT"),
    ("source_url", "TEXT"),
    ("title", "TEXT"),
    ("published_at", "TEXT"),                                  # when the SOURCE published it (UTC)
    ("retrieved_at", "TEXT"),                                  # when WE fetched it (UTC)
    ("source_updated_at", "TEXT"),
    ("expires_at", "TEXT"),
    ("event_type", "TEXT"),
    ("severity", "TEXT"),
    ("country", "TEXT"),
    ("language", "TEXT"),
    ("coord_precision", "TEXT"),                               # NULL | event_centroid | area_centroid | precise
    ("verification_status", "TEXT NOT NULL DEFAULT 'unverified'"),  # official_source | source_reported | unverified
    ("nlp_status", "TEXT NOT NULL DEFAULT 'processed'"),       # processed | needs_review | skipped_official | skipped_language | failed
    ("nlp_note", "TEXT"),
    ("is_india", "INTEGER NOT NULL DEFAULT 0"),
    ("attribution", "TEXT"),
    ("summary", "TEXT"),
    ("extra_json", "TEXT"),
    ("pending_update_json", "TEXT"),                           # newer source version waiting for a human to accept
]

NEW_TABLES = """
CREATE TABLE IF NOT EXISTS source_state (
    source               TEXT PRIMARY KEY,
    last_attempt_at      TEXT,
    last_success_at      TEXT,
    last_status          TEXT,
    last_error           TEXT,
    last_http_status     INTEGER,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    backoff_until        TEXT,
    total_items          INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,
    trigger     TEXT NOT NULL,                 -- manual | auto | cli
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL,                 -- running | success | partial | failed | skipped
    fetched     INTEGER NOT NULL DEFAULT 0,
    inserted    INTEGER NOT NULL DEFAULT 0,
    updated     INTEGER NOT NULL DEFAULT 0,
    unchanged   INTEGER NOT NULL DEFAULT 0,
    held_back   INTEGER NOT NULL DEFAULT 0,    -- reviewed items not overwritten
    errors      INTEGER NOT NULL DEFAULT 0,
    api_calls   INTEGER NOT NULL DEFAULT 0,
    message     TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_source ON ingestion_runs(source, id);
CREATE TABLE IF NOT EXISTS ingestion_errors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER,
    source      TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    kind        TEXT NOT NULL,                 -- timeout | network | http_error | rate_limited | auth | invalid_response | parse | item
    message     TEXT NOT NULL,
    item_ref    TEXT
);
CREATE INDEX IF NOT EXISTS idx_errors_source ON ingestion_errors(source, id);
CREATE TABLE IF NOT EXISTS job_locks (
    name       TEXT PRIMARY KEY,
    holder     TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS http_cache (
    url           TEXT PRIMARY KEY,
    etag          TEXT,
    last_modified TEXT,
    fetched_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS api_usage (
    source TEXT NOT NULL,
    day    TEXT NOT NULL,
    calls  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (source, day)
);
"""


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def m001_realworld(conn: sqlite3.Connection) -> None:
    have = _columns(conn, "messages")
    for name, decl in NEW_MESSAGE_COLUMNS:
        if name not in have:
            conn.execute(f"ALTER TABLE messages ADD COLUMN {name} {decl}")  # names are fixed literals above
    conn.executescript(NEW_TABLES)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_messages_source ON messages(source_name, source_id) "
                 "WHERE source_id IS NOT NULL")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_origin ON messages(origin, published_at)")


MIGRATIONS = [(1, "real-world ingestion: provenance columns, source state, runs, errors, locks", m001_realworld)]


def current_version(conn: sqlite3.Connection) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, name TEXT, applied_at TEXT)")
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return row[0] or 0


def migrate(conn: sqlite3.Connection) -> list[int]:
    """Apply pending migrations. Returns the versions applied (empty when already up to date)."""
    applied, cur = [], current_version(conn)
    for version, name, fn in MIGRATIONS:
        if version > cur:
            fn(conn)
            conn.execute("INSERT INTO schema_version (version, name, applied_at) VALUES (?,?,?)",
                         (version, name, datetime.now().isoformat(timespec="seconds")))
            applied.append(version)
    return applied
