"""Database methods for imported (real-world) items, ingestion bookkeeping, locks and caches."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd

from nlp.preprocessing import normalize_text

SOURCE_OWNED = ("title", "original_text", "normalized_text", "source_url", "published_at", "source_updated_at",
                "expires_at", "event_type", "severity", "country", "language", "attribution", "extra_json")


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


class IngestStore:
    # ------------------------------------------------------------------ imported items
    def find_by_source(self, source_name: str, source_id: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM messages WHERE source_name=? AND source_id=?", (source_name, source_id)).fetchone()
            return dict(r) if r else None

    def source_item_exists(self, source_name: str, source_id: str) -> bool:
        with self.conn() as c:
            return c.execute("SELECT 1 FROM messages WHERE source_name=? AND source_id=?", (source_name, source_id)).fetchone() is not None

    def upsert_external(self, item, text: str, retrieved_at: str, location: dict) -> tuple[str, str]:
        """Insert or update one imported item. Returns (status, message_id) where status is
        inserted | updated | unchanged | held_back.

        Human work is protected: if a reviewer has already reviewed the item (review_status != pending) or corrected
        its location, a changed source version is NOT applied. It is parked in pending_update_json so a person can
        accept it explicitly (apply_pending_update)."""
        fp = item.fingerprint()
        extra = dict(item.extra, fingerprint=fp)
        new_fields = {
            "title": item.title, "original_text": text, "normalized_text": normalize_text(text),
            "source_url": item.url, "published_at": item.published_at, "source_updated_at": item.source_updated_at,
            "expires_at": item.expires_at, "event_type": item.event_type, "severity": item.severity,
            "country": item.country, "language": item.language, "attribution": item.attribution,
            "extra_json": json.dumps(extra, ensure_ascii=False, default=str),
        }
        with self.conn() as c:
            row = c.execute("SELECT * FROM messages WHERE source_name=? AND source_id=?", (item.source_name, item.source_id)).fetchone()
            if row is None:
                mid = "MSG-" + uuid.uuid4().hex[:8].upper()
                h = hashlib.sha256(item.dedupe_key.encode("utf-8")).hexdigest()
                ts = item.published_at or retrieved_at
                c.execute(
                    "INSERT INTO messages (id, original_text, normalized_text, timestamp, timestamp_source, source, is_simulated,"
                    " content_hash, imported_at, location_text, location_status, lat, lon, location_source, origin, source_name,"
                    " source_id, source_url, title, published_at, retrieved_at, source_updated_at, expires_at, event_type, severity,"
                    " country, language, coord_precision, verification_status, is_india, attribution, extra_json)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (mid, text, new_fields["normalized_text"], ts, "provided" if item.published_at else "import_time",
                     item.source_name, 0, h, retrieved_at, location["name"], location["status"], location["lat"], location["lon"],
                     location.get("source"), item.origin, item.source_name, item.source_id, item.url, item.title, item.published_at,
                     retrieved_at, item.source_updated_at, item.expires_at, item.event_type, item.severity, item.country,
                     item.language, item.coord_precision, item.verification_status, int(item.is_india), item.attribution,
                     new_fields["extra_json"]))
                return "inserted", mid
            old_fp = json.loads(row["extra_json"] or "{}").get("fingerprint")
            if old_fp == fp:
                c.execute("UPDATE messages SET retrieved_at=? WHERE id=?", (retrieved_at, row["id"]))
                return "unchanged", row["id"]
            protected = row["review_status"] != "pending" or row["location_source"] == "reviewer"
            if protected:
                pending = dict(new_fields, location=location, lat=item.lat, lon=item.lon, retrieved_at=retrieved_at)
                c.execute("UPDATE messages SET pending_update_json=?, retrieved_at=? WHERE id=?",
                          (json.dumps(pending, ensure_ascii=False, default=str), retrieved_at, row["id"]))
                return "held_back", row["id"]
            sets = ", ".join(f"{k}=?" for k in new_fields)
            c.execute(f"UPDATE messages SET {sets}, retrieved_at=?, timestamp=COALESCE(?, timestamp), location_text=?,"  # noqa: S608 (fixed keys)
                      " location_status=?, lat=?, lon=?, coord_precision=?, pending_update_json=NULL WHERE id=?",
                      (*new_fields.values(), retrieved_at, item.published_at, location["name"], location["status"],
                       location["lat"], location["lon"], item.coord_precision, row["id"]))
            return "updated", row["id"]

    def apply_pending_update(self, message_id: str, reviewer: str) -> bool:
        """Explicit human permission to replace a reviewed item's source content with the newer source version."""
        with self.conn() as c:
            row = c.execute("SELECT pending_update_json FROM messages WHERE id=?", (message_id,)).fetchone()
            if not row or not row["pending_update_json"]:
                return False
            p = json.loads(row["pending_update_json"])
            loc = p.pop("location")
            lat, lon = p.pop("lat"), p.pop("lon")
            retrieved = p.pop("retrieved_at")
            sets = ", ".join(f"{k}=?" for k in p)
            c.execute(f"UPDATE messages SET {sets}, retrieved_at=?, location_text=?, location_status=?, lat=?, lon=?,"  # noqa: S608
                      " pending_update_json=NULL WHERE id=?", (*p.values(), retrieved, loc["name"], loc["status"], lat, lon, message_id))
            c.execute("INSERT INTO reviews (message_id, reviewer, status, notes, created_at) VALUES (?,?,?,?,?)",
                      (message_id, reviewer, "needs_info", "Newer source version applied by reviewer; please re-check.", _iso(_now())))
            c.execute("UPDATE messages SET review_status='needs_info' WHERE id=?", (message_id,))
            return True

    def save_entities_only(self, message_id: str, entities: list[dict]) -> None:
        """Store extracted entities without touching classifications (used for official records)."""
        with self.conn() as c:
            c.execute("DELETE FROM entities WHERE message_id = ?", (message_id,))
            c.executemany(
                "INSERT INTO entities (message_id, type, value, evidence, start_char, end_char, source) VALUES (?,?,?,?,?,?,?)",
                [(message_id, e["type"], e["value"], e["evidence"], e["start"], e["end"], e["source"]) for e in entities])
            c.execute("DELETE FROM classifications WHERE message_id = ?", (message_id,))

    def set_nlp_result(self, message_id: str, nlp_status: str, note: str | None, summary: str | None) -> None:
        with self.conn() as c:
            c.execute("UPDATE messages SET nlp_status=?, nlp_note=?, summary=? WHERE id=?", (nlp_status, note, summary, message_id))

    # ------------------------------------------------------------------ source state, runs, errors
    def get_state(self, source: str) -> dict:
        with self.conn() as c:
            r = c.execute("SELECT * FROM source_state WHERE source=?", (source,)).fetchone()
            return dict(r) if r else {"source": source, "last_attempt_at": None, "last_success_at": None, "last_status": None,
                                      "last_error": None, "last_http_status": None, "consecutive_failures": 0,
                                      "backoff_until": None, "total_items": 0}

    def update_state(self, source: str, **fields) -> None:
        with self.conn() as c:
            c.execute("INSERT OR IGNORE INTO source_state (source) VALUES (?)", (source,))
            if fields:
                sets = ", ".join(f"{k}=?" for k in fields)  # keys come from our own code, never from users
                c.execute(f"UPDATE source_state SET {sets} WHERE source=?", (*fields.values(), source))  # noqa: S608

    def start_run(self, source: str, trigger: str) -> int:
        with self.conn() as c:
            cur = c.execute("INSERT INTO ingestion_runs (source, trigger, started_at, status) VALUES (?,?,?, 'running')",
                            (source, trigger, _iso(_now())))
            return cur.lastrowid

    def finish_run(self, run_id: int, status: str, **counts) -> None:
        keys = [k for k in ("fetched", "inserted", "updated", "unchanged", "held_back", "errors", "api_calls", "message") if k in counts]
        sets = ", ".join(f"{k}=?" for k in keys)
        with self.conn() as c:
            c.execute(f"UPDATE ingestion_runs SET status=?, finished_at=?{', ' + sets if sets else ''} WHERE id=?",  # noqa: S608
                      (status, _iso(_now()), *[counts[k] for k in keys], run_id))

    def log_error(self, source: str, kind: str, message: str, run_id: int | None = None, item_ref: str | None = None) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO ingestion_errors (run_id, source, occurred_at, kind, message, item_ref) VALUES (?,?,?,?,?,?)",
                      (run_id, source, _iso(_now()), kind, message[:500], item_ref))

    def recent_runs(self, source: str | None = None, limit: int = 20) -> list[dict]:
        q, args = "SELECT * FROM ingestion_runs", []
        if source:
            q += " WHERE source=?"
            args.append(source)
        with self.conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY id DESC LIMIT ?", (*args, limit))]

    def recent_errors(self, source: str | None = None, limit: int = 20) -> list[dict]:
        q, args = "SELECT * FROM ingestion_errors", []
        if source:
            q += " WHERE source=?"
            args.append(source)
        with self.conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY id DESC LIMIT ?", (*args, limit))]

    def count_source_items(self, source: str) -> int:
        with self.conn() as c:
            return c.execute("SELECT COUNT(*) FROM messages WHERE source_name=?", (source,)).fetchone()[0]

    # ------------------------------------------------------------------ lease lock (prevents duplicate jobs)
    def acquire_lock(self, name: str, holder: str, ttl_seconds: int = 300) -> bool:
        """Atomic across threads AND processes: BEGIN IMMEDIATE takes SQLite's write lock before we read."""
        now = _now()
        with self.conn() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT holder, expires_at FROM job_locks WHERE name=?", (name,)).fetchone()
            if row and row["expires_at"] > _iso(now) and row["holder"] != holder:
                return False
            c.execute("INSERT OR REPLACE INTO job_locks (name, holder, expires_at) VALUES (?,?,?)",
                      (name, holder, _iso(now + timedelta(seconds=ttl_seconds))))
            return True

    def release_lock(self, name: str, holder: str) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM job_locks WHERE name=? AND holder=?", (name, holder))

    # ------------------------------------------------------------------ HTTP cache + API usage
    def get_http_cache(self, url: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM http_cache WHERE url=?", (url,)).fetchone()
            return dict(r) if r else None

    def set_http_cache(self, url: str, etag: str | None, last_modified: str | None) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO http_cache (url, etag, last_modified, fetched_at) VALUES (?,?,?,?)",
                      (url, etag, last_modified, _iso(_now())))

    def add_api_calls(self, source: str, n: int = 1) -> int:
        day = _now().strftime("%Y-%m-%d")
        with self.conn() as c:
            c.execute("INSERT OR IGNORE INTO api_usage (source, day, calls) VALUES (?,?,0)", (source, day))
            c.execute("UPDATE api_usage SET calls = calls + ? WHERE source=? AND day=?", (n, source, day))
            return c.execute("SELECT calls FROM api_usage WHERE source=? AND day=?", (source, day)).fetchone()[0]

    def api_calls_today(self, source: str) -> int:
        day = _now().strftime("%Y-%m-%d")
        with self.conn() as c:
            r = c.execute("SELECT calls FROM api_usage WHERE source=? AND day=?", (source, day)).fetchone()
            return r[0] if r else 0

    def reports_df(self) -> pd.DataFrame:
        return self.messages_df()
