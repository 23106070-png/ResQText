"""SQLite access layer. All queries are parameterised."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from database.ingest_store import IngestStore
from database.migrations import migrate
from database.models import REVIEW_STATUSES, SCHEMA
from nlp.preprocessing import normalize_text


TEXT_COLUMNS = ("title", "source_name", "source_id", "source_url", "published_at", "retrieved_at", "source_updated_at",
                "expires_at", "event_type", "severity", "country", "language", "coord_precision", "nlp_note", "attribution",
                "summary", "extra_json", "pending_update_json", "location_source")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Database(IngestStore):
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or config.DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)
            migrate(c)  # safe, idempotent upgrade for databases created by earlier versions

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    # ------------------------------------------------------------------ messages
    @staticmethod
    def content_hash(text: str, timestamp: str, source: str, external_id: str | None) -> str:
        key = "|".join([normalize_text(text, fix_typos=False), timestamp, source, external_id or ""])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def add_message(self, text: str, timestamp: str | None = None, source: str = "manual",
                    is_simulated: bool = False, external_id: str | None = None) -> tuple[str, bool]:
        """Insert a report. Returns (message_id, created). Re-importing the very same report (same text,
        timestamp, source and id) returns the existing id with created=False. The same wording at a
        different time is a genuinely separate report and is kept."""
        if text is None or not str(text).strip():
            raise ValueError("Message text is empty.")
        ts_source = "provided" if timestamp else "import_time"
        timestamp = timestamp or now_iso()
        h = self.content_hash(text, timestamp, source, external_id)
        with self.conn() as c:
            row = c.execute("SELECT id FROM messages WHERE content_hash = ?", (h,)).fetchone()
            if row:
                return row["id"], False
            mid = "MSG-" + uuid.uuid4().hex[:8].upper()
            c.execute(
                "INSERT INTO messages (id, original_text, normalized_text, timestamp, timestamp_source, source,"
                " is_simulated, content_hash, imported_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (mid, text, normalize_text(text), timestamp, ts_source, source, int(is_simulated), h, now_iso()),
            )
            return mid, True

    def save_analysis(self, message_id: str, labels: list[str], scores: dict[str, float], confidence: float,
                      low_confidence: bool, model_name: str, entities: list[dict], location: dict,
                      people_count: str) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM classifications WHERE message_id = ?", (message_id,))
            c.execute("DELETE FROM entities WHERE message_id = ?", (message_id,))
            c.executemany(
                "INSERT INTO classifications (message_id, label, score, assigned, model_name, created_at) VALUES (?,?,?,?,?,?)",
                [(message_id, l, s, int(l in labels), model_name, now_iso()) for l, s in scores.items()],
            )
            c.executemany(
                "INSERT INTO entities (message_id, type, value, evidence, start_char, end_char, source) VALUES (?,?,?,?,?,?,?)",
                [(message_id, e["type"], e["value"], e["evidence"], e["start"], e["end"], e["source"]) for e in entities],
            )
            c.execute(
                "UPDATE messages SET location_text=?, location_status=?, lat=?, lon=?, location_source=?, people_count=?,"
                " top_confidence=?, low_confidence=?, model_name=? WHERE id=?",
                (location["name"], location["status"], location["lat"], location["lon"], location.get("source"),
                 people_count, confidence, int(low_confidence), model_name, message_id),
            )

    def get_message(self, message_id: str) -> dict | None:
        with self.conn() as c:
            m = c.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
            if not m:
                return None
            out = dict(m)
            out["classifications"] = [dict(r) for r in c.execute(
                "SELECT label, score, assigned FROM classifications WHERE message_id=? ORDER BY score DESC", (message_id,))]
            out["predicted_labels"] = [r["label"] for r in out["classifications"] if r["assigned"]]
            out["entities"] = [dict(r) for r in c.execute(
                "SELECT type, value, evidence, start_char, end_char, source FROM entities WHERE message_id=? ORDER BY start_char", (message_id,))]
            rev = c.execute("SELECT * FROM reviews WHERE message_id=? ORDER BY id DESC LIMIT 1", (message_id,)).fetchone()
            out["latest_review"] = dict(rev) if rev else None
            out["final_labels"] = (json.loads(rev["corrected_labels"]) if rev and rev["corrected_labels"] else out["predicted_labels"])
            return out

    def messages_df(self) -> pd.DataFrame:
        """One row per message with effective labels (human corrections win) and extracted resources."""
        with self.conn() as c:
            msgs = pd.read_sql_query("SELECT * FROM messages ORDER BY timestamp DESC", c)
            cls = pd.read_sql_query("SELECT message_id, label FROM classifications WHERE assigned=1", c)
            res = pd.read_sql_query("SELECT message_id, value FROM entities WHERE type='RESOURCE'", c)
            rev = pd.read_sql_query(
                "SELECT message_id, corrected_labels FROM reviews WHERE id IN (SELECT MAX(id) FROM reviews GROUP BY message_id)", c)
            dup = pd.read_sql_query(
                "SELECT message_id FROM duplicate_suggestions WHERE status='pending' GROUP BY message_id", c)
        if msgs.empty:
            msgs["labels"], msgs["resources"], msgs["has_dup"] = [], [], []
            return msgs
        pred = cls.groupby("message_id")["label"].apply(list).to_dict()
        corr = {r.message_id: json.loads(r.corrected_labels) for r in rev.itertuples() if r.corrected_labels}
        resources = res.groupby("message_id")["value"].apply(list).to_dict()
        msgs["predicted_labels"] = msgs["id"].map(lambda i: pred.get(i, []))
        msgs["labels"] = msgs["id"].map(lambda i: corr.get(i, pred.get(i, [])))
        msgs["resources"] = msgs["id"].map(lambda i: resources.get(i, []))
        msgs["has_dup"] = msgs["id"].isin(set(dup["message_id"]))
        msgs["ts"] = pd.to_datetime(msgs["timestamp"], errors="coerce")
        # Text columns that are empty for submitted reports come back as NaN; use None so `x or fallback` and string
        # operations behave consistently everywhere (dashboard and API).
        for col in TEXT_COLUMNS:
            if col in msgs.columns:
                msgs[col] = msgs[col].astype(object).where(msgs[col].notna(), None)
        return msgs

    # ------------------------------------------------------------------ duplicates
    def add_duplicate_suggestions(self, message_id: str, suggestions: list, backend: str) -> int:
        n = 0
        with self.conn() as c:
            for s in suggestions:
                cur = c.execute(
                    "INSERT OR IGNORE INTO duplicate_suggestions (message_id, other_id, similarity, adjusted_score,"
                    " location_relation, time_gap_hours, reason, backend, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (message_id, s.other_id, s.similarity, s.adjusted_score, s.location_relation,
                     s.time_gap_hours, s.reason, backend, now_iso()))
                n += cur.rowcount
        return n

    def list_duplicates(self, message_id: str | None = None, status: str | None = None) -> list[dict]:
        q = ("SELECT d.*, a.original_text AS text_a, a.location_text AS loc_a, a.timestamp AS ts_a,"
             " b.original_text AS text_b, b.location_text AS loc_b, b.timestamp AS ts_b"
             " FROM duplicate_suggestions d JOIN messages a ON a.id=d.message_id JOIN messages b ON b.id=d.other_id WHERE 1=1")
        args: list = []
        if message_id:
            q += " AND (d.message_id = ? OR d.other_id = ?)"
            args += [message_id, message_id]
        if status:
            q += " AND d.status = ?"
            args.append(status)
        with self.conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY d.adjusted_score DESC", args)]

    def decide_duplicate(self, suggestion_id: int, decision: str, reviewer: str) -> None:
        if decision not in {"confirmed", "rejected", "pending"}:
            raise ValueError("decision must be confirmed, rejected or pending")
        with self.conn() as c:
            cur = c.execute("UPDATE duplicate_suggestions SET status=?, decided_by=?, decided_at=? WHERE id=?",
                            (decision, reviewer, now_iso(), suggestion_id))
            if cur.rowcount == 0:
                raise KeyError(f"No duplicate suggestion with id {suggestion_id}")

    # ------------------------------------------------------------------ reviews & locations
    def add_review(self, message_id: str, reviewer: str, status: str, corrected_labels: list[str] | None = None,
                   corrected_location: str | None = None, lat: float | None = None, lon: float | None = None,
                   notes: str = "") -> int:
        if status not in REVIEW_STATUSES or status == "pending":
            raise ValueError(f"status must be one of {REVIEW_STATUSES[1:]}")
        if not reviewer.strip():
            raise ValueError("Reviewer name is required.")
        bad = [x for x in (corrected_labels or []) if x not in config.LABELS]
        if bad:
            raise ValueError(f"Unknown label(s): {bad}")
        if (lat is None) != (lon is None):
            raise ValueError("Provide both latitude and longitude, or neither.")
        if lat is not None and not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Latitude/longitude out of range.")
        with self.conn() as c:
            if not c.execute("SELECT 1 FROM messages WHERE id=?", (message_id,)).fetchone():
                raise KeyError(f"No message with id {message_id}")
            cur = c.execute(
                "INSERT INTO reviews (message_id, reviewer, status, corrected_labels, corrected_location, corrected_lat,"
                " corrected_lon, notes, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (message_id, reviewer.strip(), status, json.dumps(corrected_labels) if corrected_labels else None,
                 corrected_location, lat, lon, notes, now_iso()))
            c.execute("UPDATE messages SET review_status=? WHERE id=?", (status, message_id))
            if corrected_location:
                has = lat is not None
                c.execute("UPDATE messages SET location_text=?, location_status=?, lat=?, lon=?, location_source=? WHERE id=?",
                          (corrected_location, "manual" if has else "unverified", lat, lon, "reviewer", message_id))
            return cur.lastrowid

    def set_location(self, message_id: str, name: str, lat: float | None, lon: float | None) -> None:
        self.add_review(message_id, "location-editor", "corrected", None, name, lat, lon, "Manual location correction")

    def reviewed_export(self) -> pd.DataFrame:
        with self.conn() as c:
            return pd.read_sql_query(
                "SELECT m.id AS message_id, m.timestamp, m.original_text, m.location_text AS location, m.location_status,"
                " m.lat, m.lon, m.is_simulated, r.status AS review_status, r.reviewer, r.corrected_labels, r.notes,"
                " r.created_at AS reviewed_at FROM messages m JOIN reviews r ON r.message_id = m.id"
                " WHERE r.id IN (SELECT MAX(id) FROM reviews GROUP BY message_id) ORDER BY r.created_at DESC", c)

    # ------------------------------------------------------------------ evaluations & housekeeping
    def add_evaluation(self, kind: str, dataset: str, n_train: int | None, n_test: int | None,
                       metrics: dict, is_synthetic: bool = True) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO model_evaluations (kind, dataset, n_train, n_test, is_synthetic, metrics_json, created_at)"
                      " VALUES (?,?,?,?,?,?,?)", (kind, dataset, n_train, n_test, int(is_synthetic), json.dumps(metrics), now_iso()))

    def latest_evaluation(self, kind: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM model_evaluations WHERE kind=? ORDER BY id DESC LIMIT 1", (kind,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["metrics"] = json.loads(d.pop("metrics_json"))
        return d

    def counts(self) -> dict:
        with self.conn() as c:
            g = lambda q: c.execute(q).fetchone()[0]  # noqa: E731
            return {"messages": g("SELECT COUNT(*) FROM messages"),
                    "simulated": g("SELECT COUNT(*) FROM messages WHERE is_simulated=1"),
                    "pending": g("SELECT COUNT(*) FROM messages WHERE review_status='pending'"),
                    "reviewed": g("SELECT COUNT(*) FROM messages WHERE review_status!='pending'"),
                    "dup_pending": g("SELECT COUNT(*) FROM duplicate_suggestions WHERE status='pending'"),
                    "low_confidence": g("SELECT COUNT(*) FROM messages WHERE low_confidence=1")}

    def clear_all(self) -> None:
        with self.conn() as c:
            for t in ("duplicate_suggestions", "reviews", "classifications", "entities", "messages"):
                c.execute(f"DELETE FROM {t}")  # table names are fixed literals, not user input
