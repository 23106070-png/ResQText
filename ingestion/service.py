"""Runs adapters safely: one job per source at a time (DB lease lock), minimum intervals, back-off after failures,
per-item error isolation, and a full audit trail in ingestion_runs / ingestion_errors."""
from __future__ import annotations

import os
import uuid
from datetime import timedelta

import config
from database.db import Database
from ingestion.base import IngestContext
from ingestion.http import Fetcher, default_fetcher
from ingestion.models import SourceError, iso, parse_dt, utc_now
from ingestion.registry import default_adapters
from nlp.pipeline import Pipeline


class IngestionService:
    def __init__(self, db: Database | None = None, pipeline: Pipeline | None = None, fetch: Fetcher | None = None,
                 adapters: dict | None = None, now_fn=utc_now):
        self.db = db or Database()
        self.pipeline = pipeline or Pipeline(self.db)
        self.fetch = fetch or default_fetcher
        self.adapters = adapters or default_adapters()
        self.now_fn = now_fn

    # ------------------------------------------------------------------ status
    def source_status(self) -> list[dict]:
        out, now = [], self.now_fn()
        for name, a in self.adapters.items():
            ok, why = a.configured()
            st = self.db.get_state(name)
            interval = a.interval_minutes()
            last_ok = parse_dt(st["last_success_at"])
            age = (now - last_ok).total_seconds() / 60 if last_ok else None
            if not ok:
                health = "not_configured"
            elif last_ok is None:
                health = "failing" if st["last_status"] == "failed" else "untested"
            elif st["last_status"] == "failed":
                health = "failing"
            elif age > interval * config.STALE_FACTOR:
                health = "stale"
            else:
                health = "ok"
            out.append({"name": name, "label": a.label, "kind": a.kind, "homepage": a.homepage, "attribution": a.attribution,
                        "configured": ok, "config_note": why, "needs_credentials": a.needs_credentials,
                        "credential_help": a.credential_help, "interval_minutes": interval, "connected": last_ok is not None,
                        "health": health, "last_attempt_at": st["last_attempt_at"], "last_success_at": st["last_success_at"],
                        "age_minutes": None if age is None else round(age, 1), "last_status": st["last_status"],
                        "last_error": st["last_error"], "consecutive_failures": st["consecutive_failures"],
                        "backoff_until": st["backoff_until"], "items": self.db.count_source_items(name),
                        "api_calls_today": self.db.api_calls_today(name)})
        return out

    # ------------------------------------------------------------------ one source
    def run_source(self, name: str, trigger: str = "manual", force: bool = False) -> dict:
        a = self.adapters.get(name)
        if a is None:
            return {"source": name, "status": "skipped", "message": "Unknown source."}
        ok, why = a.configured()
        if not ok:
            return {"source": name, "status": "skipped", "message": why}
        now, st = self.now_fn(), self.db.get_state(name)
        backoff = parse_dt(st["backoff_until"])
        if backoff and now < backoff:  # never bypassed, not even by a manual refresh: the service asked us to slow down
            return {"source": name, "status": "skipped", "message": f"Backing off after errors until {iso(backoff)} UTC."}
        last = parse_dt(st["last_attempt_at"])
        if last:
            min_s = config.MIN_MANUAL_REFRESH_SECONDS if force else a.interval_minutes() * 60
            wait = min_s - (now - last).total_seconds()
            if wait > 0:
                return {"source": name, "status": "skipped", "message": f"Checked recently; next check allowed in {int(wait // 60) + 1} min."}
        holder = f"{os.getpid()}-{uuid.uuid4().hex[:6]}"
        if not self.db.acquire_lock(f"ingest:{name}", holder, ttl_seconds=600):
            return {"source": name, "status": "skipped", "message": "Another refresh of this source is already running."}
        try:
            return self._run_locked(a, trigger, now)
        finally:
            self.db.release_lock(f"ingest:{name}", holder)

    def _run_locked(self, a, trigger: str, now) -> dict:
        name = a.name
        run_id = self.db.start_run(name, trigger)
        self.db.update_state(name, last_attempt_at=iso(now))
        c = dict(fetched=0, inserted=0, updated=0, unchanged=0, held_back=0, errors=0, api_calls=0)
        fatal: SourceError | None = None
        notes: list[str] = []
        try:
            res = a.fetch(IngestContext(self.db, self.fetch, now))
            c["api_calls"] = res.api_calls
            c["fetched"] = len(res.items)
            notes = list(res.notes)
            for e in res.errors:
                self.db.log_error(name, e["kind"], e["message"], run_id, e.get("item_ref"))
                c["errors"] += 1
            for item in res.items:
                try:
                    r = self.pipeline.process_external(item, retrieved_at=iso(now))
                    key = {"inserted": "inserted", "updated": "updated", "unchanged": "unchanged", "held_back": "held_back"}[r["status"]]
                    c[key] += 1
                except Exception as exc:  # isolate: one bad item must not stop the run
                    c["errors"] += 1
                    self.db.log_error(name, "item", f"{type(exc).__name__} while storing item", run_id, item.source_id)
            if res.not_modified:
                notes.append("Unchanged since the last check (HTTP 304).")
        except SourceError as exc:
            fatal = exc
        except Exception as exc:  # never leak internals; keep a short type name only
            fatal = SourceError("parse", f"Unexpected {type(exc).__name__} while processing the response.")

        if fatal:
            self.db.log_error(name, fatal.kind, fatal.message, run_id)
            c["errors"] += 1
            st = self.db.get_state(name)
            fails = st["consecutive_failures"] + 1
            if fatal.kind == "rate_limited":
                wait_min = max((fatal.retry_after or 0) / 60, a.interval_minutes() * 2, 15)
            else:
                wait_min = min(a.interval_minutes() * (2 ** min(fails, 5)), 360) if fails >= 2 else 0
            self.db.update_state(name, last_status="failed", last_error=fatal.message, last_http_status=fatal.status,
                                 consecutive_failures=fails, backoff_until=iso(now + timedelta(minutes=wait_min)) if wait_min else None)
            self.db.finish_run(run_id, "failed", message=fatal.message, **c)
            return {"source": name, "status": "failed", "run_id": run_id, "kind": fatal.kind, "message": fatal.message, **c}

        status = "partial" if c["errors"] else "success"
        msg = " ".join(notes) or None
        self.db.update_state(name, last_success_at=iso(self.now_fn()), last_status=status, consecutive_failures=0,
                             backoff_until=None, last_error=(f"{c['errors']} item/page problem(s); see error log." if c["errors"] else None),
                             total_items=self.db.count_source_items(name))
        self.db.finish_run(run_id, status, message=msg, **c)
        return {"source": name, "status": status, "run_id": run_id, "message": msg, **c}

    # ------------------------------------------------------------------ many sources
    def refresh(self, names: list[str] | None = None, trigger: str = "manual", force: bool = False) -> list[dict]:
        return [self.run_source(n, trigger, force) for n in (names or list(self.adapters))]

    def run_due(self) -> list[dict]:
        """Called by the scheduler: runs only sources whose interval has elapsed (run_source enforces it)."""
        results = []
        for name in self.adapters:
            r = self.run_source(name, "auto")
            if r["status"] != "skipped":
                results.append(r)
        return results
