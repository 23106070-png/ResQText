"""Automatic polling. Safe to start several times: every run goes through the database lease lock and each source's
minimum interval, so duplicate schedulers (Streamlit reruns, several workers, a second terminal) cannot double-fetch.

Recommended: run it as its own process:   python -m ingestion.scheduler
Optional:    RESQTEXT_AUTOPOLL=1 starts a background thread inside the API process."""
from __future__ import annotations

import threading
import time

import config
from ingestion.service import IngestionService

_thread: threading.Thread | None = None
_stop = threading.Event()
_start_lock = threading.Lock()


def _loop(service: IngestionService) -> None:
    while not _stop.is_set():
        try:
            service.run_due()
        except Exception:  # keep polling alive; details are in ingestion_errors
            pass
        _stop.wait(config.SCHEDULER_TICK_SECONDS)


def start_background(service: IngestionService) -> bool:
    """Start at most one polling thread per process. Returns False if one is already running."""
    global _thread
    with _start_lock:
        if _thread and _thread.is_alive():
            return False
        _stop.clear()
        _thread = threading.Thread(target=_loop, args=(service,), name="resqtext-poller", daemon=True)
        _thread.start()
        return True


def stop_background() -> None:
    _stop.set()


def main() -> None:
    service = IngestionService()
    print(f"ResQText poller running (tick {config.SCHEDULER_TICK_SECONDS:.0f}s). Ctrl+C to stop.")
    try:
        while True:
            for r in service.run_due():
                print(f"[{r['source']}] {r['status']}: fetched={r.get('fetched', 0)} new={r.get('inserted', 0)} "
                      f"updated={r.get('updated', 0)} errors={r.get('errors', 0)} {r.get('message') or ''}")
            time.sleep(config.SCHEDULER_TICK_SECONDS)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
