"""Command line:  python -m ingestion status | refresh [--source NAME ...] [--force]"""
import argparse
import json

from ingestion.service import IngestionService


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m ingestion")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    r = sub.add_parser("refresh")
    r.add_argument("--source", action="append", help='e.g. --source GDACS (repeatable)')
    r.add_argument("--force", action="store_true", help="skip the normal interval (a short minimum still applies)")
    args = ap.parse_args()
    svc = IngestionService()
    if args.cmd == "status":
        for s in svc.source_status():
            print(f"{s['name']:<12} health={s['health']:<14} connected={s['connected']} last_success={s['last_success_at']} items={s['items']} {s['config_note']}")
    else:
        for res in svc.refresh(args.source, trigger="cli", force=args.force):
            print(json.dumps(res, default=str))


if __name__ == "__main__":
    main()
