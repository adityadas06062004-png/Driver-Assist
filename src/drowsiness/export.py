"""Export logged data to CSV.

    python -m drowsiness.export                       # events, all sessions -> events.csv
    python -m drowsiness.export --what samples --session 3 --out risk.csv
    python -m drowsiness.export --list                # show saved sessions
"""

from __future__ import annotations

import argparse
import time

from .storage import DEFAULT_DB, Storage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--what", choices=("events", "samples", "sessions"), default="events")
    parser.add_argument("--session", type=int, default=None, help="only this session id")
    parser.add_argument("--out", default=None, help="file to write (default: <what>.csv)")
    parser.add_argument("--list", action="store_true", help="list sessions and exit")
    args = parser.parse_args()

    with Storage(args.db) as db:
        if args.list:
            for s in db.sessions():
                start = time.strftime("%Y-%m-%d %H:%M", time.localtime(s["started_at"]))
                print(f"#{s['id']}  {start}  {db.summary(s['id'])['duration_s']:.0f}s")
            return
        out = args.out or f"{args.what}.csv"
        n = db.export_csv(out, args.what, args.session)
        print(f"Wrote {n} rows to {out}")


if __name__ == "__main__":
    main()
