"""One entry point for everything: ``driver-assist <command>``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

USAGE = """usage: driver-assist <command> [options]

commands:
  live        calibrate, then monitor the driver with the webcam (q quit, r recalibrate)
  dashboard   open the Streamlit dashboard (sessions, event log, live page)
  export      write logged events, timeline or sessions to CSV
  qa          guided self-test that records how the detector behaves
  preview     raw metrics overlay, no alerts
  collect     record labelled examples (alert or acted drowsy) for the learned model
  train       train the learned model from those recordings and compare it with the rules

Run `driver-assist <command> --help` for the options of a command.
"""


def _run_module(name: str) -> Callable[[], None]:
    def go() -> None:
        import importlib
        importlib.import_module(f"drowsiness.{name}").main()
    return go


def _dashboard() -> None:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise SystemExit('The dashboard needs Streamlit. Install it with:  pip install "driver-assist[dashboard]"')
    script = Path(__file__).with_name("dashboard.py")
    try:
        code = subprocess.call([sys.executable, "-m", "streamlit", "run", str(script), *sys.argv[1:]])
    except KeyboardInterrupt:  # Ctrl+C is the normal way to stop the dashboard
        code = 0
    raise SystemExit(code)


COMMANDS: Dict[str, Callable[[], None]] = {
    "live": _run_module("live"),
    "export": _run_module("export"),
    "qa": _run_module("qa"),
    "preview": _run_module("preview"),
    "collect": _run_module("collect"),
    "train": _run_module("train"),
    "dashboard": _dashboard,
}


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"Unknown command: {command}\n\n{USAGE}", file=sys.stderr)
        raise SystemExit(2)
    # The command modules read sys.argv themselves, so hand them only their own options.
    sys.argv = [f"driver-assist {command}", *rest]
    COMMANDS[command]()


if __name__ == "__main__":
    main()
