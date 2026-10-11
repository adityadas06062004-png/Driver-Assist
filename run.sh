#!/usr/bin/env bash
# One command for a fresh checkout: sets up a virtual environment on first run, then starts the monitor.
#   ./run.sh              live monitor
#   ./run.sh dashboard    dashboard
# Needs Python 3.9 to 3.12 (3.11 recommended).
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3.11}"
command -v "$PY" >/dev/null 2>&1 || PY=python3

if [ ! -d .venv ]; then
  echo "First run: creating .venv and installing (takes a minute)..."
  "$PY" -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -e ".[dashboard]"
fi

exec .venv/bin/driver-assist "${@:-live}"
