#!/usr/bin/env bash
# Creates .venv with Python 3.13 and installs the pinned requirements.
set -e
cd "$(dirname "$0")"

if ! command -v python3.13 >/dev/null 2>&1; then
    echo "Python 3.13 was not found. Install it, then run this script again." >&2
    exit 1
fi

python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo "Done. Activate with: source .venv/bin/activate"
