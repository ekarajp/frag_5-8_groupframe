#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ -x .venv/bin/python ]; then
    exec .venv/bin/python -B -X utf8 run.py "$@"
fi
exec python3 -B -X utf8 run.py "$@"
