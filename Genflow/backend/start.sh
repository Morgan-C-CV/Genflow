#!/usr/bin/env bash
# Start the Genflow backend API (FastAPI/uvicorn) on 127.0.0.1:8000.
set -euo pipefail

cd "$(dirname "$0")/src"

VENV="../.venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "error: $VENV not found. Create it with:" >&2
  echo "  cd .. && UV_CACHE_DIR=\"\$(cd ../.. && pwd)/.uv-cache\" uv venv --python 3.12 .venv" >&2
  echo "  cd .. && UV_CACHE_DIR=\"\$(cd ../.. && pwd)/.uv-cache\" uv pip install --python .venv/bin/python -r src/requirements.txt" >&2
  exit 1
fi

exec "$VENV/bin/python" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 "$@"
