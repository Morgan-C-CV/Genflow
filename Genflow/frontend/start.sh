#!/usr/bin/env bash
# Start the Genflow frontend dev server on http://127.0.0.1:5173.
# Requires the backend on 127.0.0.1:8000 (see ../backend/start.sh); /api is proxied.
set -euo pipefail

cd "$(dirname "$0")"

# Keep the npm cache inside the workspace: the file sandbox denies ~/.npm.
export npm_config_cache="${npm_config_cache:-$(cd .. && cd .. && pwd)/.npm-cache}"

if [[ ! -d node_modules ]]; then
  echo "node_modules missing - running npm install..."
  npm install
fi

exec npm run dev
