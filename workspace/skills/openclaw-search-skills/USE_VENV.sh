#!/usr/bin/env bash
# Simple wrapper to run search skills with the dedicated venv
set -euo pipefail

VENV="/root/.openclaw/venv/search-skills"
if [ ! -x "$VENV/bin/python" ]; then
 echo "Venv not found: $VENV" >&2
 exit1
fi

ROOT="/root/.openclaw/workspace/skills/openclaw-search-skills"

case "${1:-}" in
 search)
 shift
 exec "$VENV/bin/python" "$ROOT/search-layer/scripts/search.py" "$@"
 ;;
 fetch-thread)
 shift
 exec "$VENV/bin/python" "$ROOT/search-layer/scripts/fetch_thread.py" "$@"
 ;;
 content-extract)
 shift
 exec "$VENV/bin/python" "$ROOT/content-extract/scripts/content_extract.py" "$@"
 ;;
 mineru-extract)
 shift
 exec "$VENV/bin/python" "$ROOT/mineru-extract/scripts/mineru_extract.py" "$@"
 ;;
 *)
 echo "Usage: $0 {search|fetch-thread|content-extract|mineru-extract} [args...]" >&2
 exit2
 ;;
esac
