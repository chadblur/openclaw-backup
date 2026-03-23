#!/usr/bin/env bash
set -euo pipefail

SRC="/root/.openclaw/"
DST="/root/.openclaw-backup"
REMOTE_URL="https://github.com/chadblur/openclaw-backup.git"
BRANCH="main"
LOG_FILE="/root/.openclaw-backup-sync.log"

mkdir -p "$DST"

if [ ! -d "$DST/.git" ]; then
  git -C "$DST" init -b "$BRANCH"
fi

if git -C "$DST" remote get-url origin >/dev/null 2>&1; then
  git -C "$DST" remote set-url origin "$REMOTE_URL"
else
  git -C "$DST" remote add origin "$REMOTE_URL"
fi

cat > "$DST/.gitignore" <<'EOF'
logs/
credentials/
backups/
**/sessions/
**/node_modules/
**/lib/
**/libs/
**/.pnpm/
**/vendor/
**/venv/
**/.venv/
**/dist/
**/build/
**/coverage/
**/__pycache__/
EOF

rsync -a --delete --delete-excluded \
  --filter='P /.git/' \
  --exclude='logs/' \
  --exclude='credentials/' \
  --exclude='backups/' \
  --exclude='.git/' \
  --filter='- **/.git/' \
  --filter='- **/.gitmodules' \
  --filter='- **/.gitignore' \
  --filter='- **/sessions/' \
  --filter='- **/node_modules/' \
  --filter='- **/lib/' \
  --filter='- **/libs/' \
  --filter='- **/.pnpm/' \
  --filter='- **/vendor/' \
  --filter='- **/venv/' \
  --filter='- **/.venv/' \
  --filter='- **/dist/' \
  --filter='- **/build/' \
  --filter='- **/coverage/' \
  --filter='- **/__pycache__/' \
  "$SRC" "$DST/"

python3 <<'PY'
import json
from pathlib import Path

ROOT = Path('/root/.openclaw-backup')
TARGETS = [p for p in ROOT.glob('openclaw.json*') if p.is_file()]
SENSITIVE_KEYS = ('token', 'secret', 'password', 'api_key', 'apikey', 'key', 'webhook', 'cookie', 'authorization')


def redact(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            lk = str(k).lower()
            if any(s in lk for s in SENSITIVE_KEYS):
                out[k] = '<REDACTED>'
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    return obj

for path in TARGETS:
    try:
        data = json.loads(path.read_text())
        path.write_text(json.dumps(redact(data), ensure_ascii=False, indent=2) + '\n')
    except Exception:
        pass
PY

cd "$DST"
git add -A
if ! git diff --cached --quiet; then
  git commit -m "backup: sync openclaw state $(date '+%Y-%m-%d %H:%M:%S %z')"
fi

git push -u origin "$BRANCH" >>"$LOG_FILE" 2>&1
