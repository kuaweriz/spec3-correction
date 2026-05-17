#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OBS_BIN="${OBS_BIN:-/opt/homebrew/bin/obs}"

if [[ ! -x "$OBS_BIN" ]]; then
  echo "OBS is not installed. Install it with: brew install --cask obs" >&2
  exit 1
fi

/usr/bin/osascript -e 'tell application "OBS" to quit' >/dev/null 2>&1 || true
sleep 0.8

python3 "$ROOT_DIR/scripts/setup_obs_bridge.py"

open -a /Applications/OBS.app --args \
  --collection "spec3 correction" \
  --profile "spec3 correction" \
  --disable-updater \
  --startvirtualcam
