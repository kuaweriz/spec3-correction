#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_PATH="$HOME/Desktop/spec3 correction.app"
BUNDLE_ID="local.spec3-correction"
MODE="${1:-}"
BINARY_PATH="$APP_PATH/Contents/MacOS/spec3 correction"

quit_running_app() {
  /usr/bin/osascript -e "tell application id \"$BUNDLE_ID\" to quit" >/dev/null 2>&1 || true
  sleep 0.8
}

build_app() {
  cd "$ROOT_DIR"
  python3 scripts/package_macos_app.py
  /usr/bin/codesign --force --deep --sign - "$APP_PATH"
  /usr/bin/codesign --verify --verbose=2 "$APP_PATH"
  /usr/bin/plutil -lint "$APP_PATH/Contents/Info.plist" >/dev/null
}

launch_app() {
  /usr/bin/open "$APP_PATH"
}

verify_launch() {
  sleep 2
  if /usr/bin/pgrep -f "$APP_PATH/Contents/MacOS/spec3 correction" >/dev/null; then
    echo "spec3 correction is running"
  else
    echo "spec3 correction did not stay running" >&2
    exit 1
  fi
}

quit_running_app
build_app

case "$MODE" in
  --debug)
    /usr/bin/lldb -- "$BINARY_PATH"
    ;;
  --verify)
    launch_app
    verify_launch
    ;;
  --logs)
    launch_app
    verify_launch
    /usr/bin/tail -f "$HOME/Library/Logs/spec3 correction.log"
    ;;
  --telemetry)
    launch_app
    verify_launch
    /usr/bin/log stream --info --style compact --predicate "process == \"spec3 correction\""
    ;;
  "" )
    launch_app
    ;;
  * )
    echo "Usage: $0 [--debug|--verify|--logs|--telemetry]" >&2
    exit 2
    ;;
esac
