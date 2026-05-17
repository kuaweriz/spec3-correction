#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_PATH="$HOME/Desktop/spec3 correction.app"
BUNDLE_ID="local.spec3-correction"
EXT_BUNDLE_ID="local.spec3-correction.camera-extension"
SIGN_IDENTITY="${SPEC3_CODESIGN_IDENTITY:-Spec3 Local Code Signing Trusted}"
ENABLE_SYSTEM_EXTENSION="${SPEC3_ENABLE_SYSTEM_EXTENSION:-0}"
MODE="${1:-}"
BINARY_PATH="$APP_PATH/Contents/MacOS/spec3 correction"
EXT_PATH="$APP_PATH/Contents/Library/SystemExtensions/$EXT_BUNDLE_ID.systemextension"
APP_ENTITLEMENTS="$ROOT_DIR/scripts/spec3_app.entitlements"
EXT_ENTITLEMENTS="$ROOT_DIR/virtual_camera/Spec3CameraExtension/Spec3CameraExtension.entitlements"

quit_running_app() {
  /usr/bin/osascript -e "tell application id \"$BUNDLE_ID\" to quit" >/dev/null 2>&1 || true
  sleep 0.8
  /usr/bin/pkill -f "bin_single_window.py" >/dev/null 2>&1 || true
  /usr/bin/killall avconferenced >/dev/null 2>&1 || true
  sleep 0.5
}

build_app() {
  cd "$ROOT_DIR"
  python3 scripts/package_macos_app.py
  if /usr/bin/security find-identity -p codesigning -v | /usr/bin/grep -F "\"$SIGN_IDENTITY\"" >/dev/null; then
    if [[ -d "$EXT_PATH" ]]; then
      /usr/bin/codesign --force --options runtime --entitlements "$EXT_ENTITLEMENTS" --sign "$SIGN_IDENTITY" "$EXT_PATH"
    fi
    if [[ "$ENABLE_SYSTEM_EXTENSION" == "1" ]]; then
      /usr/bin/codesign --force --options runtime --entitlements "$APP_ENTITLEMENTS" --sign "$SIGN_IDENTITY" "$APP_PATH"
    else
      /usr/bin/codesign --force --options runtime --sign "$SIGN_IDENTITY" "$APP_PATH"
    fi
  else
    echo "Warning: '$SIGN_IDENTITY' code-signing identity not found; using ad-hoc signing." >&2
    if [[ -d "$EXT_PATH" ]]; then
      /usr/bin/codesign --force --options runtime --entitlements "$EXT_ENTITLEMENTS" --sign - "$EXT_PATH"
    fi
    if [[ "$ENABLE_SYSTEM_EXTENSION" == "1" ]]; then
      /usr/bin/codesign --force --options runtime --entitlements "$APP_ENTITLEMENTS" --sign - "$APP_PATH"
    else
      /usr/bin/codesign --force --options runtime --sign - "$APP_PATH"
    fi
  fi
  /usr/bin/codesign --verify --verbose=2 "$APP_PATH"
  if [[ -d "$EXT_PATH" ]]; then
    /usr/bin/codesign --verify --verbose=2 "$EXT_PATH"
  fi
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
