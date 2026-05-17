#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/build/virtual-camera"
OUT_BIN="$OUT_DIR/spec3-camera-extension-check"

mkdir -p "$OUT_DIR"

swiftc \
  -target arm64-apple-macosx14.0 \
  -framework CoreMediaIO \
  -framework CoreMedia \
  -framework CoreVideo \
  "$ROOT_DIR/virtual_camera/Spec3CameraExtension/main.swift" \
  "$ROOT_DIR/virtual_camera/Spec3CameraExtension/Spec3Provider.swift" \
  -o "$OUT_BIN"

echo "$OUT_BIN"
