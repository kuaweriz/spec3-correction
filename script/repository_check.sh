#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m compileall \
  bin_single_window.py \
  displayers \
  model_managers \
  scripts \
  tf_models \
  utils

test -f docs/assets/spec3-banner.svg
test -f docs/assets/spec3-ui-preview.svg
grep -q "docs/assets/spec3-banner.svg" README.md
grep -q "Documentation" README.md

git diff --check

echo "Repository check passed."
