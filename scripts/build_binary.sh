#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

python3 -m PyInstaller \
  --clean \
  --noconfirm \
  --onefile \
  --name pygulp-cluster \
  --paths src \
  src/pygulp/cli.py
