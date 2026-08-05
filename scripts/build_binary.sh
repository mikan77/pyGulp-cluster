#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

rm -rf build dist/pygulp-cluster
mkdir -p dist

python3 -m PyInstaller \
  --clean \
  --noconfirm \
  --workpath build \
  --distpath dist \
  pygulp-cluster.spec
