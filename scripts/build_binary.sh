#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

rm -rf build dist/pygulp-cluster
mkdir -p dist
DIST_DIR="$ROOT_DIR/dist/pygulp-cluster"

python3 -m PyInstaller \
  --clean \
  --noconfirm \
  --workpath build \
  --distpath dist \
  pygulp-cluster.spec

if [ -f "$DIST_DIR/pygulp-cluster.bin" ] && [ "$DIST_DIR/pygulp-cluster" != "$DIST_DIR/pygulp-cluster.bin" ]; then
  ln -sfn "pygulp-cluster.bin" "$DIST_DIR/pygulp-cluster"
fi

if [ -f "$DIST_DIR/pygulp-cluster" ]; then
  chmod +x "$DIST_DIR/pygulp-cluster"
fi
