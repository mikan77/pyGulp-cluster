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

if [ -f "$DIST_DIR/pygulp-cluster" ]; then
  chmod +x "$DIST_DIR/pygulp-cluster"
else
  echo "[ERROR] Expected onedir executable not found: $DIST_DIR/pygulp-cluster"
  exit 1
fi

if [ -f "$ROOT_DIR/dist/pygulp-cluster.bin" ]; then
  echo "[WARN] Legacy top-level pygulp-cluster.bin found. It may be a onefile build. Removing."
  rm -f "$ROOT_DIR/dist/pygulp-cluster.bin"
fi
