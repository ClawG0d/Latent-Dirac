#!/usr/bin/env bash
# Freeze the local sim engine into a lean one-folder binary with PyInstaller,
# then smoke-test that it prints the PORT line the desktop client reads.
#
# Run on the target OS (a frozen binary is platform-specific). On Windows use
# the documented command in README.md instead of this POSIX script.
#
#   cd desktop/packaging && ./build_engine.sh
#
# Requires: the engine installed with the server extra plus PyInstaller, e.g.
#   pip install -e "../..[server]" pyinstaller
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

echo "==> Freezing latent-dirac-engine with PyInstaller"
python -m PyInstaller engine.spec --noconfirm

bin="dist/latent-dirac-engine/latent-dirac-engine"
if [ ! -x "$bin" ]; then
  echo "ERROR: expected frozen binary not found at $bin" >&2
  exit 1
fi

echo "==> Smoke test: the binary must print 'PORT <n>' on stdout"
out="$(mktemp)"
# same args the client uses at runtime (see src/config.js engineSpawnSpec)
"$bin" --host 127.0.0.1 --port 0 >"$out" 2>/dev/null &
pid=$!
for _ in $(seq 1 50); do
  grep -q '^PORT [0-9]' "$out" && break
  sleep 0.2
done
kill "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true

if grep -q '^PORT [0-9]' "$out"; then
  echo "==> OK: $(grep '^PORT' "$out" | head -1)"
  rm -f "$out"
else
  echo "ERROR: the frozen engine did not print a PORT line" >&2
  rm -f "$out"
  exit 1
fi

echo "==> Built: $bin"
echo "Next: from desktop/, run 'npm run dist' to bundle it into an installer."
