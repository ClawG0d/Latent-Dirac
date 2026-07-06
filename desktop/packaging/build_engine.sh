#!/usr/bin/env bash
# Freeze the local sim engine (the stdio bridge) into a lean one-folder binary
# with PyInstaller, then smoke-test that it answers a JSON request on stdio.
#
# Run on the target OS (a frozen binary is platform-specific). On Windows use
# the documented command in README.md instead of this POSIX script.
#
#   cd desktop/packaging && ./build_engine.sh
#
# Requires PyInstaller and latent-dirac installed NON-editable (editable installs
# are not reliably collected by PyInstaller — the frozen binary would fail with
# "No module named 'latent_dirac'"), with the viz extra (brings plotly, needed
# for the inlined offline 3D):
#   pip install "../..[viz]" pyinstaller
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

echo "==> Smoke test: the binary must answer a JSON request on stdin/stdout"
out="$(mktemp)"
# send a schema request; expect the ready line then a {"ok": true, ...} response
printf '%s\n' '{"id":1,"op":"schema"}' | "$bin" >"$out" 2>/dev/null || true

if grep -q '"ready"' "$out" && grep -q '"ok": *true' "$out"; then
  echo "==> OK: engine printed ready and answered the schema request"
  rm -f "$out"
else
  echo "ERROR: the frozen engine did not answer the stdio request" >&2
  echo "--- output ---" >&2; head -5 "$out" >&2
  rm -f "$out"
  exit 1
fi

echo "==> Built: $bin"
echo "Next: from desktop/, run 'npm run dist' to bundle it into an installer."
