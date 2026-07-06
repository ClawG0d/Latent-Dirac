# Desktop packaging (Phase E) — freeze the engine, bundle the installer

## Context

Phase E of the desktop-client plan. Phases A–D landed: the local sim-engine
HTTP API (`latent_dirac/server/`), offline 3D rendering, the hosted AI gateway
(`services/ai_gateway/`), and the Electron shell (`desktop/`). This phase makes
the shell a distributable app on macOS and Windows.

## Goal

Two artifacts, built in order and **per operating system** (both are
platform-specific):

1. Freeze the Python sim engine into a lean, self-contained binary with
   **PyInstaller**.
2. Bundle that binary plus the Electron renderer into a native installer
   (`.dmg` / NSIS `.exe`) with **electron-builder**.

The frozen engine is the sidecar the app spawns; the packaged app resolves it
under `resources/engine/latent-dirac-engine/` with no environment variable.

## What lands in the repo (verifiable here)

- `desktop/packaging/engine.spec` — the PyInstaller recipe. One-folder build
  (faster startup, no per-launch temp extraction, drops into electron-builder).
  Lean by construction: `hiddenimports` for uvicorn's dynamic modules and the
  server package; `collect_data_files("plotly")` so the inlined offline 3D HTML
  has its JS bundle; `excludes` for the heavy optional stacks (jax, openpmd,
  uproot, xsuite, matplotlib). The vendored Geant4 tree is never a Python
  import, so it can never be pulled in.
- `desktop/packaging/engine_entry.py` — the frozen entry point; calls
  `latent_dirac.server.__main__.main()` (import path verified on this box).
- `desktop/packaging/build_engine.sh` — freezes then smoke-tests that the
  binary prints the `PORT <n>` line the client reads (macOS/Linux; Windows uses
  the documented one-line command).
- `desktop/package.json` `build.extraResources` — maps
  `packaging/dist/latent-dirac-engine` → `resources/engine/latent-dirac-engine`.
- `desktop/src/config.js` `engineSpawnSpec(env, {isPackaged, resourcesPath,
  platform})` — resolution order: explicit `LATENT_DIRAC_ENGINE_CMD` →
  packaged bundled binary (via `process.resourcesPath`) → dev `python -m`.
  Unit-tested cross-platform (macOS path, Windows `.exe`, override precedence).
- `desktop/packaging/README.md` — the full recipe and the owner-side steps.

## What is owner-side (documented, not automated)

- **Code signing + notarization.** macOS Developer ID + notarization; Windows
  Authenticode. These use the owner's certificates/secrets and are release-time
  steps.
- **Cross-platform build CI.** Building the `.exe` needs a Windows runner; per
  `TASK-SPLIT.md` the installer-build lane is owner/Windows-side. This phase
  does not touch `.github/workflows/ci.yml`.
- **The installer ships no secrets.** The Anthropic key stays on the hosted
  gateway; the app is pointed at the deployed gateway via
  `LATENT_DIRAC_GATEWAY_URL` (or an owner default baked into `config.js`).

## Testing

- The packaged engine-path resolution is unit-tested in
  `desktop/test/config.test.js` (macOS, Windows, override) — runs anywhere.
- The frozen-engine entry import (`latent_dirac.server.__main__.main`) is
  verified on the dev box; producing and launching the actual signed installers
  is verified on the owner's Mac and Windows machines (needs a display, native
  install, and signing certs), not in the headless CI here.

## Boundaries

New files stay under `desktop/`; nothing in `latent_dirac/`, `backends/`,
`adapters/`, `engine/`, `ci.yml`, `docs/safety_scope.md`, or the vendored
Geant4 tree is touched. No comparative-performance wording ships in the
packaging docs.
