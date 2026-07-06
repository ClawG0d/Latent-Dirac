# Packaging the Latent Dirac desktop app (Phase E)

Two artifacts, built in order, **per operating system** (a frozen binary and a
native installer are both platform-specific — macOS on a Mac, Windows on
Windows):

1. **Freeze the Python sim engine** into a lean, self-contained binary with
   PyInstaller.
2. **Bundle** that binary + the Electron renderer into a native installer
   (`.dmg` / `.exe`) with electron-builder.

The frozen engine is the sidecar the app spawns; at runtime the packaged app
resolves it under `resources/engine/latent-dirac-engine/` (see
`src/config.js` → `engineSpawnSpec`, packaged branch).

## 1. Freeze the engine

```bash
# from the repo root, in a clean venv
# NON-editable install (editable installs are not reliably collected by
# PyInstaller); viz brings plotly for the inlined offline 3D
pip install ".[viz]" pyinstaller

cd desktop/packaging
./build_engine.sh                 # macOS / Linux
#   → dist/latent-dirac-engine/latent-dirac-engine  (+ smoke-tests a stdio request)
```

Windows (PowerShell), same spec:

```powershell
python -m PyInstaller engine.spec --noconfirm
#   → dist\latent-dirac-engine\latent-dirac-engine.exe
```

`engine.spec` produces a **one-folder** build (faster startup, no per-launch
temp extraction, drops straight into electron-builder). It is lean by
construction: only the engine's real import closure ships — numpy, pydantic,
pyyaml, plotly (with its JS bundle, needed for the inlined offline 3D HTML;
the stdio bridge needs no web framework). The heavy optional stacks (jax,
openpmd, uproot, xsuite,
matplotlib) are excluded, and the vendored Geant4 tree is never a Python
import so it cannot be pulled in.

## 2. Build the installer

```bash
cd desktop
npm install
npm run dist            # electron-builder → dist/  (.dmg on macOS, .exe on Windows)
```

electron-builder copies `packaging/dist/latent-dirac-engine` into the app as
`resources/engine/latent-dirac-engine` (the `extraResources` mapping in
`package.json`). No env var is needed in the packaged app — `engineSpawnSpec`
finds the bundled binary via `process.resourcesPath`.

## Verify a packaged build

Launch the installed app; it should spawn the frozen engine, health-check it,
and render a demo scene offline. Because that needs a display and a native
install, it is verified on the owner's Mac and Windows machines, not in the
headless CI here. The risk-carrying logic (sidecar lifecycle, the packaged
engine-path resolution, the retry loop) is unit-tested cross-platform in
`../test` and runs in any environment.

## Owner-side release steps (not automated in this repo)

- **Code signing + notarization.** macOS needs a Developer ID certificate and
  notarization (`electron-builder` reads `CSC_LINK`/`CSC_KEY_PASSWORD` and the
  notarization credentials); Windows needs an Authenticode certificate. These
  use the owner's secrets and are release-time steps, documented here rather
  than committed.
- **Cross-platform build CI.** Building the `.exe` requires a Windows runner;
  per `TASK-SPLIT.md` the installer-build CI lane is owner/Windows-side. This
  phase does not touch `.github/workflows/ci.yml`.
- **The AI is BYOK.** The user enters their own Anthropic API key in-app; it is
  held only in the main process (encrypted via the OS keychain when available)
  and sent only to Anthropic. The installer ships no secrets and needs no
  gateway. (`services/ai_gateway/` remains in the repo as an optional hosted
  alternative, not used by the packaged client.)
