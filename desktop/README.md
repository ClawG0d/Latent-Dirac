# Latent Dirac desktop client

A cross-platform (macOS + Windows) Electron app: describe a simulation in
natural language, and it generates a scene, runs it **locally**, and shows the
result as interactive 3D. Only the AI call leaves the machine — the simulation
and your scenes/results stay local.

```
prompt ─▶ hosted AI gateway ─▶ scene JSON ─▶ local engine /validate ─(422)─▶ retry
                                               │(ok)
                                               ▼
                                         local engine /run ─▶ report + offline 3D HTML
```

## Layout

| Path | Role |
| --- | --- |
| `src/sidecar.js` | spawn the local sim engine, read its `PORT`, expose `baseUrl` + `stop()` |
| `src/orchestrator.js` | the prompt → gateway → validate → run loop (bounded retry) |
| `src/config.js` | gateway URL + retry count + engine launch spec (env-overridable) |
| `main.js` | Electron main: window, sidecar lifecycle, `run-prompt` IPC |
| `preload.js` | `contextBridge` exposing `runPrompt` / `onStatus` — no Node in the page |
| `renderer/` | chat panel + sandboxed 3D `<iframe>` (offline plotly via `srcdoc`) |
| `test/` | `node --test` over the logic modules, with injected `spawn`/`fetch` |

The logic that carries risk (engine lifecycle, the retry loop) lives in `src/`
and is unit-tested with no network, no Electron, and no Python. `main.js` and
`preload.js` are thin wiring, verified by launching the app on a machine with a
display.

## Develop

Prerequisites: the `latent-dirac` engine installed with the server extra
(`pip install -e ".[server]"` from the repo root) and Node 20+.

```bash
cd desktop
npm install
npm test          # node --test over src/ — runs anywhere, no display needed
npm start         # launches Electron; spawns the engine sidecar automatically
```

Configuration (all optional, via environment):

| Variable | Default | Meaning |
| --- | --- | --- |
| `LATENT_DIRAC_GATEWAY_URL` | `http://127.0.0.1:8080` | the hosted AI gateway |
| `LATENT_DIRAC_MAX_RETRIES` | `2` | corrective retries on a schema-invalid scene |
| `LATENT_DIRAC_PYTHON` | `python` | interpreter for the dev engine sidecar |
| `LATENT_DIRAC_ENGINE_CMD` | — | a frozen engine binary (packaged build) instead of `python -m` |

No Anthropic key lives here — it stays server-side on the gateway
(`services/ai_gateway/`). The client only knows the gateway URL.

## Packaging (Phase E, owner-side)

`npm run dist` runs electron-builder for a `.dmg` (macOS) / NSIS `.exe`
(Windows). Before that ships, the Python engine must be frozen with PyInstaller
(lean deps: numpy, pydantic, pyyaml, plotly, fastapi, uvicorn — never the
vendored Geant4 tree) and bundled as an `extraResources` entry, with
`LATENT_DIRAC_ENGINE_CMD` pointed at it. Code signing and notarization use the
owner's certificates and are release-time steps, documented in the Phase E
plan, not automated in this repo.
