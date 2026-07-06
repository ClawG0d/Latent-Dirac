# Latent Dirac desktop client

A cross-platform (macOS + Windows) Electron app: a Unity-style four-quadrant
dashboard — a large 3D viewport, live physics, an AI chat, and a switchable
tools panel (Ledger / Inspector / Sweep). Describe a simulation in natural
language; it generates a scene, runs it **locally**, and shows it in 3D.
**Bring your own key (BYOK):** you enter your own Anthropic API key; it is held
only in the main process and sent only to Anthropic. The simulation and your
scenes/results never leave the machine.

```
prompt ─▶ Anthropic (your key, in main) ─▶ scene JSON ─▶ engine {op:validate} ─(invalid)─▶ retry
                                                          │(ok)
                                                          ▼
                                                    engine {op:run} ─▶ report + offline 3D HTML

the engine is a local stdio sidecar (line-delimited JSON on stdin/stdout) —
no HTTP, no port; it talks only to the app over pipes.
```

## Layout

| Path | Role |
| --- | --- |
| `src/sidecar.js` | spawn the engine, await its `{ready:true}` line, expose `request({op,...})` (id-correlated JSON-RPC) + `stop()` |
| `src/ai.js` | BYOK Anthropic client — shape the forced `emit_scene` tool call, extract the scene; categorized errors |
| `src/orchestrator.js` | the prompt → generate → validate → run loop (bounded retry); `runScene` runs a loaded scene directly; categorized errors |
| `src/scene_file.js` | serialize / parse a scene for Save & Load |
| `src/errors.js` | `categorized(message, category)` helper shared by the AI client and orchestrator |
| `src/config.js` | model + retry count + engine launch spec (env-overridable) |
| `main.js` | Electron main: window, sidecar lifecycle, the BYOK key store (encrypted via `safeStorage`), `run-prompt` / `run-scene` / `save-scene` / `open-scene` / `set-api-key` / `clear-api-key` / `key-status` IPC |
| `preload.js` | `contextBridge` exposing `runPrompt` / `runScene` / `saveScene` / `openScene` / `setApiKey` / `clearApiKey` / `keyStatus` / `onStatus` — no Node, no key, in the page |
| `renderer/panels.js` | pure transforms feeding the panels (physics summary, ledger rows, scene elements, sweepable params) — UMD, unit-tested |
| `renderer/splitters.js` | draggable dock gutters — resize any pane by dragging; sizes persist (UMD; size math unit-tested) |
| `renderer/` | the four-quadrant dashboard: 3D viewport (offline plotly via a sandboxed `srcdoc` iframe), live physics, chat, and the Ledger/Inspector/Sweep tabs |
| `test/` | `node --test` over the logic modules, with injected `spawn`/`fetch` |

The logic that carries risk (engine lifecycle, the retry loop) lives in `src/`
and is unit-tested with no network, no Electron, and no Python. `main.js` and
`preload.js` are thin wiring, verified by launching the app on a machine with a
display.

Beyond the core chat → run → 3D flow, the UI offers: example-prompt chips to
start from, **New** (fresh scene), **Save** (write the current scene to a
`.json` file), and **Load** (open a scene file and run it directly, skipping the
AI step). Failures are shown with a category-specific hint (AI service
unreachable, engine not responding, the AI couldn't produce a valid scene, or a
valid scene that failed to run).

## Develop

Prerequisites: the `latent-dirac` engine installed with the server extra
(`pip install -e ".[server]"` from the repo root) and Node 20+.

```bash
cd desktop
npm install
npm test          # node --test over src/ — runs anywhere, no display needed
npm start         # launches Electron; spawns the engine sidecar automatically
```

On first launch, click **Key** to paste your Anthropic API key. It is stored on
this machine only — encrypted via the OS keychain (`safeStorage`) when
available — held in the main process, and sent only to Anthropic. The renderer
never sees it; only whether one is set.

Configuration (all optional, via environment):

| Variable | Default | Meaning |
| --- | --- | --- |
| `LATENT_DIRAC_MODEL` | `claude-sonnet-5` | the Anthropic model used for scene generation |
| `LATENT_DIRAC_MAX_RETRIES` | `2` | corrective retries on a schema-invalid scene |
| `LATENT_DIRAC_PYTHON` | `python` | interpreter for the dev engine sidecar |
| `LATENT_DIRAC_ENGINE_CMD` | — | a frozen engine binary (packaged build) instead of `python -m` |

The API key is entered in-app (not an env var) so it never lands in shell
history or a committed file. `services/ai_gateway/` remains in the repo as an
optional hosted-backend alternative, but the desktop client is BYOK and does
not require it.

## Packaging (Phase E)

Two steps, built per OS: freeze the Python engine with PyInstaller, then bundle
it + the renderer into a `.dmg`/`.exe` with electron-builder. The packaged app
auto-resolves the bundled engine under `resources/engine/` (no env var needed).
Full recipe, the lean-freeze rationale, and the owner-side release steps (code
signing, notarization, the Windows build lane) are in
[`packaging/README.md`](packaging/README.md).
