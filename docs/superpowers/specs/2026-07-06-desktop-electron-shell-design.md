# Desktop Electron shell (Phase C) — chat → scene → local run → 3D

## Context

Phase C of the desktop-client plan
(`.claude/plans/repo-genesis-world-newton-issac-anasys-buzzing-church.md`).
Phases A, B, and D already landed:

- **A/B** — `latent_dirac/server/` exposes the core over localhost JSON
  (`GET /schema`, `POST /validate`, `POST /run` with an offline,
  self-contained 3D HTML string). `python -m latent_dirac.server` binds an
  ephemeral port and prints `PORT <n>` on stdout.
- **D** — `services/ai_gateway/` turns a natural-language prompt plus the
  engine's Scene JSON Schema into a candidate scene via a forced Claude
  tool call (`POST /generate` → `{scene}`), accepting a prior validation
  error for a corrective retry. Hosted by the owner; holds the Anthropic
  key server-side.

Phase C is the **Electron shell** that ties them together: a chat panel
where the user describes a simulation, and a 3D panel that shows the run.
The simulation runs **locally** in the bundled engine; only the AI call
goes to the hosted gateway.

## Goal

A cross-platform (Mac + Windows) Electron app whose main process manages a
local Python sim-engine sidecar, and whose renderer offers a chat panel
(prompt in; report + fidelity tiers + status out) and a 3D panel (the
engine's offline `/run` HTML). One prompt runs the full loop:

```
prompt ─▶ gateway /generate ─▶ scene ─▶ engine /validate ─(422)─▶ retry
                                          │(ok)
                                          ▼
                                    engine /run ─▶ {report, html, accepted, losses}
                                          ▼
                              chat shows report; 3D panel shows html
```

## Architecture — thin Electron, testable core

Electron's `main` and `preload` import `electron` and cannot run in a
headless unit test. So the **logic** lives in plain Node modules that take
their dependencies by injection (`spawn`, `fetch`), and the Electron files
stay thin wiring. This keeps the load-bearing behaviour testable on the
CPU-only Mac with `node --test` (no display, no Electron install needed to
run the logic tests).

```
desktop/
  package.json          electron + electron-builder devDeps; scripts
  src/
    sidecar.js          spawn the engine, parse "PORT <n>", expose baseUrl + stop()
    orchestrator.js     the prompt→scene→validate→run loop (injected fetch)
    config.js           gateway URL + retry count (env-overridable)
  main.js               Electron main: window, app lifecycle, IPC → sidecar+orchestrator
  preload.js            contextBridge: expose runPrompt()/onStatus() to renderer
  renderer/
    index.html          chat panel + 3D <iframe> (sandboxed)
    renderer.js         chat UI; calls window.api.runPrompt; renders report + html
    styles.css
  test/
    sidecar.test.js     node --test: PORT parse, ready/timeout, stop()
    orchestrator.test.js node --test: happy path, 422 retry loop, run 400 surface
  README.md             dev-run + the Phase E packaging handoff
```

### `src/sidecar.js` — engine lifecycle (injectable spawn)

`startSidecar({ spawn, command, args, readyTimeoutMs })` → Promise of
`{ baseUrl, port, stop() }`.

- Spawns the engine (default `python -m latent_dirac.server --host 127.0.0.1
  --port 0`; in the frozen build, the PyInstaller binary path instead).
- Reads stdout line-by-line for `PORT <n>`; resolves `baseUrl =
  http://127.0.0.1:<n>`. This is the exact contract `latent_dirac/server/
  __main__.py` prints.
- Rejects on: process exit before a port line, or `readyTimeoutMs` elapsed.
- `stop()` kills the child (SIGTERM; the server closes its socket on
  shutdown).
- No `electron` import — unit-tested with a fake child process (an
  `EventEmitter` with a fake `stdout`).

### `src/orchestrator.js` — the prompt loop (injectable fetch)

`generateAndRun({ prompt, currentScene }, { fetch, gatewayUrl, engineUrl,
maxRetries, onStatus })` → `{ scene, report, html, accepted, losses }`.

1. `GET engineUrl/schema` once → `schema`.
2. Loop up to `maxRetries + 1` times:
   a. `POST gatewayUrl/generate` `{ prompt, schema, current_scene,
      validation_error }` → `{ scene }`.
   b. `POST engineUrl/validate` `{ scene }`. On **200** break (valid). On
      **422** set `validation_error = body.errors`, `onStatus('retrying')`,
      continue. Any other status → throw (unexpected).
3. If no valid scene after the retries → throw a clear
   "could not produce a valid scene" error carrying the last errors.
4. `POST engineUrl/run` `{ scene }`. **200** → return the body plus the
   scene. **400** → throw a run-time error carrying `detail`/`error_type`
   (e.g. an engine-needing element with no binary) — surfaced to chat, not
   retried (a schema retry cannot fix a missing engine). **422** here is
   defensive (validate already passed) → throw.
5. `onStatus` fires `generating` / `validating` / `retrying` / `running` /
   `done` so the renderer can show progress.

Validate-before-run is deliberate: validation is cheap and gives clean
retry semantics, so we never spend a simulation run on a scene the schema
rejects.

### `main.js` — Electron wiring (thin, not unit-tested)

- On `app.whenReady`: `startSidecar(...)`, health-check `GET /health`,
  create the `BrowserWindow` (with `preload.js`, `contextIsolation: true`,
  `nodeIntegration: false`).
- `ipcMain.handle('run-prompt', ...)` calls `generateAndRun` with the live
  `engineUrl` and the configured `gatewayUrl`, forwarding `onStatus` to the
  renderer via `webContents.send('status', ...)`.
- Single-instance lock; on `window-all-closed` / `before-quit`, call
  `sidecar.stop()` so no orphan Python process survives.
- Surfaces a sidecar/health failure as a visible error, not a silent blank
  window.

### `preload.js` + renderer

- `preload.js` exposes `window.api = { runPrompt(prompt, currentScene),
  onStatus(cb) }` over `contextBridge` — no Node in the renderer.
- `renderer.js`: a prompt box + send button; on submit calls
  `window.api.runPrompt`, streams status labels, then renders the returned
  `report` (text via `pre.textContent`, with the fidelity-tier lines the
  scene report already carries) and loads `html` into a sandboxed
  `<iframe>` via `srcdoc`, so the offline plotly renders with no network.
- The iframe carries `sandbox="allow-scripts"` (no `allow-same-origin`): it
  gets an opaque origin, does **not** inherit the page's strict CSP, and so
  runs the trusted local plotly inline scripts in isolation. `srcdoc` (not a
  `blob:` URL) is required — a sandbox-without-same-origin frame cannot load
  a parent-origin blob, but inline `srcdoc` content loads fine. The page's
  own CSP stays strict (`connect-src 'none'`, no remote scripts).
- On error, shows the message (validation give-up or run-time 400) in the
  chat log; the app stays usable for the next prompt.

Owner-side GUI check (needs a display, not run in CI here): confirm the 3D
panel actually renders under the CSP. CSP inheritance for sandboxed frames
has varied across Chromium versions; if a build blanks the panel, the
fallback is to loosen the frame's script policy for the trusted local
content (or have the engine emit its own permissive CSP inside the HTML) —
documented rather than pre-applied, since it cannot be verified here.

## Contract with the engine (locked from Phase A)

- `GET /schema` → Scene JSON Schema.
- `POST /validate` `{scene}` → `200 {ok:true}` | `422 {ok:false, errors}`.
- `POST /run` `{scene, animate?, color?, max_particles?, scope_note?}` →
  `200 {report, html, accepted, losses}` | `422 {ok:false, errors}` |
  `400 {detail, error_type}`.
- `GET /health` → `{status, engine_version}`.
- Sidecar stdout prints `PORT <n>` once bound.

## Testing (runs on the CPU-only Mac, no display)

- `node --test desktop/test/*.test.js` — `sidecar.js` (PORT parse from a
  fake stdout; reject on early exit; reject on timeout; `stop()` kills) and
  `orchestrator.js` (happy path; one 422 then success; give-up after
  `maxRetries`; run-time 400 surfaced; schema fetched once). All with
  injected `spawn`/`fetch`; no network, no Electron, no Python.
- Electron GUI verification (launch, spawn real sidecar, render a demo
  scene) happens on the owner's machine with a display — noted in the
  desktop README; not run here.

## Polish increment (2026-07-06)

A follow-up pass on the shell, logic TDD'd, GUI owner-verified:

- **Save / Load a scene.** `src/scene_file.js` (`serializeScene` /
  `parseSceneFile`, unit-tested) plus main-process `save-scene` / `open-scene`
  IPC handlers using `dialog` + `fs`. A loaded scene runs through a new
  `orchestrator.runScene(scene, {...})` that validates then runs it **directly,
  skipping the gateway** — a loaded scene is authoritative. Scenes persist as
  pretty-printed JSON (the engine loads JSON as well as YAML).
- **Example-prompt chips + New scene.** The renderer seeds a few clickable
  starter prompts and a **New** button that clears `currentScene` and resets the
  3D panel, so the next prompt starts fresh rather than editing.
- **Categorized errors.** `orchestrator` errors now carry a `category`
  (`gateway-unreachable`, `gateway-error`, `engine-unreachable`,
  `validation-giveup`, `engine-runtime`); a gateway/engine network failure is
  caught and re-thrown with a clear message naming the URL. The renderer maps
  each category to a human hint.

## AI model change — BYOK (2026-07-06)

The AI integration moved from a hosted gateway (owner pays, unified billing) to
**bring-your-own-key**: the user enters their own Anthropic API key.

- The key lives **only in the Electron main process** — persisted encrypted via
  `safeStorage` (OS keychain) when available, held in memory otherwise. The
  renderer never receives it; it only learns whether one is set (`key-status`)
  and can set/clear it (`set-api-key` / `clear-api-key`).
- The client calls the Anthropic API **directly from the main process** (Node,
  so no CORS), reusing the proven request shaping (forced `emit_scene` tool,
  the honesty system prompt) now in `desktop/src/ai.js`.
- `orchestrator.generateAndRun` takes an injected `generate({prompt, schema,
  currentScene, validationError})` instead of a gateway URL; main wires it to
  `ai.generateScene({..., apiKey, model, fetch})`. The orchestrator stays
  network-free in tests (inject a fake `generate`).
- New error categories flow to the renderer: `ai-no-key`, `ai-bad-key`,
  `ai-unreachable`, `ai-error`, `ai-no-scene` (each with a UI hint).
- `services/ai_gateway/` stays in the repo as an optional hosted alternative but
  is no longer the default path; the packaged client is BYOK and ships no
  secrets.

## Boundaries / risks

- New top-level `desktop/` tree; outside the Mac/Windows Python lanes in
  `TASK-SPLIT.md`. Does not touch `latent_dirac/`, `backends/`,
  `adapters/`, `engine/`, `ci.yml`, `docs/safety_scope.md`, or the vendored
  Geant4 tree.
- No secrets in the client: the gateway holds the Anthropic key; the app
  only knows the gateway URL (env-overridable, a placeholder default until
  the owner deploys).
- Honesty discipline carries into the UI: the report's fidelity tiers and
  provenance are shown verbatim; no comparative-performance wording ships
  in UI copy.
- Packaging (freeze the sidecar with PyInstaller, bundle with
  electron-builder, signing/notarization) is **Phase E**, owner-side; this
  phase ends at a dev-runnable shell plus the logic tests.
