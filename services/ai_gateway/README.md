# Latent Dirac AI gateway

A small standalone service that turns a natural-language request into a
candidate Latent Dirac **scene** for the desktop client. It is **not**
part of the `latent-dirac` Python package and never runs simulations —
it only shapes the LLM call. The client validates and runs the returned
scene locally.

## What it does

`POST /generate` with:

```json
{
  "prompt": "a positron pair through a solenoid into an aperture",
  "schema": { ... },                // the engine's GET /schema output
  "current_scene": { ... },         // optional: edit this scene
  "validation_error": [ ... ]       // optional: fix these (retry loop)
}
```

It calls Claude with the schema as a forced `emit_scene` tool, so the
model must return an object matching the engine's contract, and responds
`{ "scene": { ... } }`. The client then posts that scene to its **local**
engine `POST /validate` / `POST /run`; on a validation error it calls
`/generate` again with `validation_error` set (bounded corrective retry).

The gateway is engine-version-agnostic: the schema always comes from the
caller, never hardcoded here.

## Run (owner infrastructure)

The Anthropic key lives **server-side only**:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...          # never shipped in the client
export GATEWAY_MODEL=claude-sonnet-5  # optional; this is the default
uvicorn services.ai_gateway.app:create_app --factory --host 0.0.0.0 --port 8080
```

Hosting, TLS, authentication, rate limiting, and cost control are owner
infrastructure and are intentionally not wired up here — this repo ships
the service code and its contract, tested with the LLM client mocked
(`test_app.py`, no key or network needed).
