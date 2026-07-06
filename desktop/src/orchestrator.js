"use strict";
// The prompt loop: natural language -> hosted gateway -> candidate scene ->
// local engine /validate (with a bounded corrective retry) -> local engine
// /run -> report + offline 3D HTML. fetch is injected so this is unit-tested
// with no network; in production main.js passes Node's global fetch.
//
// Engine contract (Phase A, latent_dirac/server/app.py):
//   GET  /schema   -> Scene JSON Schema
//   POST /validate {scene} -> 200 {ok:true} | 422 {ok:false, errors}
//   POST /run      {scene} -> 200 {report, html, accepted, losses}
//                             | 422 {ok:false, errors} | 400 {detail, error_type}
// Gateway contract (Phase D, services/ai_gateway/app.py):
//   POST /generate {prompt, schema, current_scene?, validation_error?} -> {scene}

async function postJson(fetch, url, payload) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  let body = null;
  try {
    body = await resp.json();
  } catch {
    body = null;
  }
  return { status: resp.status, ok: resp.ok, body };
}

async function generateAndRun(
  { prompt, currentScene = null },
  { fetch, gatewayUrl, engineUrl, maxRetries = 2, onStatus = () => {} }
) {
  // 1. The engine's schema is the AI's output contract. Fetch it once.
  const schemaResp = await fetch(`${engineUrl}/schema`);
  if (!schemaResp.ok) {
    throw new Error(`engine /schema failed (status ${schemaResp.status})`);
  }
  const schema = await schemaResp.json();

  // 2. Generate -> validate, retrying on a schema-validation failure only.
  let scene = null;
  let validationError = null;
  let lastErrors = null;

  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    onStatus(attempt === 0 ? "generating" : "retrying");
    const gen = await postJson(fetch, `${gatewayUrl}/generate`, {
      prompt,
      schema,
      current_scene: currentScene,
      validation_error: validationError,
    });
    if (!gen.ok) {
      throw new Error(`gateway /generate failed (status ${gen.status})`);
    }
    const candidate = gen.body && gen.body.scene;
    if (!candidate) {
      throw new Error("gateway returned no scene; retry or refine the prompt");
    }

    onStatus("validating");
    const val = await postJson(fetch, `${engineUrl}/validate`, { scene: candidate });
    if (val.status === 200 && val.body && val.body.ok) {
      scene = candidate;
      break;
    }
    if (val.status === 422) {
      lastErrors = (val.body && val.body.errors) || null;
      validationError = lastErrors;
      continue;
    }
    throw new Error(`engine /validate returned unexpected status ${val.status}`);
  }

  if (!scene) {
    const err = new Error(
      `could not produce a valid scene after ${maxRetries + 1} attempts`
    );
    err.errors = lastErrors;
    throw err;
  }

  // 3. Run locally. A run-time 400 (e.g. an element needing an absent engine)
  //    is surfaced as-is, not retried — a schema retry cannot fix it.
  onStatus("running");
  const run = await postJson(fetch, `${engineUrl}/run`, { scene });
  if (run.status === 400) {
    const err = new Error((run.body && run.body.detail) || "engine run failed");
    err.errorType = run.body && run.body.error_type;
    throw err;
  }
  if (!run.ok) {
    throw new Error(`engine /run failed (status ${run.status})`);
  }

  onStatus("done");
  return { scene, ...run.body };
}

module.exports = { generateAndRun, postJson };
