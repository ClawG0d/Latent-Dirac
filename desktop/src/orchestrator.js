"use strict";
// The prompt loop: natural language -> a candidate scene (via the injected
// `generate`, which the main process wires to the BYOK Anthropic client) ->
// local engine /validate (with a bounded corrective retry) -> local engine
// /run -> report + offline 3D HTML.
//
// `generate` and `fetch` are injected so this is unit-tested with no network,
// no key, and no Python. Engine contract (Phase A, latent_dirac/server/app.py):
//   GET  /schema   -> Scene JSON Schema
//   POST /validate {scene} -> 200 {ok:true} | 422 {ok:false, errors}
//   POST /run      {scene} -> 200 {report, html, accepted, losses}
//                             | 422 {ok:false, errors} | 400 {detail, error_type}

const { categorized } = require("./errors");

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

async function fetchSchema(fetch, engineUrl) {
  try {
    const resp = await fetch(`${engineUrl}/schema`);
    if (!resp.ok) {
      throw categorized(`engine /schema failed (status ${resp.status})`, "engine-unreachable");
    }
    return await resp.json();
  } catch (err) {
    if (err.category) throw err;
    throw categorized(`cannot reach the local sim engine (${engineUrl}): ${err.message}`, "engine-unreachable");
  }
}

async function generateAndRun(
  { prompt, currentScene = null },
  { fetch, engineUrl, generate, maxRetries = 2, onStatus = () => {} }
) {
  // 1. The engine's schema is the AI's output contract. Fetch it once.
  const schema = await fetchSchema(fetch, engineUrl);

  // 2. Generate -> validate, retrying on a schema-validation failure only.
  //    `generate` throws its own categorized error (ai-no-key, ai-bad-key,
  //    ai-unreachable, ai-error, ai-no-scene); let it propagate unrelabeled.
  let scene = null;
  let validationError = null;
  let lastErrors = null;

  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    onStatus(attempt === 0 ? "generating" : "retrying");
    const candidate = await generate({ prompt, schema, currentScene, validationError });
    if (!candidate) {
      throw categorized("no scene was produced; retry or refine the prompt", "ai-no-scene");
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
    throw categorized(`engine /validate returned unexpected status ${val.status}`, "engine-runtime");
  }

  if (!scene) {
    throw categorized(
      `could not produce a valid scene after ${maxRetries + 1} attempts`,
      "validation-giveup",
      { errors: lastErrors }
    );
  }

  return runScene(scene, { fetch, engineUrl, onStatus });
}

// Run a scene the client already holds (freshly generated, or loaded from a
// file). Validates first for a clean error, then runs. A run-time 400 (e.g. an
// element needing an absent engine) is surfaced as-is, not retried.
async function runScene(scene, { fetch, engineUrl, onStatus = () => {} }) {
  onStatus("validating");
  const val = await postJson(fetch, `${engineUrl}/validate`, { scene });
  if (val.status === 422) {
    throw categorized("the scene is invalid", "validation-giveup", {
      errors: val.body && val.body.errors,
    });
  }
  if (!(val.status === 200 && val.body && val.body.ok)) {
    throw categorized(`engine /validate returned unexpected status ${val.status}`, "engine-runtime");
  }

  onStatus("running");
  const run = await postJson(fetch, `${engineUrl}/run`, { scene });
  if (run.status === 400) {
    throw categorized((run.body && run.body.detail) || "engine run failed", "engine-runtime", {
      errorType: run.body && run.body.error_type,
    });
  }
  if (!run.ok) {
    throw categorized(`engine /run failed (status ${run.status})`, "engine-runtime");
  }

  onStatus("done");
  return { scene, ...run.body };
}

module.exports = { generateAndRun, runScene, postJson, fetchSchema };
