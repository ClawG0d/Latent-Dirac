"use strict";
// The prompt loop: natural language -> a candidate scene (via the injected
// `generate`, the BYOK Anthropic client) -> local engine validate (with a
// bounded corrective retry) -> local engine run -> report + offline 3D HTML.
//
// `engine` is the stdio sidecar handle (`engine.request({op, ...})` -> response
// object) and `generate` are injected, so this is unit-tested with no process,
// no network, no key. Engine op contract (latent_dirac/bridge):
//   {op:"schema"}            -> {ok:true, result:<schema>}
//   {op:"validate", scene}   -> {ok:true} | {ok:false, error:{type:"validation", errors}}
//   {op:"run", scene}        -> {ok:true, result:{report,html,accepted,losses}}
//                               | {ok:false, error:{type:"validation"|"runtime", ...}}

const { categorized } = require("./errors");

async function schemaOf(engine) {
  let resp;
  try {
    resp = await engine.request({ op: "schema" });
  } catch (err) {
    throw categorized(`cannot reach the local sim engine: ${err.message}`, "engine-unreachable");
  }
  if (!resp.ok) {
    throw categorized("engine could not return its schema", "engine-runtime");
  }
  return resp.result;
}

// Per-source-type param schemas — the scene schema types source.params as an
// open dict, so this is what tells the AI a source's required params. Best
// effort: if unavailable, the validate retry loop still self-corrects.
async function sourceParamsOf(engine) {
  try {
    const resp = await engine.request({ op: "source_params" });
    return resp && resp.ok ? resp.result : null;
  } catch {
    return null;
  }
}

// run a validated scene and normalize the response / errors
async function runValidated(engine, scene, onStatus) {
  onStatus("running");
  let resp;
  try {
    resp = await engine.request({ op: "run", scene });
  } catch (err) {
    throw categorized(`cannot reach the local sim engine: ${err.message}`, "engine-unreachable");
  }
  if (resp.ok) {
    onStatus("done");
    return { scene, ...resp.result };
  }
  const error = resp.error || {};
  if (error.type === "validation") {
    throw categorized("the scene is invalid", "validation-giveup", { errors: error.errors });
  }
  throw categorized(error.detail || "engine run failed", "engine-runtime", { errorType: error.error_type });
}

async function generateAndRun(
  { prompt, currentScene = null },
  { engine, generate, maxRetries = 2, onStatus = () => {} }
) {
  const schema = await schemaOf(engine);
  const sourceParams = await sourceParamsOf(engine);

  let scene = null;
  let validationError = null;
  let lastErrors = null;

  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    onStatus(attempt === 0 ? "generating" : "retrying");
    // generate throws its own categorized error (ai-no-key, ai-bad-key, ...)
    const candidate = await generate({ prompt, schema, sourceParams, currentScene, validationError });
    if (!candidate) {
      throw categorized("no scene was produced; retry or refine the prompt", "ai-no-scene");
    }

    onStatus("validating");
    let val;
    try {
      val = await engine.request({ op: "validate", scene: candidate });
    } catch (err) {
      throw categorized(`cannot reach the local sim engine: ${err.message}`, "engine-unreachable");
    }
    if (val.ok) {
      scene = candidate;
      break;
    }
    if (val.error && val.error.type === "validation") {
      lastErrors = val.error.errors || null;
      validationError = lastErrors;
      continue;
    }
    throw categorized("engine returned an unexpected validate response", "engine-runtime");
  }

  if (!scene) {
    throw categorized(
      `could not produce a valid scene after ${maxRetries + 1} attempts`,
      "validation-giveup",
      { errors: lastErrors }
    );
  }

  return runValidated(engine, scene, onStatus);
}

// Run a scene the client already holds (loaded from a file), skipping the AI.
// Validate first for a clean error, then run.
async function runScene(scene, { engine, onStatus = () => {} }) {
  onStatus("validating");
  let val;
  try {
    val = await engine.request({ op: "validate", scene });
  } catch (err) {
    throw categorized(`cannot reach the local sim engine: ${err.message}`, "engine-unreachable");
  }
  if (!val.ok) {
    if (val.error && val.error.type === "validation") {
      throw categorized("the scene is invalid", "validation-giveup", { errors: val.error.errors });
    }
    throw categorized("engine returned an unexpected validate response", "engine-runtime");
  }
  return runValidated(engine, scene, onStatus);
}

module.exports = { generateAndRun, runScene, schemaOf, sourceParamsOf };
