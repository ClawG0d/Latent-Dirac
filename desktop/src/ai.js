"use strict";
// BYOK Anthropic client: turn a prompt + the engine's Scene JSON Schema into a
// candidate scene via a forced tool call, using the user's own API key.
//
// The key is supplied by the caller (the main process, which holds it) — it is
// never hardcoded and never reaches the renderer. fetch is injected so this is
// unit-tested with no network and no key. The request shaping mirrors the older
// hosted gateway, so the honesty discipline travels with it.

const { categorized } = require("./errors");

const DEFAULT_MODEL = "claude-sonnet-5";
const TOOL_NAME = "emit_scene";
const ANTHROPIC_URL = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION = "2023-06-01";

const SYSTEM_PROMPT = [
  "You design Latent Dirac simulation scenes for an antimatter-factory ",
  "simulator. Given a natural-language request, emit ONE scene object via ",
  "the emit_scene tool, matching the provided JSON schema exactly. Rules:\n",
  "- Use only source/element/solver types allowed by the schema; invalid ",
  "keys or types are rejected by fail-fast validation.\n",
  "- Honesty discipline: every model has a fidelity tier; do not claim more ",
  "than the parameterized/surrogate/table-based models provide, and prefer ",
  "always-available elements. Elements that need an external engine ",
  "(matter_slab needs a Geant4 build; xsuite_lattice needs xtrack) should ",
  "only be used when the request clearly calls for them.\n",
  "- Give every element and the source a unique label; labels become the ",
  "loss-ledger stage names.\n",
  "- If a prior validation error is provided, fix exactly what it reports.",
].join("");

function buildMessages({ prompt, currentScene, validationError, sourceParams }) {
  const parts = [`Request: ${prompt}`];
  if (sourceParams) {
    // the scene schema types source.params as an open dict; this is the real
    // per-type param contract the source.params object must match
    parts.push(
      "The `params` object of the chosen source `type` MUST match that type's " +
        `JSON schema here: ${JSON.stringify(sourceParams)}`
    );
  }
  if (currentScene) {
    parts.push(`Current scene (edit this): ${JSON.stringify(currentScene)}`);
  }
  if (validationError) {
    parts.push(
      "The previous attempt failed validation with these errors; correct " +
        `them: ${JSON.stringify(validationError)}`
    );
  }
  return [{ role: "user", content: parts.join("\n\n") }];
}

function buildRequest({ prompt, schema, sourceParams = null, currentScene = null, validationError = null, model }) {
  return {
    model: model || DEFAULT_MODEL,
    max_tokens: 4096,
    system: SYSTEM_PROMPT,
    tools: [
      { name: TOOL_NAME, description: "Emit one Latent Dirac scene matching the schema.", input_schema: schema },
    ],
    tool_choice: { type: "tool", name: TOOL_NAME },
    messages: buildMessages({ prompt, currentScene, validationError, sourceParams }),
  };
}

function extractScene(body) {
  const content = (body && body.content) || [];
  for (const block of content) {
    if (block && block.type === "tool_use" && block.name === TOOL_NAME) {
      return block.input;
    }
  }
  throw categorized("the model returned no scene; retry or refine the prompt", "ai-no-scene");
}

async function generateScene({ apiKey, prompt, schema, sourceParams = null, currentScene = null, validationError = null, model, fetch }) {
  if (!apiKey) {
    throw categorized("no Anthropic API key set — add one in Settings", "ai-no-key");
  }
  const request = buildRequest({ prompt, schema, sourceParams, currentScene, validationError, model });

  let resp;
  try {
    resp = await fetch(ANTHROPIC_URL, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": ANTHROPIC_VERSION,
      },
      body: JSON.stringify(request),
    });
  } catch (err) {
    throw categorized(`cannot reach the Anthropic API: ${err.message}`, "ai-unreachable");
  }

  let body = null;
  try {
    body = await resp.json();
  } catch {
    body = null;
  }

  if (resp.status === 401 || resp.status === 403) {
    throw categorized("the API key was rejected — check it in Settings", "ai-bad-key");
  }
  if (!resp.ok) {
    const detail = body && body.error && body.error.message;
    throw categorized(`Anthropic API error (${resp.status})${detail ? ": " + detail : ""}`, "ai-error");
  }
  return extractScene(body);
}

module.exports = {
  buildRequest,
  buildMessages,
  extractScene,
  generateScene,
  SYSTEM_PROMPT,
  DEFAULT_MODEL,
  TOOL_NAME,
  ANTHROPIC_URL,
};
