"use strict";
// node --test: BYOK Anthropic call shaping + extraction. Pure/injected fetch —
// no network, no key. The key lives in the main process at runtime; this
// module just shapes the request and reads the tool_use back.
const test = require("node:test");
const assert = require("node:assert");

const { buildRequest, extractScene, generateScene, SYSTEM_PROMPT, DEFAULT_MODEL } = require("../src/ai");

const SCHEMA = { type: "object", properties: { source: {} } };

test("buildRequest forces the emit_scene tool with the engine schema", () => {
  const r = buildRequest({ prompt: "a pair", schema: SCHEMA });
  assert.equal(r.tools[0].name, "emit_scene");
  assert.deepEqual(r.tools[0].input_schema, SCHEMA);
  assert.deepEqual(r.tool_choice, { type: "tool", name: "emit_scene" });
  assert.equal(r.model, DEFAULT_MODEL);
  assert.ok(r.system && r.system.length > 0);
  assert.ok(JSON.stringify(r.messages).includes("a pair"));
});

test("buildRequest includes current scene and validation error when given", () => {
  const r = buildRequest({
    prompt: "x", schema: SCHEMA, currentScene: { name: "cur" }, validationError: [{ msg: "bad" }],
  });
  const s = JSON.stringify(r.messages);
  assert.ok(s.includes("cur"), "current scene forwarded");
  assert.ok(s.toLowerCase().includes("bad"), "validation error forwarded");
});

test("buildRequest includes the source param schemas when provided", () => {
  const r = buildRequest({
    prompt: "x", schema: SCHEMA,
    sourceParams: { positron_pair: { required: ["primary_count", "mean_energy_MeV"] } },
  });
  const s = JSON.stringify(r.messages);
  assert.ok(s.includes("primary_count"), "forwards the source param contract");
  assert.ok(s.includes("params"), "tells the model params must match");
});

test("buildRequest honours a model override", () => {
  const r = buildRequest({ prompt: "x", schema: SCHEMA, model: "claude-opus-4-8" });
  assert.equal(r.model, "claude-opus-4-8");
});

test("SYSTEM_PROMPT carries the honesty discipline", () => {
  assert.ok(SYSTEM_PROMPT.toLowerCase().includes("fidelity"));
});

test("extractScene returns the tool_use input", () => {
  const scene = { name: "ok" };
  const body = { content: [{ type: "text", text: "hi" }, { type: "tool_use", name: "emit_scene", input: scene }] };
  assert.deepEqual(extractScene(body), scene);
});

test("extractScene throws ai-no-scene when there is no tool_use block", () => {
  assert.throws(() => extractScene({ content: [{ type: "text", text: "no" }] }), (e) => {
    assert.equal(e.category, "ai-no-scene");
    return true;
  });
});

test("generateScene refuses without an API key", async () => {
  await assert.rejects(
    generateScene({ apiKey: "", prompt: "x", schema: SCHEMA, fetch: async () => ({}) }),
    (e) => { assert.equal(e.category, "ai-no-key"); return true; }
  );
});

test("generateScene calls Anthropic with the key and returns the scene", async () => {
  const scene = { name: "ok" };
  let seen = null;
  const fetch = async (url, opts) => {
    seen = { url, opts };
    return { ok: true, status: 200, json: async () => ({ content: [{ type: "tool_use", name: "emit_scene", input: scene }] }) };
  };
  const out = await generateScene({ apiKey: "sk-test", prompt: "a pair", schema: SCHEMA, fetch });
  assert.deepEqual(out, scene);
  assert.match(seen.url, /api\.anthropic\.com/);
  assert.equal(seen.opts.headers["x-api-key"], "sk-test");
  assert.ok(seen.opts.headers["anthropic-version"], "sends anthropic-version");
});

test("generateScene maps 401 to ai-bad-key", async () => {
  const fetch = async () => ({ ok: false, status: 401, json: async () => ({ error: { message: "invalid" } }) });
  await assert.rejects(generateScene({ apiKey: "bad", prompt: "x", schema: SCHEMA, fetch }), (e) => {
    assert.equal(e.category, "ai-bad-key");
    return true;
  });
});

test("generateScene maps a network throw to ai-unreachable", async () => {
  const fetch = async () => { throw new TypeError("fetch failed"); };
  await assert.rejects(generateScene({ apiKey: "x", prompt: "x", schema: SCHEMA, fetch }), (e) => {
    assert.equal(e.category, "ai-unreachable");
    return true;
  });
});

test("generateScene maps other HTTP errors to ai-error", async () => {
  const fetch = async () => ({ ok: false, status: 529, json: async () => ({ error: { message: "overloaded" } }) });
  await assert.rejects(generateScene({ apiKey: "x", prompt: "x", schema: SCHEMA, fetch }), (e) => {
    assert.equal(e.category, "ai-error");
    return true;
  });
});
