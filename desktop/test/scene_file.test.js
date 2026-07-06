"use strict";
// node --test: scene file (de)serialization used by Save/Load. Pure functions,
// no fs/dialog — those live in main.js and are verified when the app runs.
const test = require("node:test");
const assert = require("node:assert");

const { serializeScene, parseSceneFile } = require("../src/scene_file");

test("serializeScene round-trips through parseSceneFile", () => {
  const scene = { name: "demo", solver: { dt_s: 4e-12, steps: 40 }, elements: [{ type: "monitor" }] };
  const parsed = parseSceneFile(serializeScene(scene));
  assert.deepEqual(parsed, scene);
});

test("serializeScene pretty-prints and ends with a newline", () => {
  const text = serializeScene({ a: 1 });
  assert.ok(text.endsWith("\n"), "ends with newline");
  assert.ok(text.includes("\n  "), "indented");
});

test("parseSceneFile rejects invalid JSON", () => {
  assert.throws(() => parseSceneFile("{not json"), /not valid JSON/);
});

test("parseSceneFile rejects a non-object (array)", () => {
  assert.throws(() => parseSceneFile("[1, 2, 3]"), /must be a JSON object/);
});

test("parseSceneFile rejects a non-object (null)", () => {
  assert.throws(() => parseSceneFile("null"), /must be a JSON object/);
});
