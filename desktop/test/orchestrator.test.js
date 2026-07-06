"use strict";
// node --test: the prompt loop with an injected stdio `engine` (engine.request)
// and an injected `generate` (the BYOK AI step). No process, no network, no key.
const test = require("node:test");
const assert = require("node:assert");

const { generateAndRun, runScene } = require("../src/orchestrator");
const { categorized } = require("../src/errors");

// fake engine: routes keyed by op; a route is a response object or (msg)->resp
function fakeEngine(routes) {
  const calls = [];
  return {
    calls,
    request: async (msg) => {
      calls.push(msg);
      const h = routes[msg.op];
      if (h === undefined) throw new Error(`no route for op ${msg.op}`);
      return typeof h === "function" ? h(msg) : h;
    },
  };
}

function recorder(scenes) {
  const calls = [];
  let i = 0;
  return {
    calls,
    generate: async (args) => {
      calls.push(args);
      const s = scenes[Math.min(i, scenes.length - 1)];
      i += 1;
      return s;
    },
  };
}

const RUN_OK = { ok: true, result: { report: "Latent Dirac scene report", html: "<html>3d</html>", accepted: 12.5, losses: { iris: 3, surviving: 12.5 } } };

test("runs the full loop and returns report + html + summary", async () => {
  const engine = fakeEngine({ schema: { ok: true, result: {} }, validate: { ok: true }, run: RUN_OK });
  const { generate } = recorder([{ name: "ok" }]);
  const out = await generateAndRun({ prompt: "a pair" }, { engine, generate, maxRetries: 2 });
  assert.equal(out.report, "Latent Dirac scene report");
  assert.equal(out.html, "<html>3d</html>");
  assert.equal(out.accepted, 12.5);
  assert.deepEqual(out.scene, { name: "ok" });
});

test("fetches the schema once and forwards it to generate", async () => {
  const engine = fakeEngine({ schema: { ok: true, result: { properties: { source: {} } } }, validate: { ok: true }, run: RUN_OK });
  const rec = recorder([{ name: "ok" }]);
  await generateAndRun({ prompt: "p" }, { engine, generate: rec.generate });
  assert.equal(engine.calls.filter((c) => c.op === "schema").length, 1);
  assert.deepEqual(rec.calls[0].schema, { properties: { source: {} } });
});

test("fetches source_params and forwards them to generate", async () => {
  const engine = fakeEngine({
    schema: { ok: true, result: {} },
    source_params: { ok: true, result: { positron_pair: { required: ["primary_count"] } } },
    validate: { ok: true },
    run: RUN_OK,
  });
  const rec = recorder([{ name: "ok" }]);
  await generateAndRun({ prompt: "p" }, { engine, generate: rec.generate });
  assert.deepEqual(rec.calls[0].sourceParams, { positron_pair: { required: ["primary_count"] } });
});

test("forwards currentScene to generate", async () => {
  const engine = fakeEngine({ schema: { ok: true, result: {} }, validate: { ok: true }, run: RUN_OK });
  const rec = recorder([{ name: "edited" }]);
  await generateAndRun({ prompt: "longer", currentScene: { name: "existing" } }, { engine, generate: rec.generate });
  assert.deepEqual(rec.calls[0].currentScene, { name: "existing" });
});

test("feeds a validation error back into the next generate call and retries", async () => {
  let validateN = 0;
  const engine = fakeEngine({
    schema: { ok: true, result: {} },
    validate: () => {
      validateN += 1;
      return validateN === 1
        ? { ok: false, error: { type: "validation", errors: [{ msg: "unknown" }] } }
        : { ok: true };
    },
    run: RUN_OK,
  });
  const rec = recorder([{ name: "bad" }, { name: "good" }]);
  const statuses = [];
  const out = await generateAndRun({ prompt: "p" }, { engine, generate: rec.generate, onStatus: (s) => statuses.push(s) });
  assert.equal(out.scene.name, "good");
  assert.equal(rec.calls.length, 2);
  assert.ok(rec.calls[1].validationError, "second generate call carries the validation error");
  assert.ok(statuses.includes("retrying"));
});

test("gives up with a category after maxRetries", async () => {
  const engine = fakeEngine({
    schema: { ok: true, result: {} },
    validate: { ok: false, error: { type: "validation", errors: [{ msg: "nope" }] } },
  });
  const rec = recorder([{ name: "bad" }]);
  await assert.rejects(
    generateAndRun({ prompt: "p" }, { engine, generate: rec.generate, maxRetries: 1 }),
    (e) => { assert.equal(e.category, "validation-giveup"); return true; }
  );
});

test("surfaces a run-time error as engine-runtime, no retry", async () => {
  const engine = fakeEngine({
    schema: { ok: true, result: {} },
    validate: { ok: true },
    run: { ok: false, error: { type: "runtime", detail: "set LATENT_DIRAC_G4_TRANSFORMER", error_type: "RuntimeError" } },
  });
  const rec = recorder([{ name: "ok" }]);
  await assert.rejects(
    generateAndRun({ prompt: "a slab" }, { engine, generate: rec.generate }),
    (e) => { assert.equal(e.category, "engine-runtime"); assert.match(e.message, /LATENT_DIRAC_G4_TRANSFORMER/); return true; }
  );
  assert.equal(rec.calls.length, 1);
});

test("propagates a generate error unrelabeled (BYOK key/network failures)", async () => {
  const engine = fakeEngine({ schema: { ok: true, result: {} } });
  const generate = async () => { throw categorized("the API key was rejected", "ai-bad-key"); };
  await assert.rejects(
    generateAndRun({ prompt: "p" }, { engine, generate }),
    (e) => { assert.equal(e.category, "ai-bad-key"); return true; }
  );
});

test("categorizes an unreachable engine (request rejects)", async () => {
  const engine = { request: async () => { throw new Error("engine exited (code 1)"); } };
  const rec = recorder([{ name: "ok" }]);
  await assert.rejects(
    generateAndRun({ prompt: "p" }, { engine, generate: rec.generate }),
    (e) => { assert.equal(e.category, "engine-unreachable"); return true; }
  );
});

test("runScene runs a loaded scene directly (no schema, no generate)", async () => {
  const engine = fakeEngine({ validate: { ok: true }, run: RUN_OK });
  const out = await runScene({ name: "loaded" }, { engine });
  assert.equal(out.report, "Latent Dirac scene report");
  assert.deepEqual(out.scene, { name: "loaded" });
  assert.equal(engine.calls.filter((c) => c.op === "schema").length, 0);
});

test("runScene rejects an invalid loaded scene with a category", async () => {
  const engine = fakeEngine({ validate: { ok: false, error: { type: "validation", errors: [{ msg: "bad" }] } } });
  await assert.rejects(runScene({ name: "bad" }, { engine }), (e) => {
    assert.equal(e.category, "validation-giveup");
    return true;
  });
});
