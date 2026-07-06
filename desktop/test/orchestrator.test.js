"use strict";
// node --test: the prompt loop with an injected `generate` (the BYOK AI step)
// and an injected `fetch` (the local engine). No network, no key, no Python.
const test = require("node:test");
const assert = require("node:assert");

const { generateAndRun, runScene } = require("../src/orchestrator");
const { categorized } = require("../src/errors");

function json(status, body) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

// engine routes only ("METHOD /path"); values are responses or (url,opts)->resp
function fakeFetch(calls, routes) {
  return async (url, opts = {}) => {
    const method = (opts.method || "GET").toUpperCase();
    const path = new URL(url).pathname;
    const key = `${method} ${path}`;
    const handler = routes[key];
    if (!handler) throw new Error(`no fake route for ${key}`);
    const body = opts.body ? JSON.parse(opts.body) : undefined;
    calls.push({ method, path, url, body });
    return typeof handler === "function" ? handler(url, opts) : handler;
  };
}

// a generate() that records its calls and returns queued scenes
function recorder(scenes) {
  const calls = [];
  let i = 0;
  const generate = async (args) => {
    calls.push(args);
    const s = scenes[Math.min(i, scenes.length - 1)];
    i += 1;
    return s;
  };
  return { generate, calls };
}

const engineUrl = "http://127.0.0.1:9";

test("runs the full loop and returns report + html + summary", async () => {
  const calls = [];
  const scene = { name: "ok" };
  const fetch = fakeFetch(calls, {
    "GET /schema": json(200, { properties: {} }),
    "POST /validate": json(200, { ok: true }),
    "POST /run": json(200, { report: "Latent Dirac scene report", html: "<html>3d</html>", accepted: 12.5, losses: { iris: 3 } }),
  });
  const { generate } = recorder([scene]);
  const out = await generateAndRun({ prompt: "a pair" }, { fetch, engineUrl, generate, maxRetries: 2 });
  assert.equal(out.report, "Latent Dirac scene report");
  assert.equal(out.html, "<html>3d</html>");
  assert.equal(out.accepted, 12.5);
  assert.deepEqual(out.scene, scene);
});

test("fetches the engine schema once and forwards it to generate", async () => {
  const calls = [];
  const fetch = fakeFetch(calls, {
    "GET /schema": json(200, { properties: { source: {} } }),
    "POST /validate": json(200, { ok: true }),
    "POST /run": json(200, { report: "R", html: "H", accepted: 0, losses: {} }),
  });
  const rec = recorder([{ name: "ok" }]);
  await generateAndRun({ prompt: "p" }, { fetch, engineUrl, generate: rec.generate, maxRetries: 2 });
  assert.equal(calls.filter((c) => c.path === "/schema").length, 1);
  assert.deepEqual(rec.calls[0].schema, { properties: { source: {} } });
});

test("forwards currentScene to generate", async () => {
  const calls = [];
  const current = { name: "existing" };
  const fetch = fakeFetch(calls, {
    "GET /schema": json(200, {}),
    "POST /validate": json(200, { ok: true }),
    "POST /run": json(200, { report: "R", html: "H", accepted: 0, losses: {} }),
  });
  const rec = recorder([{ name: "edited" }]);
  await generateAndRun({ prompt: "longer", currentScene: current }, { fetch, engineUrl, generate: rec.generate });
  assert.deepEqual(rec.calls[0].currentScene, current);
});

test("feeds a validation error back into the next generate call and retries", async () => {
  const calls = [];
  let validateN = 0;
  const fetch = fakeFetch(calls, {
    "GET /schema": () => json(200, {}),
    "POST /validate": () => {
      validateN += 1;
      return validateN === 1
        ? json(422, { ok: false, errors: [{ loc: ["elements", 0], msg: "unknown" }] })
        : json(200, { ok: true });
    },
    "POST /run": () => json(200, { report: "R", html: "H", accepted: 1, losses: {} }),
  });
  const rec = recorder([{ name: "bad" }, { name: "good" }]);
  const statuses = [];
  const out = await generateAndRun({ prompt: "p" }, { fetch, engineUrl, generate: rec.generate, onStatus: (s) => statuses.push(s) });
  assert.equal(out.scene.name, "good");
  assert.equal(rec.calls.length, 2);
  assert.ok(rec.calls[1].validationError, "second generate call carries the validation error");
  assert.ok(statuses.includes("retrying"));
});

test("gives up with a category after maxRetries", async () => {
  const fetch = fakeFetch([], {
    "GET /schema": () => json(200, {}),
    "POST /validate": () => json(422, { ok: false, errors: [{ msg: "nope" }] }),
  });
  const rec = recorder([{ name: "bad" }]);
  await assert.rejects(
    generateAndRun({ prompt: "p" }, { fetch, engineUrl, generate: rec.generate, maxRetries: 1 }),
    (e) => { assert.equal(e.category, "validation-giveup"); return true; }
  );
});

test("surfaces a run-time 400 as engine-runtime, no retry", async () => {
  const fetch = fakeFetch([], {
    "GET /schema": () => json(200, {}),
    "POST /validate": () => json(200, { ok: true }),
    "POST /run": () => json(400, { detail: "set LATENT_DIRAC_G4_TRANSFORMER", error_type: "RuntimeError" }),
  });
  const rec = recorder([{ name: "ok" }]);
  await assert.rejects(
    generateAndRun({ prompt: "a slab" }, { fetch, engineUrl, generate: rec.generate }),
    (e) => { assert.equal(e.category, "engine-runtime"); assert.match(e.message, /LATENT_DIRAC_G4_TRANSFORMER/); return true; }
  );
  assert.equal(rec.calls.length, 1);
});

test("propagates a generate error unrelabeled (BYOK key/network failures)", async () => {
  const fetch = fakeFetch([], { "GET /schema": () => json(200, {}) });
  const generate = async () => { throw categorized("the API key was rejected", "ai-bad-key"); };
  await assert.rejects(
    generateAndRun({ prompt: "p" }, { fetch, engineUrl, generate }),
    (e) => { assert.equal(e.category, "ai-bad-key"); return true; }
  );
});

test("categorizes an unreachable engine (schema fetch throws)", async () => {
  const fetch = fakeFetch([], { "GET /schema": () => { throw new TypeError("connection refused"); } });
  const rec = recorder([{ name: "ok" }]);
  await assert.rejects(
    generateAndRun({ prompt: "p" }, { fetch, engineUrl, generate: rec.generate }),
    (e) => { assert.equal(e.category, "engine-unreachable"); return true; }
  );
});

test("runScene runs a loaded scene directly (no generate, no schema)", async () => {
  const calls = [];
  const scene = { name: "loaded" };
  const fetch = fakeFetch(calls, {
    "POST /validate": json(200, { ok: true }),
    "POST /run": json(200, { report: "R", html: "H", accepted: 5, losses: {} }),
  });
  const out = await runScene(scene, { fetch, engineUrl });
  assert.equal(out.report, "R");
  assert.deepEqual(out.scene, scene);
  assert.equal(calls.filter((c) => c.path === "/schema").length, 0);
});

test("runScene rejects an invalid loaded scene with a category", async () => {
  const fetch = fakeFetch([], { "POST /validate": json(422, { ok: false, errors: [{ msg: "bad" }] }) });
  await assert.rejects(runScene({ name: "bad" }, { fetch, engineUrl }), (e) => {
    assert.equal(e.category, "validation-giveup");
    return true;
  });
});
