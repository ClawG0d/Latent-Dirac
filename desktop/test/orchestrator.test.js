"use strict";
// node --test: the prompt -> gateway -> validate -> run loop, with an injected
// fetch so no network, no gateway, and no Python engine are needed. Locks the
// Phase A engine contract and the Phase D gateway contract.
const test = require("node:test");
const assert = require("node:assert");

const { generateAndRun } = require("../src/orchestrator");

function json(status, body) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

// records every call and dispatches on "METHOD /path"; a route value may be a
// response object or a function (url, opts) -> response (for stateful sequences)
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

const base = { gatewayUrl: "https://gw", engineUrl: "http://127.0.0.1:9", maxRetries: 2 };

test("runs the full loop and returns report + html + summary on a valid scene", async () => {
  const calls = [];
  const scene = { name: "ok" };
  const fetch = fakeFetch(calls, {
    "GET /schema": json(200, { properties: {} }),
    "POST /generate": json(200, { scene }),
    "POST /validate": json(200, { ok: true }),
    "POST /run": json(200, {
      report: "Latent Dirac scene report",
      html: "<html>3d</html>",
      accepted: 12.5,
      losses: { iris: 3 },
    }),
  });
  const out = await generateAndRun({ prompt: "a pair through a solenoid" }, { ...base, fetch });
  assert.equal(out.report, "Latent Dirac scene report");
  assert.equal(out.html, "<html>3d</html>");
  assert.equal(out.accepted, 12.5);
  assert.deepEqual(out.losses, { iris: 3 });
  assert.deepEqual(out.scene, scene);
});

test("fetches the engine schema once and forwards it to the gateway", async () => {
  const calls = [];
  const fetch = fakeFetch(calls, {
    "GET /schema": json(200, { properties: { source: {} } }),
    "POST /generate": json(200, { scene: { name: "ok" } }),
    "POST /validate": json(200, { ok: true }),
    "POST /run": json(200, { report: "R", html: "H", accepted: 0, losses: {} }),
  });
  await generateAndRun({ prompt: "p" }, { ...base, fetch });
  assert.equal(calls.filter((c) => c.path === "/schema").length, 1);
  const gen = calls.find((c) => c.path === "/generate");
  assert.deepEqual(gen.body.schema, { properties: { source: {} } });
});

test("forwards current_scene when editing an existing scene", async () => {
  const calls = [];
  const current = { name: "existing" };
  const fetch = fakeFetch(calls, {
    "GET /schema": json(200, {}),
    "POST /generate": json(200, { scene: { name: "edited" } }),
    "POST /validate": json(200, { ok: true }),
    "POST /run": json(200, { report: "R", html: "H", accepted: 0, losses: {} }),
  });
  await generateAndRun({ prompt: "make it longer", currentScene: current }, { ...base, fetch });
  const gen = calls.find((c) => c.path === "/generate");
  assert.deepEqual(gen.body.current_scene, current);
});

test("feeds a validation error back to the gateway and retries", async () => {
  const calls = [];
  let generateN = 0;
  let validateN = 0;
  const fetch = fakeFetch(calls, {
    "GET /schema": () => json(200, { properties: {} }),
    "POST /generate": () => {
      generateN += 1;
      return json(200, { scene: { name: generateN === 1 ? "bad" : "good" } });
    },
    "POST /validate": () => {
      validateN += 1;
      return validateN === 1
        ? json(422, { ok: false, errors: [{ loc: ["elements", 0, "type"], msg: "unknown" }] })
        : json(200, { ok: true });
    },
    "POST /run": () => json(200, { report: "R", html: "H", accepted: 1, losses: {} }),
  });
  const statuses = [];
  const out = await generateAndRun(
    { prompt: "p" },
    { ...base, fetch, onStatus: (s) => statuses.push(s) }
  );
  assert.equal(out.scene.name, "good");
  assert.equal(generateN, 2);
  const secondGen = calls.filter((c) => c.path === "/generate")[1];
  assert.ok(secondGen.body.validation_error, "second generate carries validation_error");
  assert.ok(statuses.includes("retrying"), "emits a retrying status");
});

test("gives up after maxRetries when the scene never validates", async () => {
  const fetch = fakeFetch([], {
    "GET /schema": () => json(200, {}),
    "POST /generate": () => json(200, { scene: { name: "bad" } }),
    "POST /validate": () => json(422, { ok: false, errors: [{ msg: "nope" }] }),
  });
  await assert.rejects(
    generateAndRun({ prompt: "p" }, { ...base, fetch, maxRetries: 1 }),
    /could not produce a valid scene/
  );
});

test("surfaces a run-time 400 from the engine without retrying", async () => {
  let generateN = 0;
  const fetch = fakeFetch([], {
    "GET /schema": () => json(200, {}),
    "POST /generate": () => {
      generateN += 1;
      return json(200, { scene: { name: "ok" } });
    },
    "POST /validate": () => json(200, { ok: true }),
    "POST /run": () =>
      json(400, { detail: "set LATENT_DIRAC_G4_TRANSFORMER", error_type: "RuntimeError" }),
  });
  await assert.rejects(
    generateAndRun({ prompt: "a slab" }, { ...base, fetch }),
    /LATENT_DIRAC_G4_TRANSFORMER/
  );
  assert.equal(generateN, 1, "a run-time failure is not retried");
});

test("throws when the gateway returns no scene", async () => {
  const fetch = fakeFetch([], {
    "GET /schema": () => json(200, {}),
    "POST /generate": () => json(200, {}),
  });
  await assert.rejects(generateAndRun({ prompt: "p" }, { ...base, fetch }), /no scene/);
});
