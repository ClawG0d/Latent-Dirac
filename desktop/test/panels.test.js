"use strict";
// node --test: pure data transforms that feed the dashboard panels from a
// /run result and the scene. No DOM — the renderer formats these.
const test = require("node:test");
const assert = require("node:assert");

const { physicsSummary, ledgerRows, sceneElements, numericParams, setParam } = require("../renderer/panels");

// realistic /run shape: loss_ledger() adds a reserved "surviving" entry (==accepted)
const RESULT = {
  scene: {
    source: { type: "positron_pair", label: "pairs" },
    elements: [
      { type: "solenoid", label: "cap", b_tesla: 0.6, radius_m: 0.03, length_m: 0.2 },
      { type: "aperture", label: "iris", radius_m: 0.01 },
      { type: "monitor", label: "end" },
    ],
  },
  accepted: 1203,
  losses: { cap: 12, iris: 202, surviving: 1203 },
};

test("physicsSummary excludes the reserved surviving key from lost/stages", () => {
  const s = physicsSummary(RESULT);
  assert.equal(s.accepted, 1203);
  assert.equal(s.lost, 214); // 12 + 202, NOT + surviving 1203
  assert.equal(s.stages, 2); // cap, iris — not surviving
  assert.ok(Math.abs(s.transmissionPct - (100 * 1203) / 1417) < 1e-6);
});

test("physicsSummary is safe on an empty/あいまい result", () => {
  const s = physicsSummary({});
  assert.equal(s.accepted, 0);
  assert.equal(s.lost, 0);
  assert.equal(s.transmissionPct, 0);
  assert.equal(s.stages, 0);
});

test("ledgerRows lists per-stage losses and drops surviving", () => {
  const rows = ledgerRows(RESULT.losses);
  assert.deepEqual(rows, [{ stage: "cap", lost: 12 }, { stage: "iris", lost: 202 }]);
});

test("sceneElements lists the source then each element with a summary", () => {
  const els = sceneElements(RESULT.scene);
  assert.equal(els[0].kind, "source");
  assert.equal(els[0].type, "positron_pair");
  assert.equal(els[1].type, "solenoid");
  assert.equal(els[1].label, "cap");
  assert.ok(els[1].summary.includes("b_tesla"), "summary shows params");
  assert.equal(els.length, 4);
});

test("numericParams collects tweakable numeric element params", () => {
  const ps = numericParams(RESULT.scene);
  const solB = ps.find((p) => p.key === "b_tesla");
  assert.ok(solB, "finds solenoid b_tesla");
  assert.equal(solB.value, 0.6);
  assert.deepEqual(solB.path, [0, "b_tesla"]);
  // aperture radius_m is numeric too
  assert.ok(ps.some((p) => p.path[0] === 1 && p.key === "radius_m"));
});

test("setParam returns a deep clone with the param changed (original untouched)", () => {
  const next = setParam(RESULT.scene, [0, "b_tesla"], 0.9);
  assert.equal(next.elements[0].b_tesla, 0.9);
  assert.equal(RESULT.scene.elements[0].b_tesla, 0.6, "original scene unchanged");
});
