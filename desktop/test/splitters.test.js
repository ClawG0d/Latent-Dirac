"use strict";
// node --test: the pure drag-size math. The DOM wiring (makeSplitter/
// initSplitters) needs a browser and is verified by using the app.
const test = require("node:test");
const assert = require("node:assert");

const { resizeBasis } = require("../renderer/splitters");

test("resizeBasis applies the delta within range", () => {
  assert.equal(resizeBasis(400, 50, 160, 800), 450);
  assert.equal(resizeBasis(400, -120, 160, 800), 280);
});

test("resizeBasis clamps to the minimum (sibling can't be crushed)", () => {
  assert.equal(resizeBasis(200, -100, 160, 800), 160);
});

test("resizeBasis clamps to the maximum (sibling keeps its space)", () => {
  assert.equal(resizeBasis(700, 300, 160, 800), 800);
});

test("resizeBasis is safe when max < min (tiny window)", () => {
  // hi is floored to min, so the result is exactly min, never below
  assert.equal(resizeBasis(500, 0, 160, 100), 160);
});
