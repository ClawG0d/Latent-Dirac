"use strict";
// node --test: the engine-sidecar lifecycle logic, with an injected spawn so
// no real Python process is started. Verifies the PORT contract that
// latent_dirac/server/__main__.py prints on stdout.
const test = require("node:test");
const assert = require("node:assert");
const { EventEmitter } = require("node:events");

const { startSidecar } = require("../src/sidecar");

function fakeChild() {
  const child = new EventEmitter();
  child.stdout = new EventEmitter();
  child.killed = false;
  child.kill = () => {
    child.killed = true;
    return true;
  };
  return child;
}

test("resolves baseUrl when the sidecar prints PORT", async () => {
  const child = fakeChild();
  const p = startSidecar({ spawn: () => child, command: "x", args: [], readyTimeoutMs: 1000 });
  child.stdout.emit("data", Buffer.from("some startup log\nPORT 54889\n"));
  const s = await p;
  assert.equal(s.port, 54889);
  assert.equal(s.baseUrl, "http://127.0.0.1:54889");
});

test("handles a PORT line split across two stdout chunks", async () => {
  const child = fakeChild();
  const p = startSidecar({ spawn: () => child, command: "x", args: [], readyTimeoutMs: 1000 });
  child.stdout.emit("data", Buffer.from("PO"));
  child.stdout.emit("data", Buffer.from("RT 6100\n"));
  const s = await p;
  assert.equal(s.port, 6100);
});

test("rejects when the sidecar exits before printing a port", async () => {
  const child = fakeChild();
  const p = startSidecar({ spawn: () => child, command: "x", args: [], readyTimeoutMs: 1000 });
  child.emit("exit", 1);
  await assert.rejects(p, /before reporting a port/);
});

test("rejects on spawn error", async () => {
  const child = fakeChild();
  const p = startSidecar({ spawn: () => child, command: "x", args: [], readyTimeoutMs: 1000 });
  child.emit("error", new Error("ENOENT python"));
  await assert.rejects(p, /ENOENT python/);
});

test("rejects when no port arrives within the timeout", async () => {
  const child = fakeChild();
  const p = startSidecar({ spawn: () => child, command: "x", args: [], readyTimeoutMs: 20 });
  await assert.rejects(p, /within 20ms/);
});

test("stop() kills the child process", async () => {
  const child = fakeChild();
  const p = startSidecar({ spawn: () => child, command: "x", args: [], readyTimeoutMs: 1000 });
  child.stdout.emit("data", Buffer.from("PORT 1234\n"));
  const s = await p;
  s.stop();
  assert.equal(child.killed, true);
});
