"use strict";
// node --test: the stdio JSON-RPC sidecar, with an injected spawn so no real
// process runs. The engine prints {"ready":true}, then answers one JSON line
// per request, echoing the request id.
const test = require("node:test");
const assert = require("node:assert");
const { EventEmitter } = require("node:events");

const { startSidecar } = require("../src/sidecar");

function fakeChild() {
  const child = new EventEmitter();
  child.stdout = new EventEmitter();
  child.stdin = {
    written: [],
    write(s) {
      this.written.push(s);
      return true;
    },
  };
  child.killed = false;
  child.kill = () => {
    child.killed = true;
    return true;
  };
  return child;
}

function line(obj) {
  return Buffer.from(JSON.stringify(obj) + "\n");
}

async function ready(child, opts = {}) {
  const p = startSidecar({ spawn: () => child, command: "x", args: [], readyTimeoutMs: 1000, ...opts });
  child.stdout.emit("data", line({ ready: true }));
  return p;
}

test("resolves with an engine handle when the child prints ready", async () => {
  const child = fakeChild();
  const engine = await ready(child);
  assert.equal(typeof engine.request, "function");
  assert.equal(typeof engine.stop, "function");
});

test("ready line split across two chunks still resolves", async () => {
  const child = fakeChild();
  const p = startSidecar({ spawn: () => child, command: "x", args: [], readyTimeoutMs: 1000 });
  child.stdout.emit("data", Buffer.from('{"rea'));
  child.stdout.emit("data", Buffer.from('dy":true}\n'));
  const engine = await p;
  assert.equal(typeof engine.request, "function");
});

test("ignores non-JSON noise before the ready line", async () => {
  const child = fakeChild();
  const p = startSidecar({ spawn: () => child, command: "x", args: [], readyTimeoutMs: 1000 });
  child.stdout.emit("data", Buffer.from("a warning from some import\n"));
  child.stdout.emit("data", line({ ready: true }));
  await p; // resolves despite the noise
});

test("request writes a JSON line with an id and resolves the matching response", async () => {
  const child = fakeChild();
  const engine = await ready(child);
  const rp = engine.request({ op: "schema" });
  const sent = JSON.parse(child.stdin.written.at(-1));
  assert.equal(sent.op, "schema");
  assert.ok(Number.isInteger(sent.id));
  child.stdout.emit("data", line({ id: sent.id, ok: true, result: { properties: {} } }));
  const resp = await rp;
  assert.equal(resp.ok, true);
  assert.deepEqual(resp.result, { properties: {} });
});

test("correlates out-of-order responses by id", async () => {
  const child = fakeChild();
  const engine = await ready(child);
  const p1 = engine.request({ op: "a" });
  const p2 = engine.request({ op: "b" });
  const id1 = JSON.parse(child.stdin.written[0]).id;
  const id2 = JSON.parse(child.stdin.written[1]).id;
  // respond to the second request first
  child.stdout.emit("data", line({ id: id2, ok: true, result: "B" }));
  child.stdout.emit("data", line({ id: id1, ok: true, result: "A" }));
  assert.equal((await p1).result, "A");
  assert.equal((await p2).result, "B");
});

test("rejects when the child exits before ready", async () => {
  const child = fakeChild();
  const p = startSidecar({ spawn: () => child, command: "x", args: [], readyTimeoutMs: 1000 });
  child.emit("exit", 1);
  await assert.rejects(p, /before reporting ready/);
});

test("rejects on spawn error", async () => {
  const child = fakeChild();
  const p = startSidecar({ spawn: () => child, command: "x", args: [], readyTimeoutMs: 1000 });
  child.emit("error", new Error("ENOENT python"));
  await assert.rejects(p, /ENOENT python/);
});

test("rejects when no ready line arrives within the timeout", async () => {
  const child = fakeChild();
  const p = startSidecar({ spawn: () => child, command: "x", args: [], readyTimeoutMs: 20 });
  await assert.rejects(p, /within 20ms/);
});

test("pending requests reject if the engine dies mid-flight", async () => {
  const child = fakeChild();
  const engine = await ready(child);
  const rp = engine.request({ op: "run" });
  child.emit("exit", 1);
  await assert.rejects(rp, /exited/);
});

test("stop() kills the child", async () => {
  const child = fakeChild();
  const engine = await ready(child);
  engine.stop();
  assert.equal(child.killed, true);
});
