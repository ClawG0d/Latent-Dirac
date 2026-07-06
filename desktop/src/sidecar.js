"use strict";
// Engine sidecar over stdio JSON-RPC. Spawns the local Latent Dirac engine
// (`python -m latent_dirac.bridge` in dev, the frozen binary when packaged),
// waits for its {"ready":true} line, then exchanges one JSON request/response
// per line — no HTTP, no port. spawn is injected so this is unit-tested with a
// fake child process.

const { StringDecoder } = require("node:string_decoder");

function startSidecar({ spawn, command, args = [], readyTimeoutMs = 15000 }) {
  return new Promise((resolveReady, rejectReady) => {
    const child = spawn(command, args);
    // decode across chunk boundaries so a multi-byte char split between two
    // stdout chunks can never corrupt a line (which would hang that request)
    const decoder = new StringDecoder("utf8");
    let buffer = "";
    let ready = false;
    let nextId = 1;
    let deadError = null;
    const pending = new Map(); // id -> { resolve, reject }

    const timer = setTimeout(() => {
      if (!ready) {
        clearTimeout(timer);
        try {
          child.kill();
        } catch {
          /* already gone */
        }
        rejectReady(new Error(`engine did not report ready within ${readyTimeoutMs}ms`));
      }
    }, readyTimeoutMs);
    if (typeof timer.unref === "function") timer.unref();

    function rejectAllPending(err) {
      for (const { reject } of pending.values()) reject(err);
      pending.clear();
    }

    function handleLine(text) {
      let msg;
      try {
        msg = JSON.parse(text);
      } catch {
        return; // ignore non-JSON noise (e.g. an import warning before ready)
      }
      if (!ready) {
        if (msg && msg.ready) {
          ready = true;
          clearTimeout(timer);
          resolveReady({ request, stop, process: child });
        }
        return;
      }
      if (msg && msg.id != null && pending.has(msg.id)) {
        const { resolve } = pending.get(msg.id);
        pending.delete(msg.id);
        resolve(msg);
      }
    }

    child.stdout.on("data", (chunk) => {
      buffer += typeof chunk === "string" ? chunk : decoder.write(chunk);
      let nl;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const text = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (text) handleLine(text);
      }
    });

    child.on("error", (err) => {
      deadError = err;
      clearTimeout(timer);
      if (!ready) rejectReady(err);
      rejectAllPending(err);
    });

    child.on("exit", (code) => {
      const err = deadError || new Error(`engine exited (code ${code})`);
      clearTimeout(timer);
      if (!ready) rejectReady(new Error(`engine exited (code ${code}) before reporting ready`));
      rejectAllPending(err);
    });

    function request(msg) {
      return new Promise((resolve, reject) => {
        if (deadError) {
          reject(deadError);
          return;
        }
        const id = nextId++;
        pending.set(id, { resolve, reject });
        try {
          child.stdin.write(JSON.stringify({ id, ...msg }) + "\n");
        } catch (err) {
          pending.delete(id);
          reject(err);
        }
      });
    }

    function stop() {
      try {
        child.kill();
      } catch {
        /* already gone */
      }
    }
  });
}

module.exports = { startSidecar };
