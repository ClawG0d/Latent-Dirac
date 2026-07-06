"use strict";
// Engine-sidecar lifecycle. Spawns the local Latent Dirac sim engine and waits
// for the "PORT <n>" line it prints on stdout (see latent_dirac/server/
// __main__.py), then hands back the localhost base URL and a stop() to kill it.
//
// spawn is injected so this is unit-tested with a fake child process; in
// production main.js passes Node's child_process.spawn.

const PORT_LINE = /^PORT\s+(\d+)$/;

function startSidecar({ spawn, command, args = [], readyTimeoutMs = 15000 }) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args);
    let buffer = "";
    let settled = false;

    const finish = (fn) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      fn();
    };

    const timer = setTimeout(() => {
      finish(() => {
        try {
          child.kill();
        } catch {
          // best-effort; the process may already be gone
        }
        reject(new Error(`sidecar did not report a port within ${readyTimeoutMs}ms`));
      });
    }, readyTimeoutMs);
    if (typeof timer.unref === "function") timer.unref();

    child.stdout.on("data", (chunk) => {
      buffer += chunk.toString();
      let nl;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        const match = line.match(PORT_LINE);
        if (match) {
          const port = Number(match[1]);
          finish(() =>
            resolve({
              port,
              baseUrl: `http://127.0.0.1:${port}`,
              stop: () => child.kill(),
              process: child,
            })
          );
          return;
        }
      }
    });

    child.on("error", (err) => finish(() => reject(err)));
    child.on("exit", (code) =>
      finish(() => reject(new Error(`sidecar exited (code ${code}) before reporting a port`)))
    );
  });
}

module.exports = { startSidecar };
