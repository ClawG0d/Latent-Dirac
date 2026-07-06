"use strict";
// Runtime configuration. No secrets here — the Anthropic key lives on the
// hosted gateway, never in the client. The gateway URL is owner-deployed; the
// default is a local placeholder until then.

const path = require("node:path");

const DEFAULT_GATEWAY_URL = "http://127.0.0.1:8080";

function loadConfig(env = process.env) {
  const retries = Number(env.LATENT_DIRAC_MAX_RETRIES);
  return {
    gatewayUrl: env.LATENT_DIRAC_GATEWAY_URL || DEFAULT_GATEWAY_URL,
    maxRetries: Number.isFinite(retries) ? retries : 2,
  };
}

// How to launch the local sim engine. The server binds an ephemeral port and
// prints "PORT <n>" on stdout in every case. Resolution order:
//   1. LATENT_DIRAC_ENGINE_CMD  — explicit override (always wins)
//   2. packaged build           — the frozen PyInstaller binary bundled by
//      electron-builder as an extraResource under resources/engine/
//   3. development              — the installed package (`python -m ...`)
// opts (from main.js): { isPackaged, resourcesPath, platform }.
function engineSpawnSpec(env = process.env, opts = {}) {
  const portArgs = ["--host", "127.0.0.1", "--port", "0"];

  const explicit = env.LATENT_DIRAC_ENGINE_CMD;
  if (explicit) {
    return { command: explicit, args: portArgs };
  }

  if (opts.isPackaged && opts.resourcesPath) {
    const platform = opts.platform || process.platform;
    const exe = platform === "win32" ? "latent-dirac-engine.exe" : "latent-dirac-engine";
    // matches the electron-builder extraResources mapping in package.json:
    // packaging/dist/latent-dirac-engine -> resources/engine/latent-dirac-engine
    const command = path.join(opts.resourcesPath, "engine", "latent-dirac-engine", exe);
    return { command, args: portArgs };
  }

  const python = env.LATENT_DIRAC_PYTHON || "python";
  return { command: python, args: ["-m", "latent_dirac.server", ...portArgs] };
}

module.exports = { loadConfig, engineSpawnSpec, DEFAULT_GATEWAY_URL };
