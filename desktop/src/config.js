"use strict";
// Runtime configuration. BYOK: the Anthropic key is the user's, entered in the
// app and held only in the main process (see main.js) — it is never stored here
// and never reaches the renderer.

const path = require("node:path");

function loadConfig(env = process.env) {
  const retries = Number(env.LATENT_DIRAC_MAX_RETRIES);
  return {
    // null -> the BYOK client (src/ai.js) uses its DEFAULT_MODEL
    model: env.LATENT_DIRAC_MODEL || null,
    maxRetries: Number.isFinite(retries) ? retries : 3,
  };
}

// How to launch the local sim engine (the stdio JSON-RPC bridge — it prints
// {"ready":true} and then talks JSON over stdin/stdout; no port). Resolution:
//   1. LATENT_DIRAC_ENGINE_CMD  — explicit override (always wins)
//   2. packaged build           — the frozen PyInstaller binary bundled by
//      electron-builder as an extraResource under resources/engine/
//   3. development              — the installed package (`python -m latent_dirac.bridge`)
// opts (from main.js): { isPackaged, resourcesPath, platform }.
function engineSpawnSpec(env = process.env, opts = {}) {
  const explicit = env.LATENT_DIRAC_ENGINE_CMD;
  if (explicit) {
    return { command: explicit, args: [] };
  }

  if (opts.isPackaged && opts.resourcesPath) {
    const platform = opts.platform || process.platform;
    const exe = platform === "win32" ? "latent-dirac-engine.exe" : "latent-dirac-engine";
    // matches the electron-builder extraResources mapping in package.json:
    // packaging/dist/latent-dirac-engine -> resources/engine/latent-dirac-engine
    const command = path.join(opts.resourcesPath, "engine", "latent-dirac-engine", exe);
    return { command, args: [] };
  }

  const python = env.LATENT_DIRAC_PYTHON || "python";
  return { command: python, args: ["-m", "latent_dirac.bridge"] };
}

module.exports = { loadConfig, engineSpawnSpec };
