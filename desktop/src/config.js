"use strict";
// Runtime configuration. No secrets here — the Anthropic key lives on the
// hosted gateway, never in the client. The gateway URL is owner-deployed; the
// default is a local placeholder until then.

const DEFAULT_GATEWAY_URL = "http://127.0.0.1:8080";

function loadConfig(env = process.env) {
  const retries = Number(env.LATENT_DIRAC_MAX_RETRIES);
  return {
    gatewayUrl: env.LATENT_DIRAC_GATEWAY_URL || DEFAULT_GATEWAY_URL,
    maxRetries: Number.isFinite(retries) ? retries : 2,
  };
}

// How to launch the local sim engine. In development it is the installed
// package (`python -m latent_dirac.server`); in a packaged build (Phase E) it
// is the frozen PyInstaller binary, pointed to by LATENT_DIRAC_ENGINE_CMD. The
// server binds an ephemeral port and prints "PORT <n>" on stdout either way.
function engineSpawnSpec(env = process.env) {
  const portArgs = ["--host", "127.0.0.1", "--port", "0"];
  const frozen = env.LATENT_DIRAC_ENGINE_CMD;
  if (frozen) {
    return { command: frozen, args: portArgs };
  }
  const python = env.LATENT_DIRAC_PYTHON || "python";
  return { command: python, args: ["-m", "latent_dirac.server", ...portArgs] };
}

module.exports = { loadConfig, engineSpawnSpec, DEFAULT_GATEWAY_URL };
