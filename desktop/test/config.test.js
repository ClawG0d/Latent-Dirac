"use strict";
const test = require("node:test");
const assert = require("node:assert");

const { loadConfig, engineSpawnSpec, DEFAULT_GATEWAY_URL } = require("../src/config");

test("defaults the gateway URL and retry count when env is empty", () => {
  const cfg = loadConfig({});
  assert.equal(cfg.gatewayUrl, DEFAULT_GATEWAY_URL);
  assert.equal(cfg.maxRetries, 2);
});

test("honours gateway URL and retry overrides from env", () => {
  const cfg = loadConfig({ LATENT_DIRAC_GATEWAY_URL: "https://ai.ora.io", LATENT_DIRAC_MAX_RETRIES: "4" });
  assert.equal(cfg.gatewayUrl, "https://ai.ora.io");
  assert.equal(cfg.maxRetries, 4);
});

test("engineSpawnSpec launches the installed package in development", () => {
  const spec = engineSpawnSpec({});
  assert.equal(spec.command, "python");
  assert.deepEqual(spec.args, ["-m", "latent_dirac.server", "--host", "127.0.0.1", "--port", "0"]);
});

test("engineSpawnSpec uses the frozen binary when LATENT_DIRAC_ENGINE_CMD is set", () => {
  const spec = engineSpawnSpec({ LATENT_DIRAC_ENGINE_CMD: "/opt/latent-dirac/engine" });
  assert.equal(spec.command, "/opt/latent-dirac/engine");
  assert.deepEqual(spec.args, ["--host", "127.0.0.1", "--port", "0"]);
});

test("engineSpawnSpec resolves the bundled engine in a packaged build (macOS)", () => {
  const spec = engineSpawnSpec(
    {},
    { isPackaged: true, resourcesPath: "/App.app/Contents/Resources", platform: "darwin" }
  );
  assert.ok(
    spec.command.endsWith("/engine/latent-dirac-engine/latent-dirac-engine"),
    `unexpected command: ${spec.command}`
  );
  assert.deepEqual(spec.args, ["--host", "127.0.0.1", "--port", "0"]);
});

test("engineSpawnSpec resolves the bundled .exe in a packaged build (Windows)", () => {
  const spec = engineSpawnSpec(
    {},
    { isPackaged: true, resourcesPath: "C:\\app\\resources", platform: "win32" }
  );
  assert.ok(spec.command.endsWith("latent-dirac-engine.exe"), `unexpected command: ${spec.command}`);
});

test("an explicit LATENT_DIRAC_ENGINE_CMD overrides even a packaged build", () => {
  const spec = engineSpawnSpec(
    { LATENT_DIRAC_ENGINE_CMD: "/custom/engine" },
    { isPackaged: true, resourcesPath: "/App/Resources", platform: "darwin" }
  );
  assert.equal(spec.command, "/custom/engine");
});
