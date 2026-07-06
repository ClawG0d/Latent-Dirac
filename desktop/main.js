"use strict";
// Electron main process. Thin wiring only: manage the local Python sim-engine
// sidecar and the window, and route the renderer's run-prompt IPC through the
// orchestrator. All load-bearing logic lives in src/ (unit-tested); this file
// needs a display and is verified on the owner's machine, not in CI here.

const path = require("node:path");
const fs = require("node:fs/promises");
const { spawn } = require("node:child_process");

const { app, BrowserWindow, ipcMain, dialog, safeStorage } = require("electron");

const { startSidecar } = require("./src/sidecar");
const { generateAndRun, runScene } = require("./src/orchestrator");
const { generateScene } = require("./src/ai");
const { serializeScene, parseSceneFile } = require("./src/scene_file");
const { loadConfig, engineSpawnSpec } = require("./src/config");

const config = loadConfig();
let mainWindow = null;
let sidecar = null;

// BYOK: the user's Anthropic key lives only here in the main process (never in
// the renderer). Persisted encrypted via the OS keychain (safeStorage) when
// available; held in memory for the session either way.
let apiKey = null;
const keyFile = () => path.join(app.getPath("userData"), "anthropic-key.bin");

async function loadKey() {
  try {
    const buf = await fs.readFile(keyFile());
    apiKey = safeStorage.isEncryptionAvailable() ? safeStorage.decryptString(buf) : buf.toString("utf8");
  } catch {
    apiKey = null;
  }
}

async function saveKey(key) {
  apiKey = key || null;
  try {
    if (!key) {
      await fs.rm(keyFile(), { force: true });
      return;
    }
    const data = safeStorage.isEncryptionAvailable()
      ? safeStorage.encryptString(key)
      : Buffer.from(key, "utf8");
    await fs.writeFile(keyFile(), data, { mode: 0o600 });
  } catch {
    // persistence failed (e.g. no keychain) — the key still works this session
  }
}

function stopEngine() {
  if (!sidecar) return;
  try {
    sidecar.stop();
  } catch {
    // the process may already be gone
  }
  sidecar = null;
}

async function startEngine() {
  // in a packaged build, launch the frozen engine bundled as an extraResource;
  // in dev, `python -m latent_dirac.bridge` (see src/config.js). startSidecar
  // resolves once the engine prints its {"ready":true} line.
  const spec = engineSpawnSpec(process.env, {
    isPackaged: app.isPackaged,
    resourcesPath: process.resourcesPath,
    platform: process.platform,
  });
  sidecar = await startSidecar({ spawn, command: spec.command, args: spec.args });
}

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    title: "Latent Dirac",
    backgroundColor: "#0b0f17",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  await mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));
}

function sendStatus(stage) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("status", stage);
  }
}

function failure(err) {
  return {
    ok: false,
    error: err && err.message ? err.message : String(err),
    category: err && err.category ? err.category : "unknown",
    // structured validation errors (from a give-up), so the UI can show why
    errors: (err && err.errors) || null,
  };
}

ipcMain.handle("run-prompt", async (_event, { prompt, currentScene }) => {
  if (!sidecar) {
    return { ok: false, error: "the sim engine is not running", category: "engine-unreachable" };
  }
  try {
    const generate = (a) => generateScene({ ...a, apiKey, model: config.model, fetch });
    const result = await generateAndRun(
      { prompt, currentScene },
      { engine: sidecar, generate, maxRetries: config.maxRetries, onStatus: sendStatus }
    );
    return { ok: true, result };
  } catch (err) {
    return failure(err);
  }
});

// BYOK key management — the renderer only ever learns whether a key is set,
// never the key itself.
ipcMain.handle("set-api-key", async (_event, key) => {
  await saveKey((key || "").trim());
  return { ok: true, hasKey: !!apiKey, encrypted: !!apiKey && safeStorage.isEncryptionAvailable() };
});
ipcMain.handle("clear-api-key", async () => {
  await saveKey(null);
  return { ok: true, hasKey: false };
});
ipcMain.handle("key-status", () => ({ hasKey: !!apiKey }));

// run a scene the renderer already holds (loaded from a file), no AI
ipcMain.handle("run-scene", async (_event, scene) => {
  if (!sidecar) {
    return { ok: false, error: "the sim engine is not running", category: "engine-unreachable" };
  }
  try {
    const result = await runScene(scene, { engine: sidecar, onStatus: sendStatus });
    return { ok: true, result };
  } catch (err) {
    return failure(err);
  }
});

ipcMain.handle("save-scene", async (_event, scene) => {
  if (!scene) return { ok: false, error: "no scene to save yet" };
  try {
    const { canceled, filePath } = await dialog.showSaveDialog(mainWindow, {
      title: "Save scene",
      defaultPath: "scene.json",
      filters: [{ name: "Scene JSON", extensions: ["json"] }],
    });
    if (canceled || !filePath) return { ok: false, canceled: true };
    await fs.writeFile(filePath, serializeScene(scene), "utf8");
    return { ok: true, filePath };
  } catch (err) {
    return failure(err);
  }
});

ipcMain.handle("open-scene", async () => {
  try {
    const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
      title: "Open scene",
      properties: ["openFile"],
      filters: [{ name: "Scene JSON", extensions: ["json"] }],
    });
    if (canceled || !filePaths || !filePaths.length) return { ok: false, canceled: true };
    const text = await fs.readFile(filePaths[0], "utf8");
    const scene = parseSceneFile(text);
    return { ok: true, scene, filePath: filePaths[0] };
  } catch (err) {
    return failure(err);
  }
});

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(async () => {
    await loadKey();
    try {
      await startEngine();
    } catch (err) {
      dialog.showErrorBox(
        "Latent Dirac engine failed to start",
        String(err && err.message ? err.message : err)
      );
      app.quit();
      return;
    }
    await createWindow();
  });

  app.on("window-all-closed", () => {
    stopEngine();
    app.quit();
  });

  app.on("before-quit", stopEngine);
}
