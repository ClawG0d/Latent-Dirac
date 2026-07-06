"use strict";
// Electron main process. Thin wiring only: manage the local Python sim-engine
// sidecar and the window, and route the renderer's run-prompt IPC through the
// orchestrator. All load-bearing logic lives in src/ (unit-tested); this file
// needs a display and is verified on the owner's machine, not in CI here.

const path = require("node:path");
const { spawn } = require("node:child_process");

const { app, BrowserWindow, ipcMain, dialog } = require("electron");

const { startSidecar } = require("./src/sidecar");
const { generateAndRun } = require("./src/orchestrator");
const { loadConfig, engineSpawnSpec } = require("./src/config");

const config = loadConfig();
let mainWindow = null;
let sidecar = null;

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
  // in dev, `python -m latent_dirac.server` (see src/config.js)
  const spec = engineSpawnSpec(process.env, {
    isPackaged: app.isPackaged,
    resourcesPath: process.resourcesPath,
    platform: process.platform,
  });
  sidecar = await startSidecar({ spawn, command: spec.command, args: spec.args });
  const resp = await fetch(`${sidecar.baseUrl}/health`);
  if (!resp.ok) {
    throw new Error(`engine health check failed (status ${resp.status})`);
  }
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

ipcMain.handle("run-prompt", async (_event, { prompt, currentScene }) => {
  if (!sidecar) {
    return { ok: false, error: "the sim engine is not running" };
  }
  try {
    const result = await generateAndRun(
      { prompt, currentScene },
      {
        fetch,
        gatewayUrl: config.gatewayUrl,
        engineUrl: sidecar.baseUrl,
        maxRetries: config.maxRetries,
        onStatus: sendStatus,
      }
    );
    return { ok: true, result };
  } catch (err) {
    return { ok: false, error: err && err.message ? err.message : String(err) };
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
