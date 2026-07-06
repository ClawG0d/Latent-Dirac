"use strict";
// Bridge between the sandboxed renderer and the main process. Exposes only two
// narrow calls; no Node, no engine URL, no secrets reach the page.

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("api", {
  // returns { ok: true, result } or { ok: false, error, category }
  runPrompt: (prompt, currentScene = null) =>
    ipcRenderer.invoke("run-prompt", { prompt, currentScene }),
  // run a scene the renderer already holds (loaded from a file), no gateway
  runScene: (scene) => ipcRenderer.invoke("run-scene", scene),
  // save the current scene to disk -> { ok, filePath } | { ok:false, ... }
  saveScene: (scene) => ipcRenderer.invoke("save-scene", scene),
  // open a scene file -> { ok, scene, filePath } | { ok:false, ... }
  openScene: () => ipcRenderer.invoke("open-scene"),
  // BYOK key: the renderer only sets/clears/queries; it never reads the key back
  setApiKey: (key) => ipcRenderer.invoke("set-api-key", key),
  clearApiKey: () => ipcRenderer.invoke("clear-api-key"),
  keyStatus: () => ipcRenderer.invoke("key-status"),
  // subscribe to progress stages ("generating" | "validating" | "retrying" |
  // "running" | "done"); returns an unsubscribe function
  onStatus: (callback) => {
    const listener = (_event, stage) => callback(stage);
    ipcRenderer.on("status", listener);
    return () => ipcRenderer.removeListener("status", listener);
  },
});
