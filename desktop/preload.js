"use strict";
// Bridge between the sandboxed renderer and the main process. Exposes only two
// narrow calls; no Node, no engine URL, no secrets reach the page.

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("api", {
  // returns { ok: true, result } or { ok: false, error }
  runPrompt: (prompt, currentScene = null) =>
    ipcRenderer.invoke("run-prompt", { prompt, currentScene }),
  // subscribe to progress stages ("generating" | "validating" | "retrying" |
  // "running" | "done"); returns an unsubscribe function
  onStatus: (callback) => {
    const listener = (_event, stage) => callback(stage);
    ipcRenderer.on("status", listener);
    return () => ipcRenderer.removeListener("status", listener);
  },
});
