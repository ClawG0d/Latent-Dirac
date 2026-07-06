"use strict";
// Chat UI. Sends a prompt to the main process (which runs the gateway ->
// validate -> run loop), streams status stages, then shows the text report and
// loads the offline 3D HTML into the panel. window.api is defined by preload.js.

const form = document.getElementById("form");
const input = document.getElementById("prompt");
const send = document.getElementById("send");
const log = document.getElementById("log");
const statusEl = document.getElementById("status");
const frame = document.getElementById("frame");
const placeholder = document.getElementById("placeholder");

// the last valid scene, so a follow-up prompt can edit it rather than start over
let currentScene = null;

const STATUS_TEXT = {
  generating: "Generating a scene from your description…",
  validating: "Checking the scene against the engine schema…",
  retrying: "The scene needed a fix; asking again…",
  running: "Running the simulation locally…",
  done: "Done.",
};

function addMessage(kind, { meta, text } = {}) {
  const el = document.createElement("div");
  el.className = `msg msg--${kind}`;
  if (meta) {
    const m = document.createElement("div");
    m.className = "msg__meta";
    m.textContent = meta;
    el.appendChild(m);
  }
  if (text !== undefined) {
    const pre = document.createElement("pre");
    pre.textContent = text;
    el.appendChild(pre);
  }
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

function showStatus(stage) {
  const text = STATUS_TEXT[stage] || stage;
  statusEl.hidden = false;
  statusEl.innerHTML = "";
  const dot = document.createElement("span");
  dot.className = "status__dot";
  statusEl.appendChild(dot);
  statusEl.appendChild(document.createTextNode(text));
}

function clearStatus() {
  statusEl.hidden = true;
  statusEl.textContent = "";
}

function render3d(html) {
  // srcdoc (not a blob URL): a sandbox="allow-scripts" frame has an opaque
  // origin and cannot load a parent-origin blob, but inline srcdoc content
  // loads fine and its inline plotly scripts run, escaping the parent CSP.
  frame.srcdoc = html;
  frame.hidden = false;
  placeholder.hidden = true;
}

window.api.onStatus(showStatus);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = input.value.trim();
  if (!prompt) return;

  addMessage("user", { text: prompt });
  input.value = "";
  input.disabled = true;
  send.disabled = true;

  try {
    const response = await window.api.runPrompt(prompt, currentScene);
    if (response.ok) {
      const { result } = response;
      currentScene = result.scene;
      const accepted = typeof result.accepted === "number" ? result.accepted.toPrecision(4) : result.accepted;
      addMessage("bot", { meta: `accepted yield ≈ ${accepted}`, text: result.report });
      render3d(result.html);
    } else {
      addMessage("error", { meta: "Could not run that", text: response.error });
    }
  } catch (err) {
    addMessage("error", { meta: "Unexpected error", text: err && err.message ? err.message : String(err) });
  } finally {
    clearStatus();
    input.disabled = false;
    send.disabled = false;
    input.focus();
  }
});
