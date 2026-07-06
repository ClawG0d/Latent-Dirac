"use strict";
// Chat UI. Sends a prompt to the main process (which calls Anthropic with the
// user's key, then validate -> run), streams status stages, shows the report and
// loads the offline 3D HTML into the panel. Also: save/load a scene, start a
// fresh scene, seed example prompts, and category-aware error hints.
// window.api is defined by preload.js.

const form = document.getElementById("form");
const input = document.getElementById("prompt");
const send = document.getElementById("send");
const log = document.getElementById("log");
const statusEl = document.getElementById("status");
const frame = document.getElementById("frame");
const placeholder = document.getElementById("placeholder");
const newBtn = document.getElementById("new-scene");
const saveBtn = document.getElementById("save-scene");
const loadBtn = document.getElementById("load-scene");
const keyBtn = document.getElementById("key-btn");

// the last valid scene, so a follow-up prompt edits it rather than start over
let currentScene = null;

const STATUS_TEXT = {
  generating: "Generating a scene from your description…",
  validating: "Checking the scene against the engine schema…",
  retrying: "The scene needed a fix; asking again…",
  running: "Running the simulation locally…",
  done: "Done.",
};

// a short, human hint per error category from the orchestrator / BYOK client
const CATEGORY_HINT = {
  "ai-no-key": "Add your Anthropic API key first — click “Key”.",
  "ai-bad-key": "That API key was rejected — click “Key” to update it.",
  "ai-unreachable": "Can't reach the Anthropic API — check your connection.",
  "ai-error": "The AI service returned an error. Try again in a moment.",
  "ai-no-scene": "The AI didn't return a scene. Try rephrasing the request.",
  "engine-unreachable": "The local sim engine isn't responding.",
  "validation-giveup": "The AI couldn't produce a valid scene. Try rephrasing the request.",
  "engine-runtime": "The scene is valid but couldn't be run (an element may need an external engine).",
};

const EXAMPLES = [
  "a positron pair through a solenoid into an aperture",
  "a beta-plus source decelerated into a Penning trap",
  "an antiproton beam through a quadrupole doublet then a monitor",
  "a positron beam onto an annihilation plate",
];

function addMessage(kind, { meta, text, node } = {}) {
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
  if (node) el.appendChild(node);
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

function showStatus(stage) {
  statusEl.hidden = false;
  statusEl.innerHTML = "";
  const dot = document.createElement("span");
  dot.className = "status__dot";
  statusEl.appendChild(dot);
  statusEl.appendChild(document.createTextNode(STATUS_TEXT[stage] || stage));
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

function showResult(result) {
  currentScene = result.scene;
  const accepted =
    typeof result.accepted === "number" ? result.accepted.toPrecision(4) : result.accepted;
  addMessage("bot", { meta: `accepted yield ≈ ${accepted}`, text: result.report });
  render3d(result.html);
}

function showError(response) {
  const hint = CATEGORY_HINT[response.category];
  addMessage("error", {
    meta: hint || "Could not complete that",
    text: response.error,
  });
}

function setBusy(busy) {
  input.disabled = busy;
  send.disabled = busy;
  newBtn.disabled = busy;
  saveBtn.disabled = busy;
  loadBtn.disabled = busy;
  if (!busy) {
    clearStatus();
    input.focus();
  }
}

async function runPrompt(prompt) {
  addMessage("user", { text: prompt });
  setBusy(true);
  try {
    const response = await window.api.runPrompt(prompt, currentScene);
    if (response.ok) showResult(response.result);
    else showError(response);
  } catch (err) {
    addMessage("error", { meta: "Unexpected error", text: err && err.message ? err.message : String(err) });
  } finally {
    setBusy(false);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const prompt = input.value.trim();
  if (!prompt) return;
  input.value = "";
  runPrompt(prompt);
});

newBtn.addEventListener("click", () => {
  currentScene = null;
  frame.srcdoc = "";
  frame.hidden = true;
  placeholder.hidden = false;
  addMessage("bot", { meta: "New scene", text: "Started fresh — describe a new beamline." });
});

saveBtn.addEventListener("click", async () => {
  if (!currentScene) {
    addMessage("bot", { meta: "Nothing to save", text: "Run a scene first, then save it." });
    return;
  }
  const res = await window.api.saveScene(currentScene);
  if (res.ok) addMessage("bot", { meta: "Saved", text: res.filePath });
  else if (!res.canceled) showError(res);
});

loadBtn.addEventListener("click", async () => {
  const res = await window.api.openScene();
  if (!res.ok) {
    if (!res.canceled) showError(res);
    return;
  }
  addMessage("user", { meta: "Loaded scene", text: res.filePath });
  setBusy(true);
  try {
    const run = await window.api.runScene(res.scene);
    if (run.ok) showResult(run.result);
    else showError(run);
  } finally {
    setBusy(false);
  }
});

function askKey() {
  const form = document.createElement("form");
  form.className = "keyform";
  const input = document.createElement("input");
  input.type = "password";
  input.placeholder = "sk-ant-…";
  input.autocomplete = "off";
  const save = document.createElement("button");
  save.type = "submit";
  save.textContent = "Save";
  const clear = document.createElement("button");
  clear.type = "button";
  clear.textContent = "Clear";
  const err = document.createElement("span");
  err.className = "keyform__err";
  err.setAttribute("role", "alert");
  form.appendChild(input);
  form.appendChild(save);
  form.appendChild(clear);
  form.appendChild(err);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    err.textContent = "";
    const v = input.value.trim();
    if (!v) return;
    if (!v.startsWith("sk-ant-")) {
      err.textContent = "That doesn't look like an Anthropic key (they start with “sk-ant-”).";
      return;
    }
    const r = await window.api.setApiKey(v);
    input.value = "";
    const where = r.encrypted
      ? "stored encrypted on this machine"
      : "stored on this machine (no OS keychain here, so at rest in plaintext)";
    addMessage("bot", { meta: "Key", text: r.hasKey ? `API key saved — ${where}.` : "No key set." });
  });
  clear.addEventListener("click", async () => {
    await window.api.clearApiKey();
    input.value = "";
    addMessage("bot", { meta: "Key", text: "API key cleared." });
  });
  addMessage("bot", { meta: "Your Anthropic API key (BYOK)", text: "Stays on this machine, sent only to Anthropic. Not shared.", node: form });
  input.focus();
}

keyBtn.addEventListener("click", askKey);

function welcome() {
  const chips = document.createElement("div");
  chips.className = "chips";
  for (const example of EXAMPLES) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = example;
    chip.addEventListener("click", () => {
      input.value = example;
      input.focus();
    });
    chips.appendChild(chip);
  }
  addMessage("bot", {
    meta: "Try one of these",
    text: "Describe a simulation in plain language — it runs locally and shows in 3D.",
    node: chips,
  });
}

window.api.onStatus(showStatus);
welcome();

// nudge for a key on first launch (BYOK)
(async () => {
  try {
    const s = await window.api.keyStatus();
    if (!s.hasKey) {
      addMessage("bot", {
        meta: "Bring your own key",
        text: "This app uses your own Anthropic API key. Click “Key” to add it — it stays on this machine.",
      });
    }
  } catch {
    /* keyStatus unavailable outside Electron */
  }
})();
