"use strict";
// Dashboard UI. A prompt goes to the main process (which calls Anthropic with
// the user's key, then validate -> run); the result fills four panels: the 3D
// viewport, live physics, and a tabbed tools panel (Ledger / Inspector /
// Sweep). window.api is from preload.js; window.Panels is src/panels.js.

const P = window.Panels;
if (!P) {
  // panels.js failed to load (e.g. blocked by CSP) — fail loudly, not silently
  document.getElementById("log").textContent =
    "Failed to load panels.js — the app cannot start. Please report this.";
  throw new Error("window.Panels is undefined (panels.js did not load)");
}

const frame = document.getElementById("frame");
const placeholder = document.getElementById("placeholder");
const viewMeta = document.getElementById("view-meta");
const log = document.getElementById("log");
const statusEl = document.getElementById("status");
const form = document.getElementById("form");
const input = document.getElementById("prompt");
const send = document.getElementById("send");
const newBtn = document.getElementById("new-scene");
const saveBtn = document.getElementById("save-scene");
const loadBtn = document.getElementById("load-scene");
const keyBtn = document.getElementById("key-btn");
const stats = document.getElementById("stats");
const tabbody = document.getElementById("tabbody");
const physLive = document.getElementById("phys-live");
const tabButtons = Array.from(document.querySelectorAll(".tab"));

let currentScene = null;
let currentResult = null;
let activeTab = "ledger";

const STATUS_TEXT = {
  generating: "Generating a scene from your description…",
  validating: "Checking the scene against the engine schema…",
  retrying: "The scene needed a fix; asking again…",
  running: "Running the simulation locally…",
  done: "Done.",
};

const CATEGORY_HINT = {
  "ai-no-key": "Add your Anthropic API key first — click “Key”.",
  "ai-bad-key": "That API key was rejected — click “Key” to update it.",
  "ai-unreachable": "Can't reach the Anthropic API — check your connection.",
  "ai-error": "The AI service returned an error. Try again in a moment.",
  "ai-no-scene": "The AI didn't return a scene. Try rephrasing the request.",
  "engine-unreachable": "The local sim engine isn't responding.",
  "validation-giveup": "The AI couldn't produce a valid scene. Try rephrasing.",
  "engine-runtime": "The scene is valid but couldn't be run (an element may need an external engine).",
};

const EXAMPLES = [
  "a positron pair through a solenoid into an aperture",
  "a beta-plus source decelerated into a Penning trap",
  "an antiproton beam through a quadrupole doublet then a monitor",
  "a positron beam onto an annihilation plate",
];

function fmt(n) {
  if (typeof n !== "number" || !Number.isFinite(n)) return String(n);
  if (Math.abs(n) >= 1000) return Math.round(n).toLocaleString();
  return String(Math.round(n * 100) / 100);
}

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

function setBusy(busy) {
  input.disabled = busy;
  send.disabled = busy;
  newBtn.disabled = busy;
  saveBtn.disabled = busy;
  loadBtn.disabled = busy;
  const rerun = tabbody.querySelector(".rerun");
  if (rerun) rerun.disabled = busy;
  if (!busy) {
    clearStatus();
    input.focus();
  }
}

function render3d(html) {
  frame.srcdoc = html;
  frame.hidden = false;
  placeholder.hidden = true;
}

function statCell(value, cls, caption) {
  const cell = document.createElement("div");
  cell.className = "stat";
  const v = document.createElement("span");
  v.className = "v" + (cls ? " " + cls : "");
  v.textContent = value;
  const c = document.createElement("span");
  c.className = "c";
  c.textContent = caption;
  cell.appendChild(v);
  cell.appendChild(c);
  return cell;
}

function renderPhysics() {
  stats.innerHTML = "";
  if (!currentResult) {
    physLive.textContent = "";
    ["accepted", "transmission", "lost", "stages"].forEach((c) => stats.appendChild(statCell("—", "", c)));
    return;
  }
  const s = P.physicsSummary(currentResult);
  physLive.textContent = "ready";
  stats.appendChild(statCell(fmt(s.accepted), "good", "accepted"));
  stats.appendChild(statCell(s.transmissionPct.toFixed(1) + "%", "acc", "transmission"));
  stats.appendChild(statCell(fmt(s.lost), "lost", "lost"));
  stats.appendChild(statCell(String(s.stages), "", "loss stages"));
}

function emptyPanel(text) {
  const d = document.createElement("div");
  d.className = "empty";
  d.textContent = text;
  return d;
}

function renderLedger() {
  tabbody.innerHTML = "";
  if (!currentResult) {
    tabbody.appendChild(emptyPanel("Run a scene to see where particles are lost."));
    return;
  }
  const rows = P.ledgerRows(currentResult.losses);
  const wrap = document.createElement("div");
  wrap.className = "ledger";
  const s = P.physicsSummary(currentResult);
  const head = document.createElement("div");
  head.className = "lhead";
  if (!rows.length) {
    head.textContent = "No losses recorded — every particle was accepted.";
    wrap.appendChild(head);
    tabbody.appendChild(wrap);
    return;
  }
  const maxLost = Math.max(...rows.map((r) => r.lost), 1);
  head.innerHTML = `<b>${fmt(s.lost)}</b> lost of ${fmt(s.accepted + s.lost)} · by stage`;
  wrap.appendChild(head);
  rows.forEach((r) => {
    const row = document.createElement("div");
    row.className = "lrow";
    const n = document.createElement("span");
    n.className = "n";
    n.textContent = r.stage;
    n.title = r.stage;
    const bar = document.createElement("span");
    bar.className = "bar";
    const x = document.createElement("span");
    x.className = "x";
    x.style.width = (100 * r.lost) / maxLost + "%";
    bar.appendChild(x);
    const c = document.createElement("span");
    c.className = "c";
    c.textContent = fmt(r.lost);
    row.appendChild(n);
    row.appendChild(bar);
    row.appendChild(c);
    wrap.appendChild(row);
  });
  tabbody.appendChild(wrap);
}

function renderInspector() {
  tabbody.innerHTML = "";
  if (!currentScene) {
    tabbody.appendChild(emptyPanel("Run or load a scene to inspect its beamline."));
    return;
  }
  const els = P.sceneElements(currentScene);
  const wrap = document.createElement("div");
  wrap.className = "insp";
  els.forEach((e, i) => {
    const row = document.createElement("div");
    row.className = "irow";
    const ix = document.createElement("span");
    ix.className = "ix";
    ix.textContent = e.kind === "source" ? "S" : String(i);
    const nm = document.createElement("span");
    nm.className = "nm";
    const b = document.createElement("b");
    b.textContent = e.label;
    const ty = document.createElement("span");
    ty.className = "ty";
    ty.textContent = e.type;
    nm.appendChild(b);
    nm.appendChild(ty);
    if (e.summary) {
      const small = document.createElement("small");
      small.textContent = e.summary;
      nm.appendChild(small);
    }
    row.appendChild(ix);
    row.appendChild(nm);
    wrap.appendChild(row);
  });
  tabbody.appendChild(wrap);
}

function renderSweep() {
  tabbody.innerHTML = "";
  if (!currentScene) {
    tabbody.appendChild(emptyPanel("Run a scene first, then tweak a parameter and re-run."));
    return;
  }
  const params = P.numericParams(currentScene);
  if (!params.length) {
    tabbody.appendChild(emptyPanel("This scene has no numeric element parameters to sweep."));
    return;
  }
  const wrap = document.createElement("div");
  wrap.className = "sweep";

  const pLabel = document.createElement("label");
  pLabel.textContent = "Parameter";
  const select = document.createElement("select");
  params.forEach((p, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `${p.elementLabel} · ${p.key}`;
    select.appendChild(opt);
  });

  const vLabel = document.createElement("label");
  vLabel.textContent = "Value";
  const value = document.createElement("input");
  value.type = "number";
  value.step = "any";
  value.value = String(params[0].value);

  select.addEventListener("change", () => {
    value.value = String(params[Number(select.value)].value);
  });

  const rerun = document.createElement("button");
  rerun.className = "rerun";
  rerun.type = "button";
  rerun.textContent = "Re-run";
  rerun.addEventListener("click", () => {
    const p = params[Number(select.value)];
    const v = Number(value.value);
    if (!Number.isFinite(v)) return;
    const next = P.setParam(currentScene, p.path, v);
    runSceneFlow(next, `Sweep: ${p.elementLabel} · ${p.key} = ${v}`);
  });

  wrap.appendChild(pLabel);
  wrap.appendChild(select);
  wrap.appendChild(vLabel);
  wrap.appendChild(value);
  wrap.appendChild(rerun);
  tabbody.appendChild(wrap);
}

function renderTools() {
  if (activeTab === "ledger") renderLedger();
  else if (activeTab === "inspector") renderInspector();
  else renderSweep();
}

function showResultEverywhere(result, botMeta) {
  currentResult = result;
  currentScene = result.scene;
  render3d(result.html);
  viewMeta.textContent = P.sceneElements(result.scene).map((e) => e.type).join(" → ");
  renderPhysics();
  renderTools();
  const accepted = fmt(P.physicsSummary(result).accepted);
  addMessage("bot", { meta: botMeta || `accepted yield ≈ ${accepted}`, text: result.report });
}

function showError(response) {
  addMessage("error", { meta: CATEGORY_HINT[response.category] || "Could not complete that", text: response.error });
}

async function runPromptFlow(prompt) {
  addMessage("user", { text: prompt });
  setBusy(true);
  try {
    const r = await window.api.runPrompt(prompt, currentScene);
    if (r.ok) showResultEverywhere(r.result);
    else showError(r);
  } catch (err) {
    addMessage("error", { meta: "Unexpected error", text: err && err.message ? err.message : String(err) });
  } finally {
    setBusy(false);
  }
}

async function runSceneFlow(scene, botMeta) {
  setBusy(true);
  try {
    const r = await window.api.runScene(scene);
    if (r.ok) showResultEverywhere(r.result, botMeta);
    else showError(r);
  } catch (err) {
    addMessage("error", { meta: "Unexpected error", text: err && err.message ? err.message : String(err) });
  } finally {
    setBusy(false);
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const prompt = input.value.trim();
  if (!prompt) return;
  input.value = "";
  runPromptFlow(prompt);
});

newBtn.addEventListener("click", () => {
  currentScene = null;
  currentResult = null;
  frame.srcdoc = "";
  frame.hidden = true;
  placeholder.hidden = false;
  viewMeta.textContent = "";
  renderPhysics();
  renderTools();
  addMessage("bot", { meta: "New scene", text: "Started fresh — describe a new beamline." });
});

saveBtn.addEventListener("click", async () => {
  if (!currentScene) {
    addMessage("bot", { meta: "Nothing to save", text: "Run a scene first, then save it." });
    return;
  }
  try {
    const res = await window.api.saveScene(currentScene);
    if (res.ok) addMessage("bot", { meta: "Saved", text: res.filePath });
    else if (!res.canceled) showError(res);
  } catch (err) {
    addMessage("error", { meta: "Save failed", text: err && err.message ? err.message : String(err) });
  }
});

loadBtn.addEventListener("click", async () => {
  try {
    const res = await window.api.openScene();
    if (!res.ok) {
      if (!res.canceled) showError(res);
      return;
    }
    addMessage("user", { meta: "Loaded scene", text: res.filePath });
    runSceneFlow(res.scene);
  } catch (err) {
    addMessage("error", { meta: "Load failed", text: err && err.message ? err.message : String(err) });
  }
});

function askKey() {
  const kf = document.createElement("form");
  kf.className = "keyform";
  const inp = document.createElement("input");
  inp.type = "password";
  inp.placeholder = "sk-ant-…";
  inp.autocomplete = "off";
  const save = document.createElement("button");
  save.type = "submit";
  save.textContent = "Save";
  const clear = document.createElement("button");
  clear.type = "button";
  clear.textContent = "Clear";
  const err = document.createElement("span");
  err.className = "keyform__err";
  err.setAttribute("role", "alert");
  kf.appendChild(inp);
  kf.appendChild(save);
  kf.appendChild(clear);
  kf.appendChild(err);
  kf.addEventListener("submit", async (e) => {
    e.preventDefault();
    err.textContent = "";
    const v = inp.value.trim();
    if (!v) return;
    if (!v.startsWith("sk-ant-")) {
      err.textContent = "That doesn't look like an Anthropic key (they start with “sk-ant-”).";
      return;
    }
    const r = await window.api.setApiKey(v);
    inp.value = "";
    const where = r.encrypted
      ? "stored encrypted on this machine"
      : "stored on this machine (no OS keychain here, so at rest in plaintext)";
    addMessage("bot", { meta: "Key", text: r.hasKey ? `API key saved — ${where}.` : "No key set." });
  });
  clear.addEventListener("click", async () => {
    await window.api.clearApiKey();
    inp.value = "";
    addMessage("bot", { meta: "Key", text: "API key cleared." });
  });
  addMessage("bot", { meta: "Your Anthropic API key (BYOK)", text: "Stays on this machine, sent only to Anthropic.", node: kf });
  inp.focus();
}
keyBtn.addEventListener("click", askKey);

tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    tabButtons.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    activeTab = btn.dataset.tab;
    renderTools();
  });
});

function welcome() {
  const chips = document.createElement("div");
  chips.className = "chips";
  EXAMPLES.forEach((ex) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = ex;
    chip.addEventListener("click", () => {
      input.value = ex;
      input.focus();
    });
    chips.appendChild(chip);
  });
  addMessage("bot", { meta: "Try one of these", text: "Describe a simulation in plain language — it runs locally and shows in 3D.", node: chips });
}

// init
window.api.onStatus(showStatus);
renderPhysics();
renderTools();
welcome();
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
