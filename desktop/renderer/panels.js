"use strict";
// Pure data transforms feeding the dashboard panels from a /run result and the
// scene. No DOM here — the renderer formats these into HTML. UMD so the same
// source is unit-tested in Node (module.exports) and loaded in the browser
// renderer (window.Panels). Lives in renderer/ so the page can load it same-dir
// under a strict `script-src 'self'` CSP.

(function (global, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    global.Panels = api;
  }
})(typeof self !== "undefined" ? self : this, function () {
  // loss_ledger() adds a reserved "surviving" entry (== accepted); it is NOT a
  // loss stage, so every consumer must exclude it.
  const RESERVED = "surviving";

  function physicsSummary(result) {
    const losses = (result && result.losses) || {};
    const lost = Object.entries(losses).reduce(
      (a, [k, v]) => a + (k === RESERVED ? 0 : Number(v) || 0),
      0
    );
    const accepted = Number(result && result.accepted) || 0;
    const total = accepted + lost;
    return {
      accepted,
      lost,
      transmissionPct: total > 0 ? (100 * accepted) / total : 0,
      stages: Object.keys(losses).filter((k) => k !== RESERVED).length,
    };
  }

  function ledgerRows(losses) {
    return Object.entries(losses || {})
      .filter(([k]) => k !== RESERVED)
      .map(([stage, lost]) => ({ stage, lost: Number(lost) || 0 }));
  }

  function elementSummary(element) {
    return Object.keys(element)
      .filter((k) => k !== "type" && k !== "label")
      .map((k) => {
        const v = element[k];
        return typeof v === "object" ? k : `${k}=${v}`;
      })
      .join(" · ");
  }

  function sceneElements(scene) {
    const out = [];
    if (scene && scene.source) {
      out.push({
        kind: "source",
        type: scene.source.type,
        label: scene.source.label || scene.source.type,
        summary: elementSummary(scene.source),
      });
    }
    const els = (scene && scene.elements) || [];
    els.forEach((e, i) => {
      out.push({ kind: "element", index: i, type: e.type, label: e.label || e.type, summary: elementSummary(e) });
    });
    return out;
  }

  // Sweepable numeric params, elements only (a source param would need a
  // separate path shape in setParam; deferred).
  function numericParams(scene) {
    const out = [];
    const els = (scene && scene.elements) || [];
    els.forEach((e, i) => {
      Object.keys(e).forEach((k) => {
        if (typeof e[k] === "number") {
          out.push({ elementLabel: e.label || e.type, key: k, value: e[k], path: [i, k] });
        }
      });
    });
    return out;
  }

  function setParam(scene, path, value) {
    const clone = JSON.parse(JSON.stringify(scene));
    const [i, k] = path;
    clone.elements[i][k] = value;
    return clone;
  }

  return { physicsSummary, ledgerRows, sceneElements, elementSummary, numericParams, setParam };
});
