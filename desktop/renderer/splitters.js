"use strict";
// Draggable dock splitters. Each `[data-resize]` handle resizes its previous
// sibling pane (flex-basis) along `data-axis` ("x" = width, "y" = height); the
// opposite pane (flex: 1) absorbs the rest. Sizes persist in localStorage.
// UMD so the pure size math is unit-tested in Node and the rest runs in the
// browser renderer (window.Splitters).

(function (global, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else global.Splitters = api;
})(typeof self !== "undefined" ? self : this, function () {
  const STORE = "latentdirac.split.";
  const MIN = 160; // px a resized pane may not go below
  const KEEP = 200; // px the sibling pane must keep

  // pure: the new flex-basis in px for a drag, clamped so neither pane collapses
  function resizeBasis(startPx, deltaPx, minPx, maxPx) {
    const lo = minPx;
    const hi = Math.max(minPx, maxPx);
    return Math.max(lo, Math.min(hi, startPx + deltaPx));
  }

  function makeSplitter(handle, target, axis, onEnd) {
    handle.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      try {
        handle.setPointerCapture(event.pointerId);
      } catch {
        /* capture unsupported; document-level fallback still works */
      }
      handle.classList.add("dragging");
      document.body.classList.add("resizing");

      const horizontal = axis === "x";
      const start = horizontal ? event.clientX : event.clientY;
      const rect = target.getBoundingClientRect();
      const startSize = horizontal ? rect.width : rect.height;
      const parent = target.parentElement.getBoundingClientRect();
      const parentSize = horizontal ? parent.width : parent.height;
      const max = parentSize - KEEP;

      const move = (ev) => {
        const cur = horizontal ? ev.clientX : ev.clientY;
        target.style.flexBasis = resizeBasis(startSize, cur - start, MIN, max) + "px";
      };
      const up = () => {
        handle.classList.remove("dragging");
        document.body.classList.remove("resizing");
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", up);
        handle.removeEventListener("pointercancel", up); // gesture/OS interruption
        if (onEnd) onEnd();
      };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", up);
      handle.addEventListener("pointercancel", up);
    });
  }

  function initSplitters(root) {
    const doc = root || document;
    const handles = Array.from(doc.querySelectorAll("[data-resize]"));

    const save = () => {
      handles.forEach((h) => {
        const t = h.previousElementSibling;
        if (t && t.style.flexBasis) {
          try {
            localStorage.setItem(STORE + h.dataset.resize, t.style.flexBasis);
          } catch {
            /* storage unavailable — resizing still works, just not persisted */
          }
        }
      });
    };

    handles.forEach((h) => {
      const target = h.previousElementSibling;
      if (!target) return;
      const axis = h.dataset.axis === "x" ? "x" : "y";
      try {
        const saved = localStorage.getItem(STORE + h.dataset.resize);
        if (saved) target.style.flexBasis = saved;
      } catch {
        /* ignore */
      }
      makeSplitter(h, target, axis, save);
    });
  }

  return { resizeBasis, makeSplitter, initSplitters };
});
