"use strict";
// Shared helper: attach a machine-readable `category` (and optional extras) to
// an Error so the renderer can show a category-specific hint. Used by both the
// orchestrator and the BYOK AI client.

function categorized(message, category, extra = {}) {
  const err = new Error(message);
  Object.assign(err, extra);
  err.category = category; // authoritative — set after extra so it can't be clobbered
  return err;
}

module.exports = { categorized };
