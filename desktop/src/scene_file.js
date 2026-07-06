"use strict";
// (De)serialization for Save/Load of a generated scene. The scene is already a
// plain object (the engine accepts JSON scenes as well as YAML), so we persist
// pretty-printed JSON. The fs/dialog wiring lives in main.js; these stay pure
// and unit-tested.

function serializeScene(scene) {
  return JSON.stringify(scene, null, 2) + "\n";
}

function parseSceneFile(text) {
  let data;
  try {
    data = JSON.parse(text);
  } catch (err) {
    throw new Error(`not valid JSON: ${err.message}`);
  }
  if (data === null || typeof data !== "object" || Array.isArray(data)) {
    throw new Error("a scene file must be a JSON object");
  }
  return data;
}

module.exports = { serializeScene, parseSceneFile };
