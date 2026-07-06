"""Request dispatch for the local engine bridge — pure, framework-free.

Turns a decoded request dict into a response dict. The stdio loop
(`latent_dirac.bridge.__main__`) is the only I/O; keeping dispatch pure makes
it directly unit-testable and reusable.

Ops:
- {"op": "schema"}                 -> {"ok": True, "result": <Scene JSON Schema>}
- {"op": "validate", "scene": ...} -> {"ok": True} | {"ok": False, "error": {type: validation, errors}}
- {"op": "run", "scene": ...}      -> {"ok": True, "result": {report, html, accepted, losses}}
                                      | {"ok": False, "error": {type: validation|runtime, ...}}
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from latent_dirac.diagnostics.loss_ledger import loss_ledger
from latent_dirac.diagnostics.scene_report import scene_report
from latent_dirac.scene.build import run_scene
from latent_dirac.scene.loader import scene_from_mapping

_log = logging.getLogger("latent_dirac.bridge")

_DEFAULT_NOTE = "AI-generated scene; transport and acceptance diagnostic"


def _json_errors(exc: ValidationError) -> list:
    # ValidationError.errors() can carry non-JSON objects (ctx, input);
    # exc.json() renders a JSON-safe string
    return json.loads(exc.json())


def _render_html(scene, result, req: dict) -> str:
    # offline/self-contained: plotly is inlined so the desktop 3D panel renders
    # with no network
    from latent_dirac.viz.scene_3d import render_scene_3d, render_scene_animation

    animate = req.get("animate", True)
    color = req.get("color", "fate")
    max_particles = int(req.get("max_particles", 64))
    if animate:
        figure = render_scene_animation(scene, result, max_particles=max_particles, color=color)
    else:
        figure = render_scene_3d(scene, result, max_particles=max_particles)
    return figure.to_html(include_plotlyjs=True, full_html=True)


def handle_request(req: dict) -> dict:
    op = req.get("op")

    if op == "schema":
        from latent_dirac.scene.schema import Scene

        return {"ok": True, "result": Scene.model_json_schema()}

    if op == "validate":
        try:
            scene_from_mapping(req["scene"])
        except ValidationError as exc:
            return {"ok": False, "error": {"type": "validation", "errors": _json_errors(exc)}}
        return {"ok": True, "result": {"valid": True}}

    if op == "run":
        try:
            scene = scene_from_mapping(req["scene"])
        except ValidationError as exc:
            return {"ok": False, "error": {"type": "validation", "errors": _json_errors(exc)}}
        try:
            result = run_scene(scene, record_trajectories=True)
            report = scene_report(scene, result, req.get("scope_note", _DEFAULT_NOTE))
            html = _render_html(scene, result, req)
        except Exception as exc:  # run-time engine/adapter failures -> runtime error, never a crash
            _log.exception("scene run failed")
            return {
                "ok": False,
                "error": {"type": "runtime", "detail": str(exc), "error_type": type(exc).__name__},
            }
        losses = loss_ledger(result.pipeline_result.final_cloud, result.pipeline_result.stage_results)
        return {
            "ok": True,
            "result": {
                "report": report,
                "html": html,
                "accepted": result.pipeline_result.final_cloud.weighted_count(),
                "losses": losses,
            },
        }

    return {"ok": False, "error": {"type": "bad_request", "detail": f"unknown op: {op!r}"}}
