"""FastAPI app exposing the Latent Dirac core to the desktop client.

Endpoints:
- GET  /health   liveness + engine version
- GET  /schema   Scene JSON Schema (the AI's structured-output contract)
- POST /validate scene JSON -> {ok} or 422 with structured pydantic errors
- POST /run      validate + run + text report + offline 3D HTML + summary

Errors: schema validation -> 422 with `.errors()` (feeds the AI retry
loop); a run-time failure (e.g. a matter_slab with no engine binary) ->
400 with a clear message, never a 500 stack trace.
"""

from __future__ import annotations

import json
import logging
from importlib import metadata as importlib_metadata

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from latent_dirac.diagnostics.loss_ledger import loss_ledger
from latent_dirac.diagnostics.scene_report import scene_report
from latent_dirac.scene.build import run_scene
from latent_dirac.scene.loader import scene_from_mapping
from latent_dirac.scene.schema import Scene

_log = logging.getLogger("latent_dirac.server")


def _engine_version() -> str:
    try:
        return importlib_metadata.version("latent-dirac")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


class ValidateRequest(BaseModel):
    scene: dict


class RunRequest(BaseModel):
    scene: dict
    animate: bool = True
    color: str = "fate"
    max_particles: int = Field(default=64, gt=0)
    scope_note: str = "AI-generated scene; transport and acceptance diagnostic"


def _render_html(scene: Scene, result, request: RunRequest) -> str:
    # offline/self-contained: plotly is inlined so the desktop 3D panel
    # renders with no network (the CLI keeps the CDN default)
    from latent_dirac.viz.scene_3d import render_scene_3d, render_scene_animation

    if request.animate:
        figure = render_scene_animation(
            scene, result, max_particles=request.max_particles, color=request.color
        )
    else:
        figure = render_scene_3d(scene, result, max_particles=request.max_particles)
    # include_plotlyjs=True embeds the full minified plotly.js, so the HTML
    # renders offline (the desktop 3D panel has no network guarantee)
    return figure.to_html(include_plotlyjs=True, full_html=True)


def _json_errors(exc: ValidationError) -> list:
    # ValidationError.errors() can carry non-JSON objects (ctx, input);
    # exc.json() renders a JSON-safe string
    return json.loads(exc.json())


def create_app() -> FastAPI:
    app = FastAPI(title="Latent Dirac sim engine", version=_engine_version())

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "engine_version": _engine_version()}

    @app.get("/schema")
    def schema() -> dict:
        return Scene.model_json_schema()

    @app.post("/validate")
    def validate(req: ValidateRequest):
        try:
            scene_from_mapping(req.scene)
        except ValidationError as exc:
            return JSONResponse(status_code=422, content={"ok": False, "errors": _json_errors(exc)})
        return {"ok": True}

    @app.post("/run")
    def run(req: RunRequest):
        try:
            scene = scene_from_mapping(req.scene)
        except ValidationError as exc:
            return JSONResponse(status_code=422, content={"ok": False, "errors": _json_errors(exc)})
        try:
            result = run_scene(scene, record_trajectories=True)
            report = scene_report(scene, result, req.scope_note)
            html = _render_html(scene, result, req)
        except Exception as exc:  # run-time engine/adapter failures -> 400, not 500
            # log the traceback so a genuine internal bug is visible in the
            # sidecar's stderr, not just an opaque 400 to the client
            _log.exception("scene run failed")
            return JSONResponse(
                status_code=400, content={"detail": str(exc), "error_type": type(exc).__name__}
            )
        losses = loss_ledger(result.pipeline_result.final_cloud, result.pipeline_result.stage_results)
        return {
            "report": report,
            "html": html,
            "accepted": result.pipeline_result.final_cloud.weighted_count(),
            "losses": losses,
        }

    return app
