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
from latent_dirac.scene.build import build_source, run_scene
from latent_dirac.scene.loader import scene_from_mapping

_log = logging.getLogger("latent_dirac.bridge")

_DEFAULT_NOTE = "AI-generated scene; transport and acceptance diagnostic"


def _json_errors(exc: ValidationError) -> list:
    # ValidationError.errors() can carry non-JSON objects (ctx, input);
    # exc.json() renders a JSON-safe string
    return json.loads(exc.json())


# plotly's built-in Play/slider drive frames via Plotly.animate, which does NOT
# repaint gl3d (3D) marker traces — so the button appears dead. Plotly.restyle,
# by contrast, reliably updates and repaints 3D traces. This post-plot script
# strips plotly's dead controls and drives the recorded frames itself with
# restyle, behind a self-contained Play/Pause + scrub bar. {plot_id} is
# substituted by plotly.to_html; __LD_FRAMES__/__LD_MARKER__ by us.
_ANIM_JS = """
(function(){
  var gd = document.getElementById('{plot_id}');
  if (!gd || !window.Plotly) return;
  var ldFrames = __LD_FRAMES__;      // [[x[], y[], z[]], ...] per step
  var ldMarker = __LD_MARKER__;      // index of the animated cloud trace
  if (!ldFrames.length) return;
  var i = 0, timer = null;
  function draw(k){
    var f = ldFrames[k];
    window.Plotly.restyle(gd, {x:[f[0]], y:[f[1]], z:[f[2]]}, [ldMarker]);
  }
  var bar = document.createElement('div');
  bar.style.cssText = [
    'position:fixed;left:50%;bottom:12px;transform:translateX(-50%);z-index:9999',
    'display:flex;align-items:center;gap:8px;background:rgba(255,255,255,.9)',
    'border:1px solid #e4dfd5;padding:6px 10px;color:#1b2431',
    'font:12px -apple-system,BlinkMacSystemFont,sans-serif'
  ].join(';');
  var btn = document.createElement('button');
  btn.textContent = '\\u25B6 Play';
  btn.style.cssText = 'border:0;background:#3a5bd9;color:#fff;padding:5px 12px;cursor:pointer;font:inherit';
  var slider = document.createElement('input');
  slider.type = 'range';
  slider.min = '0';
  slider.max = String(ldFrames.length - 1);
  slider.value = '0';
  slider.style.width = '220px';
  var lab = document.createElement('span');
  lab.style.color = '#6b7688';
  function setLab(){ lab.textContent = 'step ' + i + ' / ' + (ldFrames.length - 1); }
  function stop(){
    if (timer){ clearInterval(timer); timer = null; btn.textContent = '\\u25B6 Play'; }
  }
  var adv = 4;   // frames advanced per tick; 18ms tick x 4 ~= 20x the old 90ms/frame
  btn.addEventListener('click', function(){
    if (timer){ stop(); return; }
    btn.textContent = '\\u23F8 Pause';
    timer = setInterval(function(){
      i = (i + adv) % ldFrames.length;
      slider.value = String(i);
      draw(i);
      setLab();
    }, 18);
  });
  slider.addEventListener('input', function(){
    stop();
    i = parseInt(slider.value, 10);
    draw(i);
    setLab();
  });
  bar.appendChild(btn); bar.appendChild(slider); bar.appendChild(lab);
  document.body.appendChild(bar);
  setLab(); draw(0);
})();
"""


def _animated_html(figure) -> str:
    # extract the per-step marker coords, then drop plotly's dead 3D controls +
    # frames and drive them ourselves via restyle (see _ANIM_JS)
    marker_index = len(figure.data) - 1
    frames_xyz = [
        [list(frame.data[0].x), list(frame.data[0].y), list(frame.data[0].z)]
        for frame in figure.frames
    ]
    figure.frames = []
    figure.layout.updatemenus = []
    figure.layout.sliders = []
    post = _ANIM_JS.replace("__LD_FRAMES__", json.dumps(frames_xyz)).replace(
        "__LD_MARKER__", str(marker_index)
    )
    return figure.to_html(include_plotlyjs=True, full_html=True, post_script=post)


def _render_html(scene, result, req: dict) -> str:
    # offline/self-contained: plotly is inlined so the desktop 3D panel renders
    # with no network
    from latent_dirac.viz.scene_3d import render_scene_3d, render_scene_animation

    animate = req.get("animate", True)
    color = req.get("color", "fate")
    max_particles = int(req.get("max_particles", 64))
    if animate:
        try:
            figure = render_scene_animation(scene, result, max_particles=max_particles, color=color)
        except ValueError:
            # a transport-less scene (e.g. a beam straight onto a plate) records
            # no stepped trajectory to animate; fall back to a static 3D below
            figure = None
        if figure is not None:
            return _animated_html(figure)
    return render_scene_3d(scene, result, max_particles=max_particles).to_html(
        include_plotlyjs=True, full_html=True
    )


def handle_request(req: dict) -> dict:
    op = req.get("op")

    if op == "schema":
        from latent_dirac.scene.schema import Scene

        return {"ok": True, "result": Scene.model_json_schema()}

    if op == "source_params":
        # The scene schema types `source.params` as an open dict, so the AI has
        # no way to know a source's required params from `schema` alone. Expose
        # each source type's param model so the client can hand it to the AI.
        from latent_dirac.scene.build import _SOURCE_CLASSES

        return {
            "ok": True,
            "result": {name: cls.model_json_schema() for name, cls in _SOURCE_CLASSES.items()},
        }

    if op == "validate":
        try:
            scene = scene_from_mapping(req["scene"])
        except ValidationError as exc:
            return {"ok": False, "error": {"type": "validation", "errors": _json_errors(exc)}}
        # `source.params` is an open dict at the scene layer — construct the
        # source here to catch param errors as *validation* (so the AI retry
        # loop can fix them) rather than letting them explode at run time.
        try:
            build_source(scene)
        except ValidationError as exc:
            return {"ok": False, "error": {"type": "validation", "errors": _json_errors(exc)}}
        except Exception:
            pass  # non-param construction issues (e.g. a missing data file) surface at run
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
        except ValidationError as exc:  # source/element param errors -> validation, retryable
            return {"ok": False, "error": {"type": "validation", "errors": _json_errors(exc)}}
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
