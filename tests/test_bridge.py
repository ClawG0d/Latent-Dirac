"""Tests for the local engine stdio bridge (desktop-client backend).

The bridge replaces the earlier HTTP server: the desktop app spawns
`python -m latent_dirac.bridge` and talks line-delimited JSON over
stdin/stdout — no HTTP, no port, no FastAPI. Pure Python, always available.
"""

from __future__ import annotations

import io
import json

from latent_dirac.bridge import handle_request
from latent_dirac.bridge.__main__ import serve


def valid_scene():
    return {
        "schema_version": 1,
        "name": "bridge-scene",
        "seed": 7,
        "solver": {"dt_s": 4.0e-12, "steps": 40},
        "source": {
            "type": "positron_pair",
            "label": "pairs",
            "params": {
                "primary_count": 10000,
                "yield_eplus_per_primary": 0.02,
                "mean_energy_MeV": 2.0,
                "energy_spread_MeV": 0.3,
                "angular_rms_rad": 0.1,
                "source_sigma_m": 0.001,
                "bunch_length_s": 1.0e-10,
                "macro_particles": 64,
            },
        },
        "elements": [
            {"type": "solenoid", "label": "cap", "b_tesla": 0.6, "radius_m": 0.03,
             "length_m": 0.2, "center_z_m": 0.1},
            {"type": "aperture", "label": "iris", "radius_m": 0.01, "z_m": 0.1},
            {"type": "monitor", "label": "end"},
        ],
    }


def test_schema_op_returns_scene_json_schema():
    resp = handle_request({"op": "schema"})
    assert resp["ok"] is True
    schema = resp["result"]
    assert "properties" in schema
    for key in ("source", "solver", "elements", "name", "seed"):
        assert key in schema["properties"]


def test_validate_accepts_a_valid_scene():
    resp = handle_request({"op": "validate", "scene": valid_scene()})
    assert resp["ok"] is True


def test_validate_rejects_unknown_element_with_structured_errors():
    scene = valid_scene()
    scene["elements"][0]["type"] = "warp_drive"
    resp = handle_request({"op": "validate", "scene": scene})
    assert resp["ok"] is False
    assert resp["error"]["type"] == "validation"
    assert resp["error"]["errors"]  # structured pydantic errors for the AI retry loop


def test_run_returns_report_html_and_summary():
    resp = handle_request({"op": "run", "scene": valid_scene()})
    assert resp["ok"] is True
    result = resp["result"]
    assert "Latent Dirac scene report" in result["report"]
    assert isinstance(result["accepted"], (int, float))
    assert isinstance(result["losses"], dict) and result["losses"]
    # loss_ledger adds the reserved "surviving" entry
    assert "surviving" in result["losses"]


def test_run_html_is_self_contained_offline():
    resp = handle_request({"op": "run", "scene": valid_scene()})
    html = resp["result"]["html"]
    assert 'src="https://cdn.plot.ly' not in html
    assert "plotly.js v" in html


def test_run_rejects_invalid_scene():
    scene = valid_scene()
    scene["solver"]["steps"] = -5
    resp = handle_request({"op": "run", "scene": scene})
    assert resp["ok"] is False
    assert resp["error"]["type"] == "validation"


def test_run_reports_engine_absence_gracefully():
    # a matter_slab with no LATENT_DIRAC_G4_TRANSFORMER fails at run time; the
    # bridge must return a runtime error object, never crash the process
    scene = valid_scene()
    scene["elements"] = [
        {"type": "matter_slab", "label": "slab", "material": "G4_Al", "thickness_mm": 1.0},
        {"type": "monitor", "label": "end"},
    ]
    resp = handle_request({"op": "run", "scene": scene})
    assert resp["ok"] is False
    assert resp["error"]["type"] == "runtime"
    assert "LATENT_DIRAC_G4_TRANSFORMER" in resp["error"]["detail"]


def test_validate_catches_bad_source_params_as_validation():
    # source.params is an open dict at the scene layer; validate must still catch
    # a wrong-param source (the common AI mistake) so the retry loop can fix it,
    # instead of it sailing through and exploding at run time
    scene = valid_scene()
    scene["source"]["params"] = {"n_particles": 10}
    resp = handle_request({"op": "validate", "scene": scene})
    assert resp["ok"] is False
    assert resp["error"]["type"] == "validation"
    msgs = json.dumps(resp["error"]["errors"])
    assert "primary_count" in msgs  # the AI is told exactly what's required


def test_run_labels_source_param_errors_as_validation_not_runtime():
    scene = valid_scene()
    scene["source"]["params"] = {"n_particles": 10}
    resp = handle_request({"op": "run", "scene": scene})
    assert resp["ok"] is False
    assert resp["error"]["type"] == "validation"  # not "runtime"


def test_source_params_op_exposes_each_source_type_schema():
    resp = handle_request({"op": "source_params"})
    assert resp["ok"] is True
    params = resp["result"]
    assert "positron_pair" in params
    required = params["positron_pair"].get("required", [])
    assert "primary_count" in required and "mean_energy_MeV" in required


def test_unknown_op_is_a_bad_request():
    resp = handle_request({"op": "explode"})
    assert resp["ok"] is False
    assert resp["error"]["type"] == "bad_request"


def test_serve_prints_ready_then_answers_each_line_with_id():
    stdin = io.StringIO(
        json.dumps({"id": 1, "op": "schema"}) + "\n"
        + json.dumps({"id": 2, "op": "validate", "scene": valid_scene()}) + "\n"
    )
    stdout = io.StringIO()
    serve(stdin, stdout)
    lines = [json.loads(x) for x in stdout.getvalue().splitlines() if x.strip()]
    assert lines[0] == {"ready": True}
    assert lines[1]["id"] == 1 and lines[1]["ok"] is True
    assert lines[2]["id"] == 2 and lines[2]["ok"] is True


def test_serve_handles_malformed_json_without_dying():
    stdin = io.StringIO("{not json\n" + json.dumps({"id": 9, "op": "schema"}) + "\n")
    stdout = io.StringIO()
    serve(stdin, stdout)
    lines = [json.loads(x) for x in stdout.getvalue().splitlines() if x.strip()]
    assert lines[0] == {"ready": True}
    assert lines[1]["ok"] is False and lines[1]["error"]["type"] == "bad_request"
    # the loop survived the bad line and still answered the next request
    assert lines[2]["id"] == 9 and lines[2]["ok"] is True
