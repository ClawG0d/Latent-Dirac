"""Tests for the local sim-engine HTTP API (desktop-client backend).

The server wraps the existing core (scene_from_mapping -> run_scene ->
scene_report + offline 3D render) over localhost JSON. Requires the
optional [server] extra; skipped when FastAPI is missing.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from latent_dirac.server import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def valid_scene():
    return {
        "schema_version": 1,
        "name": "api-scene",
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


def test_schema_endpoint_returns_scene_json_schema(client):
    resp = client.get("/schema")
    assert resp.status_code == 200
    schema = resp.json()
    assert "properties" in schema
    for key in ("source", "solver", "elements", "name", "seed"):
        assert key in schema["properties"]


def test_validate_accepts_a_valid_scene(client):
    resp = client.post("/validate", json={"scene": valid_scene()})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_validate_rejects_unknown_element(client):
    scene = valid_scene()
    scene["elements"][0]["type"] = "warp_drive"
    resp = client.post("/validate", json={"scene": scene})
    assert resp.status_code == 422
    body = resp.json()
    assert body["ok"] is False
    assert body["errors"]  # structured pydantic errors for the AI retry loop


def test_validate_rejects_extra_key(client):
    scene = valid_scene()
    scene["elements"][0]["bogus"] = 1
    resp = client.post("/validate", json={"scene": scene})
    assert resp.status_code == 422
    assert resp.json()["ok"] is False


def test_run_returns_report_html_and_summary(client):
    resp = client.post("/run", json={"scene": valid_scene()})
    assert resp.status_code == 200
    body = resp.json()
    assert "Latent Dirac scene report" in body["report"]
    assert isinstance(body["accepted"], (int, float))
    assert isinstance(body["losses"], dict) and body["losses"]
    assert "<html" in body["html"].lower() or "plotly" in body["html"].lower()


def test_run_html_is_self_contained_offline(client):
    # the desktop 3D panel must render with no network: plotly is embedded,
    # not pulled from a CDN
    resp = client.post("/run", json={"scene": valid_scene()})
    html = resp.json()["html"]
    # the real offline test: the library is inlined, not pulled from a CDN
    # <script src> (the bundled lib source itself may mention cdn.plot.ly for
    # optional topojson features, so a bare substring check is not enough)
    assert 'src="https://cdn.plot.ly' not in html
    assert "plotly.js v" in html  # the full runtime banner = inlined


def test_run_animate_flag_and_color(client):
    resp = client.post(
        "/run", json={"scene": valid_scene(), "animate": True, "color": "energy"}
    )
    assert resp.status_code == 200
    assert "plotly" in resp.json()["html"].lower()


def test_run_rejects_invalid_scene(client):
    scene = valid_scene()
    scene["solver"]["steps"] = -5
    resp = client.post("/run", json={"scene": scene})
    assert resp.status_code == 422
    assert resp.json()["ok"] is False


def test_run_reports_engine_absence_gracefully(client):
    # a matter_slab with no LATENT_DIRAC_G4_TRANSFORMER fails at run time; the
    # server must surface a 400 with a clear message, not a 500 stack trace
    scene = valid_scene()
    scene["elements"] = [
        {"type": "matter_slab", "label": "slab", "material": "G4_Al", "thickness_mm": 1.0},
        {"type": "monitor", "label": "end"},
    ]
    resp = client.post("/run", json={"scene": scene})
    assert resp.status_code == 400
    assert "LATENT_DIRAC_G4_TRANSFORMER" in resp.json()["detail"]
