"""Tests for the xsuite_lattice scene element (T2).

Requires the optional [xsuite] extra; skipped when xtrack/xsuite are
missing. The Line JSON is generated in a temp file at test time.
"""

from __future__ import annotations

import numpy as np
import pytest

xt = pytest.importorskip("xtrack")
pytest.importorskip("xsuite")  # line.track() loads its prebuilt kernels

from latent_dirac.scene.build import run_scene  # noqa: E402
from latent_dirac.scene.loader import load_scene, scene_from_mapping  # noqa: E402

# reference momentum of a 2 MeV positron: pc = sqrt(Ek^2 + 2 Ek me c^2)
P0C_EV = 2.4585e6


def write_drift_line(path, length=1.0):
    line = xt.Line(elements=[xt.Drift(length=length)])
    line.to_json(str(path))
    return path


def write_aperture_line(path, length=0.5, half=2e-3):
    line = xt.Line(elements=[xt.Drift(length=length), xt.LimitRect(min_x=-half, max_x=half)])
    line.to_json(str(path))
    return path


def _forward_pair_source(count, source_sigma_m=1e-4):
    # a small-angle forward positron beam (p_z > 0 for all, as the adapter
    # requires); mean energy 2 MeV matches P0C_EV
    return {
        "type": "positron_pair",
        "label": "cloud",
        "params": {
            "primary_count": 10000,
            "yield_eplus_per_primary": 0.02,
            "mean_energy_MeV": 2.0,
            "energy_spread_MeV": 0.05,
            "angular_rms_rad": 0.01,
            "source_sigma_m": source_sigma_m,
            "bunch_length_s": 1.0e-12,
            "macro_particles": count,
        },
    }


def lattice_scene(line_path, count=8, p0c_ev=P0C_EV, num_turns=1, source_sigma_m=1e-4):
    return {
        "count": count,
        "scene": scene_from_mapping(
            {
                "schema_version": 1,
                "name": "xsuite-lattice",
                "seed": 3,
                "solver": {"dt_s": 1e-12, "steps": 1},
                "source": _forward_pair_source(count, source_sigma_m),
                "elements": [
                    {
                        "type": "xsuite_lattice",
                        "label": "transfer-line",
                        "line_path": str(line_path),
                        "p0c_ev": p0c_ev,
                        "num_turns": num_turns,
                        "center_z_m": 0.5,
                        "length_m": 1.0,
                    },
                    {"type": "monitor", "label": "end"},
                ],
            }
        ),
    }


def test_lattice_transports_the_cloud(tmp_path):
    line_path = write_drift_line(tmp_path / "line.json", length=1.0)
    setup = lattice_scene(line_path)
    result = run_scene(setup["scene"])
    final = result.pipeline_result.final_cloud
    # a drift advances z; survivors move downstream
    assert final.position_m[final.alive][:, 2].max() > 0.5
    assert final.alive.all()  # a bare drift kills nobody


def test_aperture_in_line_stamps_the_ledger(tmp_path):
    line_path = write_aperture_line(tmp_path / "ap.json", half=2e-4)
    setup = lattice_scene(line_path, count=12, source_sigma_m=1e-3)
    result = run_scene(setup["scene"])
    final = result.pipeline_result.final_cloud
    killed = ~final.alive
    assert killed.any() and not killed.all()  # the LimitRect cut some, not all
    assert np.all(final.lost_at_element[killed] == 0)  # lattice is stage 0
    assert np.all(final.lost_at_element[final.alive] == -1)


def test_line_path_resolves_relative_to_scene_file(tmp_path, monkeypatch):
    write_drift_line(tmp_path / "line.json", length=1.0)
    scene_yaml = tmp_path / "scene.yaml"
    scene_yaml.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "name: rel-path",
                "seed: 1",
                "solver: { dt_s: 1.0e-12, steps: 1 }",
                "source:",
                "  type: positron_pair",
                "  label: cloud",
                "  params: { primary_count: 10000, yield_eplus_per_primary: 0.02,"
                " mean_energy_MeV: 2.0, energy_spread_MeV: 0.05, angular_rms_rad: 0.01,"
                " source_sigma_m: 1.0e-4, bunch_length_s: 1.0e-12, macro_particles: 4 }",
                "elements:",
                "  - { type: xsuite_lattice, label: line, line_path: line.json,"
                f" p0c_ev: {P0C_EV} }}",
                "  - { type: monitor, label: end }",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path.parent)  # run from a different cwd
    scene = load_scene(scene_yaml)
    result = run_scene(scene)  # resolves line.json next to scene.yaml, not cwd
    assert result.pipeline_result.final_cloud.alive.all()


def test_missing_line_file_errors_at_run_not_construction(tmp_path):
    setup = lattice_scene(tmp_path / "does-not-exist.json")
    scene = setup["scene"]  # construction succeeds
    assert scene is not None
    with pytest.raises((FileNotFoundError, OSError, ValueError)):
        run_scene(scene)


def test_schema_rejects_bad_params(tmp_path):
    from pydantic import ValidationError

    line_path = write_drift_line(tmp_path / "line.json")
    with pytest.raises(ValidationError):
        lattice_scene(line_path, p0c_ev=0.0)  # p0c_ev must be > 0
    with pytest.raises(ValidationError):
        lattice_scene(line_path, num_turns=0)  # num_turns must be >= 1


def test_jax_backend_rejects_the_element(tmp_path):
    pytest.importorskip("jax")
    from latent_dirac.backends.jax_scene import run_scene_batched

    setup = lattice_scene(write_drift_line(tmp_path / "line.json"))
    with pytest.raises(ValueError, match="JAX backend"):
        run_scene_batched(setup["scene"], overrides={})


def test_scene_3d_draws_the_lattice(tmp_path):
    from latent_dirac.viz.scene_3d import FIDELITY_LABELS, _element_segments

    setup = lattice_scene(write_drift_line(tmp_path / "line.json"))
    lattice = setup["scene"].elements[0]
    assert lattice.type == "xsuite_lattice"
    assert _element_segments(lattice, None)  # a box is drawn
    assert "xsuite_lattice" in FIDELITY_LABELS
