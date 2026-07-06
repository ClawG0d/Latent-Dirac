"""Field-line polylines: faithful streamlines of the model fields.

Design record: docs/superpowers/specs/2026-07-06-field-line-rendering-design.md.
"""

from __future__ import annotations

import numpy as np

from latent_dirac.fields.penning_trap import PenningTrapField
from latent_dirac.fields.solenoid import SolenoidField, ThinSheetSolenoidField
from latent_dirac.fields.uniform import UniformField
from latent_dirac.viz.field_lines import element_field_line_bundles, field_line


def test_uniform_field_line_is_straight():
    field = UniformField(B_vector_t=np.array([0.0, 0.3, 0.4]))
    line = field_line(field, seed=(0.01, 0.0, 0.0), length_m=0.1, step_m=1e-3)
    assert line.shape[0] > 50
    chords = np.diff(line, axis=0)
    directions = chords / np.linalg.norm(chords, axis=1)[:, None]
    expected = np.array([0.0, 0.6, 0.8])
    assert np.allclose(directions, expected, atol=1e-12)


def test_thin_sheet_line_funnels_and_stays_tangent():
    field = ThinSheetSolenoidField(b_tesla=0.5, radius_m=0.03, length_m=0.2, center_z_m=0.1)
    seed = (0.015, 0.0, -0.05)  # off-axis, upstream in the fringe
    line = field_line(field, seed=seed, length_m=0.3, step_m=5e-4)
    radii = np.hypot(line[:, 0], line[:, 1])
    inside = (line[:, 2] > 0.05) & (line[:, 2] < 0.15)
    assert inside.any()
    # the fringe funnels the line radially inward by the time it is inside
    assert radii[inside].min() < 0.9 * radii[0]
    # tangency: each chord is parallel to the local B at the chord midpoint
    mids = 0.5 * (line[1:] + line[:-1])
    b = field.B(mids, 0.0)
    chords = np.diff(line, axis=0)
    cross = np.cross(chords, b)
    scale = np.linalg.norm(chords, axis=1) * np.linalg.norm(b, axis=1)
    assert float(np.max(np.linalg.norm(cross, axis=1) / scale)) < 5e-3


def test_penning_trap_e_line_conserves_invariant():
    field = PenningTrapField(v0_volt=50.0, d_m=0.005, b_tesla=0.05, center_z_m=0.0)
    line = field_line(field, seed=(0.002, 0.0, 0.001), length_m=0.01, step_m=2e-5, kind="E")
    r_sq = line[:, 0] ** 2 + line[:, 1] ** 2
    invariant = r_sq * np.abs(line[:, 2])
    # r^2 |z| is the analytic invariant of the quadrupole field-line ODE
    assert float(np.std(invariant) / np.mean(invariant)) < 2e-2


def test_hard_edge_line_stops_at_the_envelope():
    field = SolenoidField(b_tesla=0.5, radius_m=0.03, length_m=0.2, center_z_m=0.1)
    outside = field_line(field, seed=(0.0, 0.0, -0.05), length_m=0.1, step_m=1e-3)
    assert outside.shape[0] == 1  # zero field at the seed: no line
    inside = field_line(field, seed=(0.01, 0.0, 0.1), length_m=0.5, step_m=1e-3)
    assert float(inside[:, 2].max()) < 0.2 + 2e-3  # stops at the hard edge


def test_identical_field_elements_deduplicate():
    from latent_dirac.scene.loader import scene_from_mapping
    from latent_dirac.viz.field_lines import field_elements_for_lines

    trap = {"type": "penning_trap", "v0_volt": 50.0, "d_m": 0.005, "b_tesla": 0.05}
    scene = scene_from_mapping(
        {
            "schema_version": 1,
            "name": "dedupe",
            "seed": 1,
            "source": {
                "type": "cold_uniform_sphere",
                "label": "c",
                "params": {"species_name": "positron", "macro_particles": 4, "radius_m": 1e-3},
            },
            "solver": {"type": "relativistic_boris", "dt_s": 1e-12, "steps": 1},
            "elements": [
                {**trap, "label": "trap-1"},
                {
                    "type": "residual_gas_loss",
                    "label": "hold",
                    "mean_lifetime_s": 1.0,
                    "hold_time_s": 0.0,
                },
                {**trap, "label": "trap-2", "steps": 7},  # steps is stage bookkeeping
                {**trap, "label": "trap-3", "b_tesla": 0.1},  # different physics
            ],
        }
    )
    kept = [element.label for element in field_elements_for_lines(scene)]
    assert kept == ["trap-1", "trap-3"]


def test_render_scene_3d_carries_field_line_traces():
    import pytest as _pytest

    _pytest.importorskip("plotly")
    from pathlib import Path

    from latent_dirac.scene.build import run_scene
    from latent_dirac.scene.loader import load_scene
    from latent_dirac.viz.scene_3d import render_scene_3d

    scene = load_scene(Path("examples/scenes/hello_beamline.yaml"))
    figure = render_scene_3d(scene, run_scene(scene, record_trajectories=True))
    line_traces = [trace for trace in figure.data if "field lines" in (trace.name or "")]
    assert line_traces, "field-carrying scenes must render field-line traces"
    assert any("field lines of the model field" in (trace.hovertext or "") for trace in line_traces)


def test_bundles_cover_field_elements():
    from latent_dirac.scene.loader import scene_from_mapping

    scene = scene_from_mapping(
        {
            "schema_version": 1,
            "name": "bundle-check",
            "seed": 1,
            "source": {
                "type": "cold_uniform_sphere",
                "label": "c",
                "params": {"species_name": "positron", "macro_particles": 4, "radius_m": 1e-3},
            },
            "solver": {"type": "relativistic_boris", "dt_s": 1e-12, "steps": 1},
            "elements": [
                {
                    "type": "solenoid",
                    "label": "sol",
                    "b_tesla": 0.5,
                    "radius_m": 0.03,
                    "length_m": 0.2,
                    "center_z_m": 0.1,
                    "profile": "thin_sheet",
                },
                {
                    "type": "uniform_field",
                    "label": "wien",
                    "B_vector_t": [0.0, 0.05, 0.0],
                    "E_vector_v_m": [1.0e7, 0.0, 0.0],
                },
                {"type": "penning_trap", "label": "trap", "v0_volt": 50.0, "d_m": 0.005, "b_tesla": 0.05},
                {
                    "type": "quadrupole",
                    "label": "quad",
                    "gradient_t_m": 5.0,
                    "length_m": 0.1,
                    "center_z_m": 0.4,
                },
                {"type": "monitor", "label": "end"},
            ],
        }
    )
    from latent_dirac.scene.build import _field_for

    extent = {"transverse_m": 0.01, "axial_m": 0.1}
    for element in scene.elements[:-1]:
        bundles = element_field_line_bundles(element, _field_for(element), extent)
        assert bundles, f"{element.type} produced no field lines"
        kinds = {kind for kind, _ in bundles}
        assert kinds <= {"B", "E"}
        for _, line in bundles:
            assert line.ndim == 2 and line.shape[1] == 3 and line.shape[0] >= 2
    # the crossed Wien pair renders both families
    wien_kinds = {
        kind
        for kind, _ in element_field_line_bundles(scene.elements[1], _field_for(scene.elements[1]), extent)
    }
    assert wien_kinds == {"B", "E"}
    # the trap shows both confinement structures
    trap_kinds = {
        kind
        for kind, _ in element_field_line_bundles(scene.elements[2], _field_for(scene.elements[2]), extent)
    }
    assert trap_kinds == {"B", "E"}
