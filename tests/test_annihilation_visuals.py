"""Annihilation demo visuals: event channels, species guard, rendering.

Design record: docs/superpowers/specs/2026-07-06-annihilation-demo-visuals-design.md.
Visualization only — at-rest two-photon kinematics, 511 keV as a label,
no energetics (safety scope).
"""

from __future__ import annotations

import numpy as np
import pytest

from latent_dirac.scene.build import run_scene
from latent_dirac.scene.loader import scene_from_mapping

POSITRON_SOURCE = {
    "type": "positron_pair",
    "label": "src",
    "params": {
        "primary_count": 10000,
        "yield_eplus_per_primary": 0.02,
        "mean_energy_MeV": 0.002,
        "energy_spread_MeV": 0.0003,
        "angular_rms_rad": 0.04,
        "source_sigma_m": 0.001,
        "bunch_length_s": 1.0e-12,
        "macro_particles": 24,
    },
}


def plate_scene(source=None):
    return scene_from_mapping(
        {
            "schema_version": 1,
            "name": "plate-line",
            "seed": 2026,
            "source": source or dict(POSITRON_SOURCE),
            "solver": {"type": "relativistic_boris", "dt_s": 4.0e-12, "steps": 250},
            "elements": [
                {"type": "uniform_field", "label": "guide", "B_vector_t": [0.0, 0.0, 0.4], "steps": 250},
                {"type": "annihilation_plate", "label": "plate", "z_m": 0.024, "radius_m": 0.012},
            ],
        }
    )


def test_events_record_time_and_particle_id():
    result = run_scene(plate_scene())
    events = result.annihilations["plate"]
    count = events["positions"].shape[0]
    assert count > 0
    assert events["time_s"].shape == (count,)
    assert events["particle_id"].shape == (count,)

    final = result.pipeline_result.final_cloud
    killed = final.particle_id[~final.alive & (final.lost_at_element == 1)]
    np.testing.assert_array_equal(np.sort(events["particle_id"]), np.sort(killed))
    # events are projected back onto the plate plane: the stage-boundary
    # kill overshoots, so vertices must sit at z_m with a clock at or
    # before the stage-end clock
    np.testing.assert_allclose(events["positions"][:, 2], 0.024, rtol=0.0, atol=1e-12)
    by_id = {int(pid): float(t) for pid, t in zip(final.particle_id, final.time_s, strict=True)}
    for pid, t in zip(events["particle_id"], events["time_s"], strict=True):
        assert t <= by_id[int(pid)]


def test_species_guard_rejects_non_positrons():
    antiproton_source = {
        "type": "cold_uniform_sphere",
        "label": "src",
        "params": {
            "species_name": "antiproton",
            "macro_particles": 8,
            "radius_m": 1e-3,
            "weight": 1.0,
        },
    }
    with pytest.raises(ValueError, match="positron"):
        run_scene(plate_scene(source=antiproton_source))


def test_scene_3d_draws_plate_and_photons():
    pytest.importorskip("plotly")
    from latent_dirac.viz.scene_3d import _element_segments, render_scene_3d

    scene = plate_scene()
    plate = scene.elements[1]
    assert _element_segments(plate, None), "the plate must have a geometric representation"

    result = run_scene(scene, record_trajectories=True)
    figure = render_scene_3d(scene, result)
    photon_traces = [trace for trace in figure.data if "photon" in (trace.name or "")]
    assert photon_traces, "annihilation events must render photon rays"
    assert any("511 keV" in (trace.hovertext or "") for trace in photon_traces)


def test_scene_3d_no_photon_trace_without_events():
    pytest.importorskip("plotly")
    from latent_dirac.viz.scene_3d import render_scene_3d

    scene = scene_from_mapping(
        {
            "schema_version": 1,
            "name": "no-plate",
            "seed": 2026,
            "source": dict(POSITRON_SOURCE),
            "solver": {"type": "relativistic_boris", "dt_s": 4.0e-12, "steps": 20},
            "elements": [
                {"type": "uniform_field", "label": "guide", "B_vector_t": [0.0, 0.0, 0.4], "steps": 20},
            ],
        }
    )
    figure = render_scene_3d(scene, run_scene(scene, record_trajectories=True))
    assert not [trace for trace in figure.data if "photon" in (trace.name or "")]


def test_draw_photon_burst_progress_regimes():
    pytest.importorskip("matplotlib")
    from tools import mpl3d

    plt, _ = mpl3d.load_matplotlib()
    positions = np.array([[0.0, 0.0, 0.01], [0.001, 0.0, 0.01]])
    directions = np.stack([np.eye(3)[:2], -np.eye(3)[:2]], axis=1)  # (2, 2, 3)

    fig = plt.figure()
    axes = fig.add_subplot(projection="3d")
    baseline = len(axes.lines) + len(axes.collections)
    mpl3d.draw_photon_burst(axes, positions, directions, np.zeros(2), max_length=0.01)
    assert len(axes.lines) + len(axes.collections) == baseline  # zero progress draws nothing

    mpl3d.draw_photon_burst(axes, positions, directions, np.array([0.5, 1.0]), max_length=0.01)
    assert len(axes.lines) > 0  # rays drawn
    assert len(axes.collections) == 1  # flash only for the in-progress event
    plt.close(fig)


def test_report_field_model_matches_profile():
    from latent_dirac.diagnostics.scene_report import field_status_lines

    thin = scene_from_mapping(
        {
            "schema_version": 1,
            "name": "profile-report",
            "seed": 1,
            "source": dict(POSITRON_SOURCE),
            "solver": {"type": "relativistic_boris", "dt_s": 1e-12, "steps": 1},
            "elements": [
                {
                    "type": "solenoid",
                    "label": "sol",
                    "b_tesla": 0.5,
                    "radius_m": 0.03,
                    "length_m": 0.2,
                    "profile": "thin_sheet",
                }
            ],
        }
    )
    text = "\n".join(field_status_lines(thin))
    assert "thin-sheet" in text
    assert "hard-edge" not in text


def test_hero_scene_accepted_core_annihilates():
    from pathlib import Path

    from latent_dirac.scene.loader import load_scene

    scene = load_scene(Path("examples/scenes/positron_capture.yaml"))
    labels = [element.label for element in scene.elements]
    assert "collector-plate" in labels
    assert scene.elements[0].profile == "thin_sheet"

    result = run_scene(scene)
    final = result.pipeline_result.final_cloud
    assert final.weighted_count() == 0.0  # every accepted positron reached the plate
    events = result.annihilations["collector-plate"]
    assert events["positions"].shape[0] > 0
