"""Tests for the animated interactive 3D viewer (Plotly-first, slice 1)."""

from __future__ import annotations

import pytest

pytest.importorskip("plotly")

from latent_dirac.scene.build import run_scene
from latent_dirac.scene.loader import scene_from_mapping
from latent_dirac.viz.scene_3d import render_scene_animation


def _capture_scene(steps=12, macro_particles=20):
    return scene_from_mapping(
        {
            "schema_version": 1,
            "name": "anim-scene",
            "seed": 7,
            "solver": {"dt_s": 4.0e-12, "steps": steps},
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
                    "macro_particles": macro_particles,
                },
            },
            "elements": [
                {
                    "type": "solenoid",
                    "label": "capture",
                    "b_tesla": 0.6,
                    "radius_m": 0.03,
                    "length_m": 0.2,
                    "center_z_m": 0.1,
                },
                {"type": "aperture", "label": "collimator", "radius_m": 0.01, "z_m": 0.1},
                {"type": "monitor", "label": "end"},
            ],
        }
    )


def _run(steps=12, macro_particles=20):
    scene = _capture_scene(steps, macro_particles)
    result = run_scene(scene, record_trajectories=True)
    return scene, result


def _combined_step_count(result):
    # mirror _combined_trajectories: first element full, later elements drop
    # their shared first row
    parts = [h for h in result.trajectories.values()]
    return parts[0].shape[0] + sum(p.shape[0] - 1 for p in parts[1:])


def test_frames_match_recorded_step_count():
    scene, result = _run(steps=12)
    figure = render_scene_animation(scene, result)
    assert len(figure.frames) == _combined_step_count(result)


def test_multi_recording_element_concat():
    # two transport elements both record trajectories, exercising the
    # multi-part concat (later element drops its shared first row)
    scene = scene_from_mapping(
        {
            "schema_version": 1,
            "name": "two-stage-anim",
            "seed": 7,
            "solver": {"dt_s": 4.0e-12, "steps": 10},
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
                    "macro_particles": 12,
                },
            },
            "elements": [
                {"type": "solenoid", "label": "cap", "b_tesla": 0.6, "radius_m": 0.03,
                 "length_m": 0.2, "center_z_m": 0.1},
                {"type": "drift", "label": "gap", "steps": 6},
                {"type": "monitor", "label": "end"},
            ],
        }
    )
    result = run_scene(scene, record_trajectories=True)
    assert len(result.trajectories) == 2  # solenoid + drift both recorded
    figure = render_scene_animation(scene, result)
    # 11 (solenoid: steps+1) + 6 (drift: steps, shared first row dropped) = 17
    assert len(figure.frames) == _combined_step_count(result)
    assert len(figure.frames) == 17


def test_moving_cloud_has_min_particle_count():
    scene, result = _run(macro_particles=20)
    figure = render_scene_animation(scene, result, max_particles=8)
    # the last data trace is the moving cloud at step 0; frames update it
    cloud0 = figure.data[-1]
    assert len(cloud0.x) == 8
    frame_cloud = figure.frames[-1].data[0]
    assert len(frame_cloud.x) == 8


def test_layout_has_animation_controls():
    scene, result = _run()
    figure = render_scene_animation(scene, result)
    updatemenus = figure.layout.updatemenus
    assert updatemenus  # play/pause menu present
    labels = [b.label for m in updatemenus for b in m.buttons]
    assert "Play" in labels
    assert figure.layout.sliders  # scrub slider present


def test_trail_toggles_the_faint_paths():
    scene, result = _run()
    with_trail = render_scene_animation(scene, result, trail=True)
    without_trail = render_scene_animation(scene, result, trail=False)
    names_with = [t.name for t in with_trail.data]
    names_without = [t.name for t in without_trail.data]
    assert "trajectories" in names_with
    assert "trajectories" not in names_without


def test_element_wireframes_are_static_traces():
    scene, result = _run()
    figure = render_scene_animation(scene, result)
    names = [t.name for t in figure.data if t.name]
    assert any("capture" in name for name in names)  # solenoid wire present


def test_missing_trajectories_raises():
    scene = _capture_scene()
    result = run_scene(scene)  # no record_trajectories
    with pytest.raises(ValueError, match="record_trajectories"):
        render_scene_animation(scene, result)


def test_nonpositive_max_particles_raises():
    scene, result = _run()
    with pytest.raises(ValueError):
        render_scene_animation(scene, result, max_particles=0)


def test_fate_coloring_maps_accepted_and_lost():
    from latent_dirac.viz.scene_3d import _ACCEPTED_COLOR, _LOST_COLOR

    scene, result = _run()
    figure = render_scene_animation(scene, result, color="fate")
    colors = list(figure.data[-1].marker.color)
    final = result.pipeline_result.final_cloud
    count = len(colors)
    expected = [_ACCEPTED_COLOR if a else _LOST_COLOR for a in final.alive[:count]]
    assert colors == expected
    # color is static across frames
    assert list(figure.frames[0].data[0].marker.color) == expected


def test_energy_coloring_is_numeric_with_colorbar():
    scene, result = _run()
    figure = render_scene_animation(scene, result, color="energy")
    marker = figure.data[-1].marker
    assert marker.colorscale is not None
    assert marker.showscale
    assert len(marker.color) == min(20, 64)  # numeric KE array, one per particle


def test_ledger_coloring_accepted_are_green():
    from latent_dirac.viz.scene_3d import _ACCEPTED_COLOR

    scene, result = _run()
    figure = render_scene_animation(scene, result, color="ledger")
    colors = list(figure.data[-1].marker.color)
    final = result.pipeline_result.final_cloud
    for c, alive in zip(colors, final.alive[: len(colors)], strict=True):
        if alive:
            assert c == _ACCEPTED_COLOR


def test_color_none_is_uniform():
    scene, result = _run()
    figure = render_scene_animation(scene, result, color="none")
    assert figure.data[-1].marker.color is None


def test_invalid_color_raises():
    scene, result = _run()
    with pytest.raises(ValueError, match="color must be one of"):
        render_scene_animation(scene, result, color="rainbow")


def test_writes_interactive_html(tmp_path):
    scene, result = _run()
    figure = render_scene_animation(scene, result)
    out = tmp_path / "anim.html"
    figure.write_html(str(out))
    text = out.read_text(encoding="utf-8")
    assert "plotly" in text.lower()
    assert len(text) > 0
