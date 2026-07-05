import pytest

from latent_dirac.scene.build import run_scene
from latent_dirac.scene.loader import scene_from_mapping

SOURCE_PARAMS = {
    "primary_count": 10000,
    "yield_eplus_per_primary": 0.02,
    "mean_energy_MeV": 3.0,
    "energy_spread_MeV": 0.4,
    "angular_rms_rad": 0.08,
    "source_sigma_m": 0.003,
    "bunch_length_s": 1.0e-12,
    "macro_particles": 24,
}


def make_scene():
    return scene_from_mapping(
        {
            "schema_version": 1,
            "name": "render-line",
            "seed": 2026,
            "source": {"type": "positron_pair", "label": "pair-source", "params": SOURCE_PARAMS},
            "solver": {"type": "relativistic_boris", "dt_s": 3.0e-12, "steps": 30},
            "elements": [
                {
                    "type": "solenoid",
                    "label": "capture-solenoid",
                    "b_tesla": 0.8,
                    "radius_m": 0.05,
                    "length_m": 0.5,
                },
                {"type": "aperture", "label": "tight-collimator", "radius_m": 0.002, "z_m": 0.05},
                {"type": "monitor", "label": "end-station"},
            ],
        }
    )


def test_render_scene_3d_builds_figure_with_elements_and_trajectories():
    pytest.importorskip("plotly")
    from latent_dirac.viz.scene_3d import render_scene_3d

    scene = make_scene()
    run_result = run_scene(scene, record_trajectories=True)
    figure = render_scene_3d(scene, run_result)

    names = [trace.name for trace in figure.data]
    assert "capture-solenoid [solenoid]" in names
    assert "tight-collimator [aperture]" in names
    assert "end-station [monitor]" in names
    assert "trajectories" in names
    assert "accepted" in names
    assert "lost" in names


def test_render_scene_3d_carries_fidelity_labels():
    pytest.importorskip("plotly")
    from latent_dirac.viz.scene_3d import render_scene_3d

    scene = make_scene()
    run_result = run_scene(scene, record_trajectories=True)
    figure = render_scene_3d(scene, run_result)

    hover_texts = [str(trace.hovertext) for trace in figure.data if trace.hovertext is not None]
    assert any("hard-edge" in text for text in hover_texts)
    assert any("fidelity" in text for text in hover_texts)
    assert any("capture-solenoid [solenoid]" in text for text in hover_texts)


def test_render_scene_3d_respects_max_particles():
    pytest.importorskip("plotly")
    from latent_dirac.viz.scene_3d import render_scene_3d

    scene = make_scene()
    run_result = run_scene(scene, record_trajectories=True)
    figure = render_scene_3d(scene, run_result, max_particles=5)

    trajectory_trace = next(trace for trace in figure.data if trace.name == "trajectories")
    # 5 particles, separated by None gaps: total points = 5 * (T + 1)
    total_steps = sum(history.shape[0] - 1 for history in run_result.trajectories.values())
    assert len(trajectory_trace.x) == 5 * (total_steps + 2)


def test_scene_core_does_not_import_plotly():
    import subprocess
    import sys

    code = (
        "import latent_dirac.scene.build, latent_dirac.scene.loader, sys; "
        "assert 'plotly' not in sys.modules; print('ok')"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "ok"


def test_render_scene_3d_requires_recorded_run():
    pytest.importorskip("plotly")
    from latent_dirac.viz.scene_3d import render_scene_3d

    scene = make_scene()
    run_result = run_scene(scene, record_trajectories=False)

    figure = render_scene_3d(scene, run_result)
    names = [trace.name for trace in figure.data]
    assert "trajectories" not in names
    assert "accepted" in names
