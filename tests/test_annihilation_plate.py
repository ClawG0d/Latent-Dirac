"""Tests for the annihilation plate: ledgered endpoint + 2-photon kinematics."""

import numpy as np

from latent_dirac.scene.build import run_scene
from latent_dirac.scene.loader import load_scene, scene_from_mapping

SOURCE_PARAMS = {
    "primary_count": 10000,
    "yield_eplus_per_primary": 0.02,
    "mean_energy_MeV": 0.002,
    "energy_spread_MeV": 0.0003,
    "angular_rms_rad": 0.03,
    "source_sigma_m": 0.0008,
    "bunch_length_s": 1.0e-12,
    "macro_particles": 48,
}


def make_plate_scene(radius_m: float = 0.01) -> dict:
    return {
        "schema_version": 1,
        "name": "plate-line",
        "seed": 2026,
        "source": {"type": "positron_pair", "label": "kev-bunch", "params": dict(SOURCE_PARAMS)},
        "solver": {"type": "relativistic_boris", "dt_s": 4.0e-12, "steps": 300},
        "elements": [
            {"type": "uniform_field", "label": "guide", "B_vector_t": [0.0, 0.0, 0.4], "steps": 300},
            {"type": "annihilation_plate", "label": "plate", "z_m": 0.02, "radius_m": radius_m},
            {"type": "monitor", "label": "end"},
        ],
    }


def test_plate_kills_and_records_matching_event_count():
    scene = scene_from_mapping(make_plate_scene())
    result = run_scene(scene)

    final = result.pipeline_result.final_cloud
    plate_index = 1
    killed = int(np.sum(final.lost_at_element == plate_index))
    events = result.annihilations["plate"]

    assert killed > 0
    assert events["positions"].shape == (killed, 3)
    assert events["photon_directions"].shape == (killed, 2, 3)


def test_photon_pairs_are_back_to_back_unit_vectors():
    scene = scene_from_mapping(make_plate_scene())
    result = run_scene(scene)
    directions = result.annihilations["plate"]["photon_directions"]

    first, second = directions[:, 0, :], directions[:, 1, :]
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(second, axis=1), 1.0, rtol=1e-12)
    dots = np.sum(first * second, axis=1)
    np.testing.assert_allclose(dots, -1.0, rtol=1e-12)


def test_particles_outside_radius_pass_the_plate():
    scene = scene_from_mapping(make_plate_scene(radius_m=0.0005))
    result = run_scene(scene)

    final = result.pipeline_result.final_cloud
    # a tight plate lets the transverse halo pass (alive beyond the plane)
    assert final.weighted_count() > 0.0
    killed = int(np.sum(final.lost_at_element == 1))
    assert 0 < killed < 48


def test_events_appear_only_after_the_particle_crosses_the_plane():
    scene = scene_from_mapping(make_plate_scene())
    result = run_scene(scene)
    positions = result.annihilations["plate"]["positions"]

    # events sit at (or just past) the plane, not scattered along the line
    assert np.all(positions[:, 2] >= 0.02 - 1e-6)


def test_annihilation_demo_scene_runs():
    scene = load_scene("examples/scenes/annihilation_endpoint.yaml")
    result = run_scene(scene, record_trajectories=True)

    events = result.annihilations["annihilation-plate"]
    assert events["positions"].shape[0] > 0
