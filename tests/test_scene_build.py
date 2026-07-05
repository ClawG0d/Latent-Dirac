import numpy as np

from latent_dirac.core.units import momentum_gev_c_to_si
from latent_dirac.scene.build import build_source, run_scene
from latent_dirac.scene.loader import scene_from_mapping
from latent_dirac.sources.positron_pair import PositronPairSource

SOURCE_PARAMS = {
    "primary_count": 10000,
    "yield_eplus_per_primary": 0.02,
    "mean_energy_MeV": 3.0,
    "energy_spread_MeV": 0.4,
    "angular_rms_rad": 0.03,
    "source_sigma_m": 0.001,
    "bunch_length_s": 1.0e-12,
    "macro_particles": 32,
}


def make_scene_mapping(elements: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "name": "test-line",
        "seed": 2026,
        "source": {"type": "positron_pair", "label": "pair-source", "params": dict(SOURCE_PARAMS)},
        "solver": {"type": "relativistic_boris", "dt_s": 2.0e-12, "steps": 20},
        "elements": elements,
    }


def test_build_source_constructs_configured_source():
    scene = scene_from_mapping(make_scene_mapping([]))
    source = build_source(scene)

    assert isinstance(source, PositronPairSource)
    assert source.macro_particles == 32


def test_unknown_source_param_is_rejected():
    import pytest
    from pydantic import ValidationError

    mapping = make_scene_mapping([])
    mapping["source"]["params"]["typo_parameter"] = 42
    scene = scene_from_mapping(mapping)

    with pytest.raises(ValidationError):
        build_source(scene)


def test_stage_names_follow_scene_labels():
    scene = scene_from_mapping(
        make_scene_mapping(
            [
                {"type": "solenoid", "label": "capture", "b_tesla": 0.8, "radius_m": 0.05, "length_m": 0.5},
                {"type": "aperture", "label": "collimator", "radius_m": 0.04, "z_m": 0.06},
                {"type": "monitor", "label": "end-station"},
            ]
        )
    )
    result = run_scene(scene)

    assert [stage.stage_name for stage in result.pipeline_result.stage_results] == [
        "capture",
        "collimator",
        "end-station",
    ]


def test_transport_advances_particles():
    scene = scene_from_mapping(
        make_scene_mapping(
            [{"type": "solenoid", "label": "capture", "b_tesla": 0.8, "radius_m": 0.05, "length_m": 0.5}]
        )
    )
    result = run_scene(scene)
    final = result.pipeline_result.final_cloud

    assert float(np.mean(final.position_m[:, 2])) > 0.0
    assert float(np.mean(final.time_s)) > 0.0


def test_drift_preserves_kinetic_momentum_magnitude():
    scene = scene_from_mapping(make_scene_mapping([{"type": "drift", "label": "gap", "steps": 10}]))
    source_cloud = build_source(scene).sample(np.random.default_rng(scene.seed))
    result = run_scene(scene)
    final = result.pipeline_result.final_cloud

    np.testing.assert_allclose(
        np.linalg.norm(final.momentum_kg_m_s, axis=1),
        np.linalg.norm(source_cloud.momentum_kg_m_s, axis=1),
        rtol=1e-12,
    )
    assert float(np.mean(final.position_m[:, 2])) > 0.0


def test_monitor_records_snapshot_at_pipeline_position():
    scene = scene_from_mapping(
        make_scene_mapping(
            [
                {"type": "monitor", "label": "before"},
                {"type": "drift", "label": "gap", "steps": 10},
                {"type": "monitor", "label": "after"},
            ]
        )
    )
    result = run_scene(scene)

    assert set(result.monitors) == {"before", "after"}
    before = result.monitors["before"]
    after = result.monitors["after"]
    assert float(np.mean(after.position_m[:, 2])) > float(np.mean(before.position_m[:, 2]))
    np.testing.assert_allclose(after.position_m, result.pipeline_result.final_cloud.position_m)


def test_momentum_window_converts_gev_c():
    scene = scene_from_mapping(
        make_scene_mapping(
            [
                {
                    "type": "momentum_window",
                    "label": "cut",
                    "p_min_gev_c": 0.0035,
                    "p_max_gev_c": 0.0037,
                }
            ]
        )
    )
    source_cloud = build_source(scene).sample(np.random.default_rng(scene.seed))
    result = run_scene(scene)
    final = result.pipeline_result.final_cloud

    momentum = np.linalg.norm(source_cloud.momentum_kg_m_s, axis=1)
    expected_alive = (momentum >= momentum_gev_c_to_si(0.0035)) & (momentum <= momentum_gev_c_to_si(0.0037))
    np.testing.assert_array_equal(final.alive, source_cloud.alive & expected_alive)


def test_fixed_seed_is_deterministic():
    mapping = make_scene_mapping(
        [{"type": "solenoid", "label": "capture", "b_tesla": 0.8, "radius_m": 0.05, "length_m": 0.5}]
    )
    first = run_scene(scene_from_mapping(mapping))
    second = run_scene(scene_from_mapping(mapping))

    np.testing.assert_array_equal(
        first.pipeline_result.final_cloud.position_m,
        second.pipeline_result.final_cloud.position_m,
    )


def test_record_trajectories_returns_stepwise_positions():
    scene = scene_from_mapping(
        make_scene_mapping(
            [
                {
                    "type": "solenoid",
                    "label": "capture",
                    "b_tesla": 0.8,
                    "radius_m": 0.05,
                    "length_m": 0.5,
                    "steps": 8,
                },
                {"type": "drift", "label": "gap", "steps": 4},
            ]
        )
    )
    result = run_scene(scene, record_trajectories=True)

    assert set(result.trajectories) == {"capture", "gap"}
    capture = result.trajectories["capture"]
    assert capture.shape == (9, 32, 3)  # steps + 1 snapshots, N particles, xyz
    assert result.trajectories["gap"].shape == (5, 32, 3)
