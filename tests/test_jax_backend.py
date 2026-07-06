"""Parity and batching tests for the JAX scene backend.

Validation mode per the positioning spec: x64 enabled, element-wise
comparison against the NumPy float64 reference pipeline.
"""

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from latent_dirac.backends.jax_scene import run_scene_batched  # noqa: E402
from latent_dirac.scene.build import run_scene  # noqa: E402
from latent_dirac.scene.loader import load_scene, scene_from_mapping  # noqa: E402

SCENES_DIR = Path("examples/scenes")

SOURCE_PARAMS = {
    "primary_count": 10000,
    "yield_eplus_per_primary": 0.02,
    "mean_energy_MeV": 3.0,
    "energy_spread_MeV": 0.4,
    "angular_rms_rad": 0.03,
    "source_sigma_m": 0.001,
    "bunch_length_s": 1.0e-12,
    "macro_particles": 48,
}


def sweep_scene_mapping() -> dict:
    return {
        "schema_version": 1,
        "name": "sweep-line",
        "seed": 2031,
        "source": {"type": "positron_pair", "label": "pair-source", "params": dict(SOURCE_PARAMS)},
        "solver": {"type": "relativistic_boris", "dt_s": 2.0e-12, "steps": 60},
        "elements": [
            {"type": "uniform_field", "label": "sweep-field", "B_vector_t": [0.0, 0.3, 0.0], "steps": 60},
            {"type": "aperture", "label": "window", "radius_m": 0.02, "z_m": 0.03},
            {"type": "momentum_window", "label": "cut", "p_min_gev_c": 0.003, "p_max_gev_c": 0.0045},
            {"type": "monitor", "label": "end"},
        ],
    }


@pytest.mark.parametrize("scene_name", ["antiproton_ledger.yaml", "dipole_quad_line.yaml"])
def test_single_config_parity_with_numpy_pipeline(scene_name):
    scene = load_scene(SCENES_DIR / scene_name)

    reference = run_scene(scene).pipeline_result.final_cloud
    batched = run_scene_batched(scene, overrides={})

    assert batched.alive.shape[0] == 1
    np.testing.assert_array_equal(batched.alive[0], reference.alive)
    np.testing.assert_array_equal(batched.lost_at_element[0], reference.lost_at_element)
    np.testing.assert_allclose(batched.position_m[0], reference.position_m, rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(batched.momentum_kg_m_s[0], reference.momentum_kg_m_s, rtol=1e-9, atol=1e-30)
    np.testing.assert_allclose(batched.accepted_weighted[0], reference.weighted_count(), rtol=1e-12)


def test_uniform_field_sweep_matches_per_config_numpy_loop():
    mapping = sweep_scene_mapping()
    scene = scene_from_mapping(mapping)
    by_values = np.array([0.0, 0.15, 0.3, 0.45, 0.6])
    b_vectors = np.stack([np.zeros_like(by_values), by_values, np.zeros_like(by_values)], axis=1)

    batched = run_scene_batched(scene, overrides={"sweep-field.B_vector_t": b_vectors})

    assert batched.alive.shape == (5, SOURCE_PARAMS["macro_particles"])
    for index, by_tesla in enumerate(by_values):
        config = sweep_scene_mapping()
        config["elements"][0]["B_vector_t"] = [0.0, float(by_tesla), 0.0]
        reference = run_scene(scene_from_mapping(config)).pipeline_result.final_cloud
        np.testing.assert_array_equal(batched.alive[index], reference.alive)
        np.testing.assert_allclose(batched.position_m[index], reference.position_m, rtol=1e-9, atol=1e-15)
        np.testing.assert_allclose(
            batched.accepted_fraction[index],
            reference.weighted_count() / float(np.sum(reference.weight)),
            rtol=1e-12,
        )


def test_scalar_parameter_sweep_over_solenoid_field():
    # hello_beamline: positron_capture now ends at an annihilation plate,
    # which the JAX backend rejects by design; hello also exercises the
    # thin_sheet profile path
    scene = load_scene(SCENES_DIR / "hello_beamline.yaml")
    b_values = np.array([0.4, 0.8, 1.2])

    batched = run_scene_batched(scene, overrides={"hello-solenoid.b_tesla": b_values})

    assert batched.accepted_fraction.shape == (3,)
    reference = run_scene(scene).pipeline_result.final_cloud
    np.testing.assert_array_equal(batched.alive[1], reference.alive)


def test_monitor_before_acceptance_keeps_ledger_indices_aligned():
    mapping = sweep_scene_mapping()
    mapping["elements"] = [
        mapping["elements"][0],
        {"type": "monitor", "label": "mid-monitor"},
        *mapping["elements"][1:],
    ]
    scene = scene_from_mapping(mapping)

    reference = run_scene(scene).pipeline_result.final_cloud
    batched = run_scene_batched(scene)

    np.testing.assert_array_equal(batched.lost_at_element[0], reference.lost_at_element)
    np.testing.assert_array_equal(batched.alive[0], reference.alive)


def test_batched_acceptance_parameter_matches_numpy_loop():
    scene = scene_from_mapping(sweep_scene_mapping())
    p_min_values = np.array([0.001, 0.003, 0.004])

    batched = run_scene_batched(scene, overrides={"cut.p_min_gev_c": p_min_values})

    for index, p_min in enumerate(p_min_values):
        config = sweep_scene_mapping()
        config["elements"][2]["p_min_gev_c"] = float(p_min)
        reference = run_scene(scene_from_mapping(config)).pipeline_result.final_cloud
        np.testing.assert_array_equal(batched.alive[index], reference.alive)
        np.testing.assert_array_equal(batched.lost_at_element[index], reference.lost_at_element)


def test_overrides_default_to_single_configuration():
    scene = scene_from_mapping(sweep_scene_mapping())
    result = run_scene_batched(scene)

    assert result.alive.shape[0] == 1


def test_empty_batch_is_rejected():
    scene = scene_from_mapping(sweep_scene_mapping())
    with pytest.raises(ValueError, match="batch"):
        run_scene_batched(scene, overrides={"cut.p_min_gev_c": np.zeros(0)})


def test_unknown_override_label_raises():
    scene = scene_from_mapping(sweep_scene_mapping())
    with pytest.raises(ValueError, match="no-such-element"):
        run_scene_batched(scene, overrides={"no-such-element.b_tesla": np.array([0.1])})


def test_unknown_override_param_raises():
    scene = scene_from_mapping(sweep_scene_mapping())
    with pytest.raises(ValueError, match="warp_factor"):
        run_scene_batched(scene, overrides={"sweep-field.warp_factor": np.array([0.1])})


def test_mismatched_batch_sizes_raise():
    scene = scene_from_mapping(sweep_scene_mapping())
    with pytest.raises(ValueError, match="batch"):
        run_scene_batched(
            scene,
            overrides={
                "sweep-field.B_vector_t": np.zeros((3, 3)),
                "cut.p_min_gev_c": np.zeros(4),
            },
        )


def test_kernel_is_xp_generic():
    import jax.numpy as jnp

    from latent_dirac.core.species import positron
    from latent_dirac.solvers.kernels import boris_step

    position = np.zeros((4, 3))
    u = np.full((4, 3), 0.5)
    time_s = np.zeros(4)
    alive = np.ones(4, dtype=bool)
    e_field = np.full((4, 3), 1.0e5)
    b_field = np.full((4, 3), 0.4)
    kwargs = {
        "dt_s": 1.0e-12,
        "charge_c": positron.charge_c,
        "mass_kg": positron.mass_kg,
    }

    np_result = boris_step(position, u, time_s, alive, e_field=e_field, b_field=b_field, **kwargs)
    jax_result = boris_step(
        jnp.asarray(position),
        jnp.asarray(u),
        jnp.asarray(time_s),
        jnp.asarray(alive),
        e_field=jnp.asarray(e_field),
        b_field=jnp.asarray(b_field),
        xp=jnp,
        **kwargs,
    )

    for np_array, jax_array in zip(np_result, jax_result, strict=True):
        np.testing.assert_allclose(np.asarray(jax_array), np_array, rtol=1e-14)
