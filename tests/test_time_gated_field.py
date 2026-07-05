"""Tests for the time-gated field wrapper and the deceleration-capture chain."""

import numpy as np
import pytest
from pydantic import ValidationError

from latent_dirac.fields.time_gated import TimeGatedField
from latent_dirac.fields.uniform import UniformField


def make_gated() -> TimeGatedField:
    inner = UniformField(
        E_vector_v_m=np.array([0.0, 0.0, -3.0e4]),
        B_vector_t=np.array([0.0, 0.0, 0.1]),
    )
    return TimeGatedField(inner=inner, t_on_s=1.0e-9, t_off_s=5.0e-9)


def test_gate_is_applied_per_particle_time():
    gated = make_gated()
    positions = np.zeros((3, 3))
    times = np.array([0.5e-9, 2.0e-9, 6.0e-9])  # before, inside, after

    e_field = gated.E(positions, times)
    b_field = gated.B(positions, times)

    np.testing.assert_allclose(e_field[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(e_field[1], [0.0, 0.0, -3.0e4])
    np.testing.assert_allclose(e_field[2], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(b_field[1], [0.0, 0.0, 0.1])
    np.testing.assert_allclose(b_field[2], [0.0, 0.0, 0.0])


def test_gate_boundaries_are_half_open():
    gated = make_gated()
    positions = np.zeros((2, 3))
    times = np.array([1.0e-9, 5.0e-9])  # t_on inclusive, t_off exclusive

    e_field = gated.E(positions, times)
    assert e_field[0, 2] == pytest.approx(-3.0e4)
    assert e_field[1, 2] == 0.0


def test_single_position_contract():
    gated = make_gated()
    inside = gated.E(np.zeros(3), 2.0e-9)
    outside = gated.E(np.zeros(3), 0.0)

    assert inside.shape == (3,)
    assert inside[2] == pytest.approx(-3.0e4)
    np.testing.assert_allclose(outside, np.zeros(3))


def test_invalid_window_is_rejected():
    inner = UniformField()
    with pytest.raises(ValidationError):
        TimeGatedField(inner=inner, t_on_s=5.0e-9, t_off_s=1.0e-9)


def test_retarding_field_energy_loss_is_analytic():
    """A positron climbing a uniform retarding E field loses qE * dz."""

    from latent_dirac.core.species import positron
    from latent_dirac.solvers.kernels import (
        boris_step,
        dimensionless_to_momentum,
        momentum_to_dimensionless,
    )

    e_z = -2.0e4  # opposes +z motion for a positron
    field_e = np.array([[0.0, 0.0, e_z]])
    field_b = np.zeros((1, 3))

    kinetic_j = 2000.0 * 1.602176634e-19
    momentum = np.array([[0.0, 0.0, np.sqrt(2.0 * positron.mass_kg * kinetic_j)]])
    position = np.zeros((1, 3))
    u = momentum_to_dimensionless(momentum, positron.mass_kg)
    time_s = np.zeros(1)
    alive = np.ones(1, dtype=bool)

    dt = 2.0e-12
    for _ in range(1200):
        position, u, time_s = boris_step(
            position,
            u,
            time_s,
            alive,
            dt_s=dt,
            charge_c=positron.charge_c,
            mass_kg=positron.mass_kg,
            e_field=field_e,
            b_field=field_b,
        )

    final_momentum = dimensionless_to_momentum(u, positron.mass_kg)
    final_kinetic = float(np.linalg.norm(final_momentum) ** 2 / (2.0 * positron.mass_kg))
    expected = kinetic_j + positron.charge_c * e_z * float(position[0, 2])

    assert final_kinetic == pytest.approx(expected, rel=5e-3)
    assert final_kinetic < kinetic_j  # it actually decelerated


def test_gated_scene_parity_between_numpy_and_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from latent_dirac.backends.jax_scene import run_scene_batched
    from latent_dirac.scene.build import run_scene
    from latent_dirac.scene.loader import scene_from_mapping

    scene = scene_from_mapping(
        {
            "schema_version": 1,
            "name": "gated-line",
            "seed": 2026,
            "source": {
                "type": "positron_pair",
                "label": "pair-source",
                "params": {
                    "primary_count": 1000,
                    "yield_eplus_per_primary": 0.02,
                    "mean_energy_MeV": 0.002,
                    "energy_spread_MeV": 0.0004,
                    "angular_rms_rad": 0.05,
                    "source_sigma_m": 0.0005,
                    "bunch_length_s": 1.0e-12,
                    "macro_particles": 24,
                },
            },
            "solver": {"type": "relativistic_boris", "dt_s": 4.0e-12, "steps": 120},
            "elements": [
                {
                    "type": "uniform_field",
                    "label": "gated-barrier",
                    "E_vector_v_m": [0.0, 0.0, -2.0e4],
                    "t_on_s": 1.0e-10,
                    "t_off_s": 1.0e-6,
                    "steps": 120,
                },
                {"type": "monitor", "label": "end"},
            ],
        }
    )

    reference = run_scene(scene).pipeline_result.final_cloud
    batched = run_scene_batched(scene)

    np.testing.assert_allclose(batched.position_m[0], reference.position_m, rtol=1e-9, atol=1e-18)
    np.testing.assert_allclose(batched.momentum_kg_m_s[0], reference.momentum_kg_m_s, rtol=1e-9, atol=1e-30)


def test_gate_window_must_come_in_pairs_in_the_schema():
    from latent_dirac.scene.loader import scene_from_mapping

    with pytest.raises(ValidationError, match="t_on_s"):
        scene_from_mapping(
            {
                "schema_version": 1,
                "name": "bad-gate",
                "seed": 1,
                "source": {
                    "type": "positron_pair",
                    "label": "s",
                    "params": {
                        "primary_count": 100,
                        "yield_eplus_per_primary": 0.02,
                        "mean_energy_MeV": 1.0,
                        "energy_spread_MeV": 0.1,
                        "angular_rms_rad": 0.01,
                        "source_sigma_m": 0.001,
                        "bunch_length_s": 1.0e-12,
                        "macro_particles": 8,
                    },
                },
                "solver": {"type": "relativistic_boris", "dt_s": 1.0e-12, "steps": 5},
                "elements": [
                    {
                        "type": "uniform_field",
                        "label": "half-gate",
                        "E_vector_v_m": [0.0, 0.0, -1.0e4],
                        "t_on_s": 1.0e-9,
                    }
                ],
            }
        )


def test_decel_capture_scene_traps_after_gate_closes():
    from latent_dirac.scene.build import run_scene
    from latent_dirac.scene.loader import load_scene

    scene = load_scene("examples/scenes/decel_capture.yaml")
    result = run_scene(scene, record_trajectories=True)

    final = result.pipeline_result.final_cloud
    assert final.weighted_count() > 0.0

    # captured bunch stays bounded around the trap center (0.09 m)
    z_final = final.position_m[final.alive, 2]
    assert z_final.size > 0
    assert np.all((z_final > 0.05) & (z_final < 0.13))

    # after the gate closes, every particle reverses axial direction at
    # least twice (it is bouncing in the well, not drifting through)
    trap_z = result.trajectories["capture-trap"][:, :, 2]
    dz_sign = np.sign(np.diff(trap_z, axis=0))
    reversals = np.sum(np.abs(np.diff(dz_sign, axis=0)) > 0, axis=0)
    assert np.all(reversals >= 2)
