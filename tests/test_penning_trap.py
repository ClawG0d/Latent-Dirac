"""Analytic validation of the ideal Penning trap field and its dynamics."""

import numpy as np
import pytest

from latent_dirac.core.species import positron
from latent_dirac.fields.penning_trap import PenningTrapField
from latent_dirac.solvers.kernels import boris_step, momentum_to_dimensionless

V0 = 10.0
D_M = 0.005
B_T = 1.0


def make_trap() -> PenningTrapField:
    return PenningTrapField(v0_volt=V0, d_m=D_M, b_tesla=B_T)


def test_electric_field_matches_analytic_gradient():
    trap = make_trap()
    points = np.array([[0.001, -0.002, 0.003], [0.0, 0.0, 0.004], [-0.003, 0.001, 0.0]])

    expected = (V0 / D_M**2) * np.column_stack([0.5 * points[:, 0], 0.5 * points[:, 1], -points[:, 2]])
    np.testing.assert_allclose(trap.E(points, np.zeros(3)), expected, rtol=1e-12)

    b_field = trap.B(points, np.zeros(3))
    np.testing.assert_allclose(b_field, np.broadcast_to([0.0, 0.0, B_T], (3, 3)))


def test_divergence_of_e_is_zero():
    trap = make_trap()
    step = 1.0e-6
    point = np.array([0.001, 0.002, 0.003])

    divergence = 0.0
    for axis in range(3):
        plus = point.copy()
        minus = point.copy()
        plus[axis] += step
        minus[axis] -= step
        divergence += (trap.E(plus, 0.0)[axis] - trap.E(minus, 0.0)[axis]) / (2.0 * step)

    assert abs(divergence) < 1e-6 * V0 / D_M**2


def test_eigenfrequencies_satisfy_invariance_relations():
    trap = make_trap()
    omega_plus, omega_minus, omega_z = trap.eigenfrequencies(positron)

    omega_c = positron.charge_c * B_T / positron.mass_kg
    assert omega_plus + omega_minus == pytest.approx(omega_c, rel=1e-12)
    assert omega_plus * omega_minus == pytest.approx(0.5 * omega_z**2, rel=1e-12)
    assert omega_plus > omega_minus > 0.0


def test_eigenfrequencies_return_magnitudes_for_antiprotons():
    from latent_dirac.core.species import antiproton

    # q < 0 needs V0 < 0 for axial confinement; frequencies are magnitudes
    trap = PenningTrapField(v0_volt=-10.0, d_m=D_M, b_tesla=B_T)
    omega_plus, omega_minus, omega_z = trap.eigenfrequencies(antiproton)

    omega_c = abs(antiproton.charge_c) * B_T / antiproton.mass_kg
    assert omega_plus > omega_minus > 0.0
    assert omega_plus + omega_minus == pytest.approx(omega_c, rel=1e-12)
    assert omega_plus * omega_minus == pytest.approx(0.5 * omega_z**2, rel=1e-12)


def test_unstable_configuration_raises():
    # huge well, tiny field: omega_c^2 <= 2 omega_z^2
    trap = PenningTrapField(v0_volt=1.0e4, d_m=0.001, b_tesla=0.001)
    with pytest.raises(ValueError, match="unstable"):
        trap.eigenfrequencies(positron)


def _propagate(trap, position, momentum, dt_s, steps, record_every=1):
    u = momentum_to_dimensionless(momentum, positron.mass_kg)
    time_s = np.zeros(position.shape[0])
    alive = np.ones(position.shape[0], dtype=bool)
    history = [position.copy()]
    for step_index in range(steps):
        e_field = trap.E(position, time_s)
        b_field = trap.B(position, time_s)
        position, u, time_s = boris_step(
            position,
            u,
            time_s,
            alive,
            dt_s=dt_s,
            charge_c=positron.charge_c,
            mass_kg=positron.mass_kg,
            e_field=e_field,
            b_field=b_field,
        )
        if (step_index + 1) % record_every == 0:
            history.append(position.copy())
    return np.stack(history)


def test_on_axis_particle_oscillates_at_axial_frequency():
    trap = make_trap()
    _, _, omega_z = trap.eigenfrequencies(positron)
    period = 2.0 * np.pi / omega_z

    steps_per_period = 400
    dt = period / steps_per_period
    z0 = 0.001
    position = np.array([[0.0, 0.0, z0]])
    momentum = np.zeros((1, 3))

    history = _propagate(trap, position, momentum, dt, 2 * steps_per_period)
    z_track = history[:, 0, 2]

    # crossing times give the half-period
    signs = np.sign(z_track)
    crossings = np.nonzero(np.diff(signs) != 0)[0]
    assert crossings.size >= 2
    half_period_steps = np.mean(np.diff(crossings))
    measured_period = 2.0 * half_period_steps * dt

    assert measured_period == pytest.approx(period, rel=1e-3)
    # amplitude preserved (no numerical damping at this resolution)
    assert np.max(np.abs(z_track)) == pytest.approx(z0, rel=1e-3)


def test_pure_modified_cyclotron_launch_rotates_at_omega_plus():
    trap = make_trap()
    omega_plus, _, _ = trap.eigenfrequencies(positron)

    # circular mode: r x omega_plus tangential velocity suppresses the magnetron
    # mode; radius kept small so v/c ~ 0.01 and the non-relativistic
    # eigenfrequency applies within the tolerance
    radius = 2.0e-5
    velocity = radius * omega_plus
    position = np.array([[radius, 0.0, 0.0]])
    momentum = positron.mass_kg * np.array([[0.0, -velocity, 0.0]])  # sense set by physics

    period = 2.0 * np.pi / omega_plus
    steps_per_period = 600
    dt = period / steps_per_period
    history = _propagate(trap, position, momentum, dt, steps_per_period)

    radial = np.linalg.norm(history[:, 0, :2], axis=1)
    np.testing.assert_allclose(radial, radius, rtol=1e-2)

    # after one omega_plus period the particle returns near its start
    displacement = np.linalg.norm(history[-1, 0, :2] - history[0, 0, :2])
    assert displacement < 0.01 * radius


def test_trapped_particle_stays_bounded():
    trap = make_trap()
    _, _, omega_z = trap.eigenfrequencies(positron)
    period = 2.0 * np.pi / omega_z
    dt = period / 200

    rng = np.random.default_rng(7)
    position = rng.normal(0.0, 3.0e-4, size=(8, 3))
    # keep axial amplitudes v_z/omega_z well inside the trap scale d
    momentum = rng.normal(0.0, 2.0e-25, size=(8, 3))

    history = _propagate(trap, position, momentum, dt, 4000, record_every=20)

    assert np.max(np.abs(history[:, :, 2])) < D_M
    assert np.max(np.linalg.norm(history[:, :, :2], axis=2)) < D_M


def test_penning_trap_scene_element_matches_numpy_pipeline():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from latent_dirac.backends.jax_scene import run_scene_batched
    from latent_dirac.scene.build import run_scene
    from latent_dirac.scene.loader import scene_from_mapping

    scene = scene_from_mapping(
        {
            "schema_version": 1,
            "name": "trap-line",
            "seed": 2026,
            "source": {
                "type": "positron_pair",
                "label": "pair-source",
                "params": {
                    "primary_count": 1000,
                    "yield_eplus_per_primary": 0.02,
                    "mean_energy_MeV": 0.001,
                    "energy_spread_MeV": 0.0002,
                    "angular_rms_rad": 0.3,
                    "source_sigma_m": 0.0003,
                    "bunch_length_s": 1.0e-12,
                    "macro_particles": 24,
                },
            },
            "solver": {"type": "relativistic_boris", "dt_s": 5.0e-12, "steps": 200},
            "elements": [
                {
                    "type": "penning_trap",
                    "label": "trap",
                    "v0_volt": 10.0,
                    "d_m": 0.005,
                    "b_tesla": 1.0,
                },
                {"type": "monitor", "label": "end"},
            ],
        }
    )

    reference = run_scene(scene).pipeline_result.final_cloud
    batched = run_scene_batched(scene)

    np.testing.assert_allclose(batched.position_m[0], reference.position_m, rtol=1e-9, atol=1e-15)
    np.testing.assert_array_equal(batched.alive[0], reference.alive)
