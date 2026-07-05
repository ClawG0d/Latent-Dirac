import numpy as np

from latent_dirac.core.constants import ELEMENTARY_CHARGE_C, SPEED_OF_LIGHT_M_PER_S
from latent_dirac.core.species import positron
from latent_dirac.solvers.kernels import boris_step


def make_arrays(u_vectors: np.ndarray):
    count = u_vectors.shape[0]
    return (
        np.zeros((count, 3)),
        np.asarray(u_vectors, dtype=float),
        np.zeros(count),
        np.ones(count, dtype=bool),
    )


def test_u_magnitude_is_preserved_in_pure_magnetic_field():
    position, u, time_s, alive = make_arrays(np.array([[0.5, 0.0, 2.0], [0.0, 1.0, 0.1]]))
    b_field = np.broadcast_to(np.array([0.0, 0.0, 1.2]), (2, 3))
    e_field = np.zeros((2, 3))

    for _ in range(200):
        position, u, time_s = boris_step(
            position,
            u,
            time_s,
            alive,
            dt_s=1.0e-12,
            charge_c=positron.charge_c,
            mass_kg=positron.mass_kg,
            e_field=e_field,
            b_field=b_field,
        )

    np.testing.assert_allclose(
        np.linalg.norm(u, axis=1),
        np.linalg.norm([[0.5, 0.0, 2.0], [0.0, 1.0, 0.1]], axis=1),
        rtol=1e-12,
    )


def test_larmor_radius_matches_analytic_value():
    b_tesla = 0.8
    u_perp = 3.0  # dimensionless transverse momentum p/(m c)
    p_si = u_perp * positron.mass_kg * SPEED_OF_LIGHT_M_PER_S
    expected_radius = p_si / (ELEMENTARY_CHARGE_C * b_tesla)

    position, u, time_s, alive = make_arrays(np.array([[u_perp, 0.0, 0.0]]))
    b_field = np.array([[0.0, 0.0, b_tesla]])
    e_field = np.zeros((1, 3))

    gamma = np.sqrt(1.0 + u_perp**2)
    cyclotron_period = 2.0 * np.pi * gamma * positron.mass_kg / (ELEMENTARY_CHARGE_C * b_tesla)
    steps = 4000
    dt = cyclotron_period / steps

    for _ in range(steps):
        position, u, time_s = boris_step(
            position,
            u,
            time_s,
            alive,
            dt_s=dt,
            charge_c=positron.charge_c,
            mass_kg=positron.mass_kg,
            e_field=e_field,
            b_field=b_field,
        )

    # after one full turn the particle returns to the origin; mid-orbit
    # excursion equals the diameter, so track the maximum displacement
    position2, u2, time2, alive2 = make_arrays(np.array([[u_perp, 0.0, 0.0]]))
    max_displacement = 0.0
    for _ in range(steps // 2):
        position2, u2, time2 = boris_step(
            position2,
            u2,
            time2,
            alive2,
            dt_s=dt,
            charge_c=positron.charge_c,
            mass_kg=positron.mass_kg,
            e_field=e_field,
            b_field=b_field,
        )
        max_displacement = max(max_displacement, float(np.linalg.norm(position2[0, :2])))

    np.testing.assert_allclose(max_displacement, 2.0 * expected_radius, rtol=1e-4)
    # full-turn closure is limited by the Boris phase error, O((omega*dt)^2) per turn
    np.testing.assert_allclose(position[0], [0.0, 0.0, 0.0], atol=1e-5 * expected_radius)


def test_dead_particles_are_frozen():
    position, u, time_s, alive = make_arrays(np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    alive = np.array([True, False])
    e_field = np.broadcast_to(np.array([1.0e6, 0.0, 0.0]), (2, 3))
    b_field = np.zeros((2, 3))

    new_position, new_u, new_time = boris_step(
        position,
        u,
        time_s,
        alive,
        dt_s=1.0e-12,
        charge_c=positron.charge_c,
        mass_kg=positron.mass_kg,
        e_field=e_field,
        b_field=b_field,
    )

    assert new_u[0, 0] != 1.0  # alive particle was accelerated
    np.testing.assert_array_equal(new_u[1], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(new_position[1], position[1])
    assert new_time[1] == time_s[1]


def test_kernel_is_pure_and_does_not_mutate_inputs():
    position, u, time_s, alive = make_arrays(np.array([[0.2, 0.1, 1.5]]))
    position_before = position.copy()
    u_before = u.copy()

    boris_step(
        position,
        u,
        time_s,
        alive,
        dt_s=1.0e-12,
        charge_c=positron.charge_c,
        mass_kg=positron.mass_kg,
        e_field=np.full((1, 3), 1.0e5),
        b_field=np.full((1, 3), 0.4),
    )

    np.testing.assert_array_equal(position, position_before)
    np.testing.assert_array_equal(u, u_before)


def test_dimensionless_momentum_is_float32_safe_for_mev_particles():
    from latent_dirac.core.units import kinetic_energy_to_momentum_magnitude, mev_to_joule

    p_si = kinetic_energy_to_momentum_magnitude(mev_to_joule(3.0), positron.mass_kg)
    u_magnitude = p_si / (positron.mass_kg * SPEED_OF_LIGHT_M_PER_S)

    assert 1.0e-3 < u_magnitude < 1.0e3  # comfortably inside float32 normal range
    assert np.square(np.float32(u_magnitude)) > np.finfo(np.float32).tiny
