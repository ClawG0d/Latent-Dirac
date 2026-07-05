import numpy as np

from latent_dirac.core.constants import e
from latent_dirac.core.species import positron
from latent_dirac.core.units import gamma_from_momentum, momentum_gev_c_to_si
from latent_dirac.fields.uniform import UniformField
from latent_dirac.solvers.relativistic_boris import RelativisticBorisSolver
from latent_dirac.state.particle_state import ParticleState


def test_uniform_b_motion_matches_larmor_radius_after_quarter_turn():
    b_tesla = 0.5
    p_mag = momentum_gev_c_to_si(0.01)
    gamma = gamma_from_momentum(p_mag, positron.mass_kg)
    larmor_radius_m = p_mag / (e * b_tesla)
    quarter_period_s = 0.5 * np.pi * gamma * positron.mass_kg / (e * b_tesla)
    steps = 1000

    cloud = ParticleState(
        species=positron,
        position_m=np.zeros((1, 3)),
        momentum_kg_m_s=np.array([[p_mag, 0.0, 0.0]]),
        time_s=np.zeros(1),
        weight=np.ones(1),
        alive=np.ones(1, dtype=bool),
        particle_id=np.array([0]),
        parent_id=np.array([-1]),
        metadata={},
    )

    propagated = RelativisticBorisSolver(
        dt_s=quarter_period_s / steps,
        steps=steps,
    ).propagate(cloud, UniformField(B_vector_t=np.array([0.0, 0.0, b_tesla])))

    assert np.isclose(propagated.position_m[0, 0], larmor_radius_m, rtol=3.0e-3)
    assert np.isclose(propagated.position_m[0, 1], -larmor_radius_m, rtol=3.0e-3)
