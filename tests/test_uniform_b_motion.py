import numpy as np

from latent_dirac.core.constants import e
from latent_dirac.core.species import positron
from latent_dirac.core.units import gamma_from_momentum, momentum_gev_c_to_si
from latent_dirac.fields.uniform import UniformField
from latent_dirac.solvers.relativistic_boris import RelativisticBorisSolver
from latent_dirac.state.particle_cloud import ParticleCloud


def test_relativistic_boris_preserves_energy_in_uniform_magnetic_field():
    b_tesla = 0.25
    p_mag = momentum_gev_c_to_si(0.01)
    gamma = gamma_from_momentum(p_mag, positron.mass_kg)
    period_s = 2.0 * np.pi * gamma * positron.mass_kg / (e * b_tesla)
    steps = 800

    cloud = ParticleCloud(
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

    propagated = RelativisticBorisSolver(dt_s=period_s / steps, steps=steps).propagate(
        cloud,
        UniformField(B_vector_t=np.array([0.0, 0.0, b_tesla])),
    )

    initial_energy = cloud.kinetic_energy_joule()[0]
    final_energy = propagated.kinetic_energy_joule()[0]
    assert np.isclose(final_energy, initial_energy, rtol=1.0e-10, atol=0.0)
