"""Simplified beta-plus positron source term.

The first version uses a documented approximate beta spectrum: a symmetric
Beta(3, 3) distribution scaled to the endpoint kinetic energy. It omits
nuclear, Coulomb/Fermi-function, material, and annihilation physics.
"""

from __future__ import annotations

import numpy as np
from pydantic import field_validator

from latent_dirac.core.species import positron
from latent_dirac.core.units import kinetic_energy_to_momentum_magnitude, mev_to_joule
from latent_dirac.sources.base import (
    SourceTerm,
    get_rng,
    isotropic_directions,
    particle_arrays,
    validate_nonnegative,
    validate_positive,
)
from latent_dirac.state.particle_state import ParticleState


class BetaPlusPositronSource(SourceTerm):
    half_life_s: float
    beta_plus_branching_ratio: float
    initial_activity_bq: float
    endpoint_energy_MeV: float
    source_radius_m: float
    macro_particles: int

    @field_validator("half_life_s", "endpoint_energy_MeV", "macro_particles")
    @classmethod
    def _positive(cls, value, info):
        return validate_positive(info.field_name, value)

    @field_validator("initial_activity_bq", "source_radius_m")
    @classmethod
    def _nonnegative(cls, value, info):
        return validate_nonnegative(info.field_name, value)

    @field_validator("beta_plus_branching_ratio")
    @classmethod
    def _branching_ratio(cls, value):
        if not 0.0 <= value <= 1.0:
            raise ValueError("beta_plus_branching_ratio must be in [0, 1]")
        return value

    def sample(self, rng: np.random.Generator | None = None) -> ParticleState:
        rng = get_rng(rng)
        count = int(self.macro_particles)
        total_yield_per_second = self.initial_activity_bq * self.beta_plus_branching_ratio

        energy_mev = self.endpoint_energy_MeV * rng.beta(3.0, 3.0, size=count)
        momentum = kinetic_energy_to_momentum_magnitude(mev_to_joule(energy_mev), positron.mass_kg)
        directions = isotropic_directions(rng, count)

        radii = self.source_radius_m * rng.random(count) ** (1.0 / 3.0)
        position_m = radii[:, np.newaxis] * isotropic_directions(rng, count)

        return ParticleState(
            species=positron,
            position_m=position_m,
            momentum_kg_m_s=momentum[:, np.newaxis] * directions,
            time_s=np.zeros(count),
            weight=np.full(count, total_yield_per_second / count),
            metadata={
                "source": "BetaPlusPositronSource",
                "model_type": "simplified",
                "physics_note": "Simplified source using an approximate beta energy distribution.",
                "assumptions": {
                    "yield_window": "one second at initial activity",
                    "energy_distribution": "Beta(3, 3) scaled to endpoint energy",
                    "emission": "isotropic directions",
                    "position_distribution": "uniform sphere",
                },
            },
            **particle_arrays(count),
        )
