"""Parameterized positron source inspired by pair-production source terms.

This is not full electromagnetic shower physics. It samples macro-particles
from user-provided yield, energy, angular, and source-size parameters.
"""

from __future__ import annotations

import numpy as np
from pydantic import field_validator

from latent_dirac.core.species import positron
from latent_dirac.core.units import kinetic_energy_to_momentum_magnitude, mev_to_joule
from latent_dirac.sources.base import (
    SourceTerm,
    forward_directions,
    get_rng,
    particle_arrays,
    validate_nonnegative,
    validate_positive,
)
from latent_dirac.state.particle_state import ParticleState


class PositronPairSource(SourceTerm):
    primary_count: float
    yield_eplus_per_primary: float
    mean_energy_MeV: float
    energy_spread_MeV: float
    angular_rms_rad: float
    source_sigma_m: float
    bunch_length_s: float
    macro_particles: int

    @field_validator("primary_count", "mean_energy_MeV", "source_sigma_m", "macro_particles")
    @classmethod
    def _positive(cls, value, info):
        return validate_positive(info.field_name, value)

    @field_validator("yield_eplus_per_primary", "energy_spread_MeV", "angular_rms_rad", "bunch_length_s")
    @classmethod
    def _nonnegative(cls, value, info):
        return validate_nonnegative(info.field_name, value)

    def sample(self, rng: np.random.Generator | None = None) -> ParticleState:
        rng = get_rng(rng)
        count = int(self.macro_particles)
        total_yield = self.primary_count * self.yield_eplus_per_primary
        energy_mev = rng.normal(self.mean_energy_MeV, self.energy_spread_MeV, size=count)
        energy_mev = np.clip(energy_mev, 1.0e-9, None)
        momentum = kinetic_energy_to_momentum_magnitude(mev_to_joule(energy_mev), positron.mass_kg)
        directions = forward_directions(rng, count, self.angular_rms_rad)

        return ParticleState(
            species=positron,
            position_m=rng.normal(0.0, self.source_sigma_m, size=(count, 3)),
            momentum_kg_m_s=momentum[:, np.newaxis] * directions,
            time_s=rng.normal(0.0, self.bunch_length_s, size=count),
            weight=np.full(count, total_yield / count),
            metadata={
                "source": "PositronPairSource",
                "model_type": "parameterized",
                "physics_note": "Parameterized source term, not full shower physics.",
                "assumptions": {
                    "energy_distribution": "normal kinetic energy clipped at positive values",
                    "angular_distribution": "small-angle Gaussian around +z",
                    "position_distribution": "3D Gaussian source size",
                },
            },
            **particle_arrays(count),
        )
