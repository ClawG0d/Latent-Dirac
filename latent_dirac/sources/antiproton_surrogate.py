"""Surrogate antiproton source term.

This is not a detailed accelerator target model. It samples accepted-source
macro-particles from simple yield, momentum, angular, and source-size inputs.
"""

from __future__ import annotations

import numpy as np
from pydantic import field_validator

from latent_dirac.core.species import antiproton
from latent_dirac.core.units import momentum_gev_c_to_si
from latent_dirac.sources.base import (
    SourceTerm,
    forward_directions,
    get_rng,
    particle_arrays,
    validate_nonnegative,
    validate_positive,
)
from latent_dirac.state.particle_cloud import ParticleCloud


class AntiprotonSurrogateSource(SourceTerm):
    primary_proton_count: float
    yield_pbar_per_primary_in_acceptance: float
    central_momentum_GeV_c: float
    momentum_spread_fraction: float
    angular_rms_rad: float
    source_sigma_m: float
    bunch_length_s: float
    macro_particles: int

    @field_validator("primary_proton_count", "central_momentum_GeV_c", "source_sigma_m", "macro_particles")
    @classmethod
    def _positive(cls, value, info):
        return validate_positive(info.field_name, value)

    @field_validator(
        "yield_pbar_per_primary_in_acceptance",
        "momentum_spread_fraction",
        "angular_rms_rad",
        "bunch_length_s",
    )
    @classmethod
    def _nonnegative(cls, value, info):
        return validate_nonnegative(info.field_name, value)

    def sample(self, rng: np.random.Generator | None = None) -> ParticleCloud:
        rng = get_rng(rng)
        count = int(self.macro_particles)
        total_yield = self.primary_proton_count * self.yield_pbar_per_primary_in_acceptance
        momentum_gev_c = rng.normal(
            self.central_momentum_GeV_c,
            self.central_momentum_GeV_c * self.momentum_spread_fraction,
            size=count,
        )
        momentum_gev_c = np.clip(momentum_gev_c, 1.0e-12, None)
        momentum = momentum_gev_c_to_si(momentum_gev_c)
        directions = forward_directions(rng, count, self.angular_rms_rad)

        return ParticleCloud(
            species=antiproton,
            position_m=rng.normal(0.0, self.source_sigma_m, size=(count, 3)),
            momentum_kg_m_s=momentum[:, np.newaxis] * directions,
            time_s=rng.normal(0.0, self.bunch_length_s, size=count),
            weight=np.full(count, total_yield / count),
            metadata={
                "source": "AntiprotonSurrogateSource",
                "model_type": "surrogate",
                "physics_note": "Surrogate source, not a detailed target model.",
                "assumptions": {
                    "momentum_distribution": "normal GeV/c around central momentum",
                    "angular_distribution": "small-angle Gaussian around +z",
                    "position_distribution": "3D Gaussian source size",
                },
            },
            **particle_arrays(count),
        )
