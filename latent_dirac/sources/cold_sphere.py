"""Idealized prepared cold cloud: a uniform sphere at rest.

Fidelity tier: placeholder — this is a prepared initial state (the
standard starting point for trap and space-charge studies), not a
formation model: no temperature, no loading dynamics, no correlations.
"""

from __future__ import annotations

import numpy as np
from pydantic import field_validator

from latent_dirac.core.species import BUILTIN_SPECIES
from latent_dirac.sources.base import (
    SourceTerm,
    get_rng,
    particle_arrays,
    validate_positive,
)
from latent_dirac.state.particle_state import ParticleState


class ColdUniformSphereSource(SourceTerm):
    species_name: str
    macro_particles: int
    radius_m: float
    weight: float = 1.0
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @field_validator("macro_particles", "radius_m", "weight")
    @classmethod
    def _positive(cls, value, info):
        return validate_positive(info.field_name, value)

    @field_validator("species_name")
    @classmethod
    def _known_species(cls, value):
        if value not in BUILTIN_SPECIES:
            raise ValueError(
                f"unknown species {value!r}; choose from {sorted(BUILTIN_SPECIES)}"
            )
        return value

    def sample(self, rng=None) -> ParticleState:
        rng = get_rng(rng)
        count = int(self.macro_particles)
        directions = rng.normal(size=(count, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
        radii = self.radius_m * rng.uniform(0.0, 1.0, count) ** (1.0 / 3.0)
        species = BUILTIN_SPECIES[self.species_name]
        return ParticleState(
            species=species,
            position_m=directions * radii[:, np.newaxis] + np.asarray(self.center_m),
            momentum_kg_m_s=np.zeros((count, 3)),
            time_s=np.zeros(count),
            weight=np.full(count, float(self.weight)),
            metadata={
                "source": "ColdUniformSphereSource",
                "model_type": "placeholder",
                "physics_note": (
                    "idealized prepared cold cloud (uniform sphere at rest); "
                    "no formation, temperature, or correlation physics"
                ),
            },
            **particle_arrays(count),
        )
