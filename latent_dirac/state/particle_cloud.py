"""Universal intermediate state for macro-particle simulations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from latent_dirac.core.species import ParticleSpecies
from latent_dirac.core.units import (
    gamma_from_momentum,
    kinetic_energy_from_momentum,
    velocity_from_momentum,
)


class ParticleCloud(BaseModel):
    """Macro-particle cloud stored in SI units.

    The `weight` array maps each macro-particle to its represented physical
    particle count. `alive` is the accepted/not-lost state used by beamline
    elements and diagnostics.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    species: ParticleSpecies
    position_m: np.ndarray
    momentum_kg_m_s: np.ndarray
    time_s: np.ndarray
    weight: np.ndarray
    alive: np.ndarray
    particle_id: np.ndarray
    parent_id: np.ndarray
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("position_m", "momentum_kg_m_s", mode="before")
    @classmethod
    def _as_float_matrix(cls, value):
        return np.asarray(value, dtype=float)

    @field_validator("time_s", "weight", mode="before")
    @classmethod
    def _as_float_vector(cls, value):
        return np.asarray(value, dtype=float)

    @field_validator("alive", mode="before")
    @classmethod
    def _as_bool_vector(cls, value):
        return np.asarray(value, dtype=bool)

    @field_validator("particle_id", "parent_id", mode="before")
    @classmethod
    def _as_int_vector(cls, value):
        return np.asarray(value, dtype=int)

    @model_validator(mode="after")
    def _validate_shapes(self):
        if self.position_m.ndim != 2 or self.position_m.shape[1] != 3:
            raise ValueError("position_m must have shape (N, 3)")
        if self.momentum_kg_m_s.shape != self.position_m.shape:
            raise ValueError("momentum_kg_m_s must have shape (N, 3)")
        n_particles = self.position_m.shape[0]
        for field_name in ("time_s", "weight", "alive", "particle_id", "parent_id"):
            value = getattr(self, field_name)
            if value.shape != (n_particles,):
                raise ValueError(f"{field_name} must have shape (N,)")
        if np.any(self.weight < 0.0):
            raise ValueError("weight must be non-negative")
        return self

    def weighted_count(self) -> float:
        return float(np.sum(self.weight[self.alive]))

    def gamma(self) -> np.ndarray:
        return gamma_from_momentum(self.momentum_kg_m_s, self.species.mass_kg)

    def velocity(self) -> np.ndarray:
        return velocity_from_momentum(self.momentum_kg_m_s, self.species.mass_kg)

    def kinetic_energy_joule(self) -> np.ndarray:
        return kinetic_energy_from_momentum(self.momentum_kg_m_s, self.species.mass_kg)

    def mean_kinetic_energy_joule(self) -> float:
        live_weights = self.weight[self.alive]
        if live_weights.size == 0 or np.sum(live_weights) == 0.0:
            return 0.0
        live_energy = self.kinetic_energy_joule()[self.alive]
        return float(np.average(live_energy, weights=live_weights))

    def copy(self) -> "ParticleCloud":
        return ParticleCloud(
            species=self.species,
            position_m=self.position_m.copy(),
            momentum_kg_m_s=self.momentum_kg_m_s.copy(),
            time_s=self.time_s.copy(),
            weight=self.weight.copy(),
            alive=self.alive.copy(),
            particle_id=self.particle_id.copy(),
            parent_id=self.parent_id.copy(),
            metadata=deepcopy(self.metadata),
        )

    def apply_alive_mask(self, mask) -> "ParticleCloud":
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != self.alive.shape:
            raise ValueError("alive mask must have shape (N,)")
        self.alive = self.alive & mask_array
        return self
