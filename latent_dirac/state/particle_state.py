"""Universal simulation state for macro-particle transport.

`ParticleState` is a pytree-compatible dataclass, not a pydantic model:
pydantic owns the Model/scene layer (static configuration, fail-fast
schemas), while simulation state stays a plain container of array leaves.
`tree_flatten`/`tree_unflatten` follow the JAX conventions, but Phase 3
registration will still need a thin wrapper: unflatten must bypass
`__post_init__` validation (JAX may unflatten with tracer or placeholder
leaves), and the mutable `metadata` dict in the static aux data is not
hashable for jit cache keys. SI units live at this boundary; kernels work
in dimensionless variables internally.

Particles are never deleted. Losses are recorded by the `alive` mask and
the per-particle ledger channel `lost_at_element` (int32, -1 = alive),
which keeps array shapes static and makes every antiparticle's fate
addressable by pipeline stage.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from latent_dirac.core.species import ParticleSpecies
from latent_dirac.core.units import (
    gamma_from_momentum,
    kinetic_energy_from_momentum,
    velocity_from_momentum,
)

_ARRAY_FIELDS = (
    "position_m",
    "momentum_kg_m_s",
    "time_s",
    "weight",
    "alive",
    "particle_id",
    "parent_id",
    "lost_at_element",
)


@dataclass
class ParticleState:
    """Macro-particle state stored in SI units (SoA layout).

    `weight` maps each macro-particle to its represented physical particle
    count. `alive` is the accepted/not-lost mask used by beamline elements
    and diagnostics. `lost_at_element` records the pipeline stage index
    that killed each particle (-1 while alive).
    """

    species: ParticleSpecies
    position_m: np.ndarray
    momentum_kg_m_s: np.ndarray
    time_s: np.ndarray
    weight: np.ndarray
    alive: np.ndarray
    particle_id: np.ndarray
    parent_id: np.ndarray
    lost_at_element: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.position_m = np.asarray(self.position_m, dtype=float)
        self.momentum_kg_m_s = np.asarray(self.momentum_kg_m_s, dtype=float)
        self.time_s = np.asarray(self.time_s, dtype=float)
        self.weight = np.asarray(self.weight, dtype=float)
        self.alive = np.asarray(self.alive, dtype=bool)
        self.particle_id = np.asarray(self.particle_id, dtype=int)
        self.parent_id = np.asarray(self.parent_id, dtype=int)

        if self.position_m.ndim != 2 or self.position_m.shape[1] != 3:
            raise ValueError("position_m must have shape (N, 3)")
        if self.momentum_kg_m_s.shape != self.position_m.shape:
            raise ValueError("momentum_kg_m_s must have shape (N, 3)")
        n_particles = self.position_m.shape[0]

        if self.lost_at_element is None:
            self.lost_at_element = np.full(n_particles, -1, dtype=np.int32)
        else:
            self.lost_at_element = np.asarray(self.lost_at_element, dtype=np.int32)

        for field_name in ("time_s", "weight", "alive", "particle_id", "parent_id", "lost_at_element"):
            value = getattr(self, field_name)
            if value.shape != (n_particles,):
                raise ValueError(f"{field_name} must have shape (N,)")
        if np.any(self.weight < 0.0):
            raise ValueError("weight must be non-negative")

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

    def copy(self) -> ParticleState:
        return ParticleState(
            species=self.species,
            position_m=self.position_m.copy(),
            momentum_kg_m_s=self.momentum_kg_m_s.copy(),
            time_s=self.time_s.copy(),
            weight=self.weight.copy(),
            alive=self.alive.copy(),
            particle_id=self.particle_id.copy(),
            parent_id=self.parent_id.copy(),
            lost_at_element=self.lost_at_element.copy(),
            metadata=deepcopy(self.metadata),
        )

    def apply_alive_mask(self, mask) -> ParticleState:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != self.alive.shape:
            raise ValueError("alive mask must have shape (N,)")
        self.alive = self.alive & mask_array
        return self

    def tree_flatten(self) -> tuple[tuple[np.ndarray, ...], tuple]:
        """Flatten in the JAX pytree convention: array leaves, static aux."""

        leaves = tuple(getattr(self, name) for name in _ARRAY_FIELDS)
        aux = (self.species, self.metadata)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux: tuple, leaves: tuple) -> ParticleState:
        # metadata is shared by reference (static aux convention), unlike
        # copy(), which deep-copies it
        species, metadata = aux
        kwargs = dict(zip(_ARRAY_FIELDS, leaves, strict=True))
        return cls(species=species, metadata=metadata, **kwargs)
