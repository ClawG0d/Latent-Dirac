"""Base interfaces and sampling helpers for source terms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from latent_dirac.state.particle_cloud import ParticleCloud


class SourceTerm(BaseModel, ABC):
    """A source term that samples a macro-particle cloud.

    Unknown constructor parameters are rejected so that typos in
    hand-written scene `params` fail fast instead of silently producing a
    default-behaving source.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @abstractmethod
    def sample(self, rng: np.random.Generator | None = None) -> ParticleCloud:
        raise NotImplementedError


def get_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def validate_nonnegative(name: str, value: float) -> float:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def validate_positive(name: str, value: float) -> float:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def forward_directions(
    rng: np.random.Generator,
    count: int,
    angular_rms_rad: float,
) -> np.ndarray:
    slopes = rng.normal(0.0, angular_rms_rad, size=(count, 2))
    directions = np.column_stack([slopes[:, 0], slopes[:, 1], np.ones(count)])
    return directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]


def isotropic_directions(rng: np.random.Generator, count: int) -> np.ndarray:
    directions = rng.normal(size=(count, 3))
    norms = np.linalg.norm(directions, axis=1)
    while np.any(norms == 0.0):
        directions[norms == 0.0] = rng.normal(size=(np.sum(norms == 0.0), 3))
        norms = np.linalg.norm(directions, axis=1)
    return directions / norms[:, np.newaxis]


def particle_arrays(count: int) -> dict[str, Any]:
    return {
        "alive": np.ones(count, dtype=bool),
        "particle_id": np.arange(count, dtype=int),
        "parent_id": np.full(count, -1, dtype=int),
    }
