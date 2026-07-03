"""Solver interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from latent_dirac.fields.base import Field
from latent_dirac.state.particle_cloud import ParticleCloud


class Solver(BaseModel, ABC):
    """A transport solver that propagates a particle cloud through a field."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def propagate(self, cloud: ParticleCloud, field: Field) -> ParticleCloud:
        raise NotImplementedError
