"""Beamline element interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from latent_dirac.state.particle_state import ParticleState


class BeamlineElement(ABC):
    """Applies an acceptance or transport operation to a particle cloud."""

    @abstractmethod
    def apply(self, cloud: ParticleState) -> ParticleState:
        raise NotImplementedError
