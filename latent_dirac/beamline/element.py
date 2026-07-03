"""Beamline element interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from latent_dirac.state.particle_cloud import ParticleCloud


class BeamlineElement(ABC):
    """Applies an acceptance or transport operation to a particle cloud."""

    @abstractmethod
    def apply(self, cloud: ParticleCloud) -> ParticleCloud:
        raise NotImplementedError
