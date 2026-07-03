"""Radial aperture acceptance."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, field_validator

from latent_dirac.beamline.element import BeamlineElement
from latent_dirac.state.particle_cloud import ParticleCloud


class Aperture(BaseModel, BeamlineElement):
    """Radial aperture evaluated at the cloud's current transverse position."""

    radius_m: float
    z_m: float

    @field_validator("radius_m")
    @classmethod
    def _positive_radius(cls, value):
        if value <= 0.0:
            raise ValueError("radius_m must be positive")
        return value

    def apply(self, cloud: ParticleCloud) -> ParticleCloud:
        result = cloud.copy()
        radial = np.linalg.norm(result.position_m[:, :2], axis=1)
        result.apply_alive_mask(radial <= self.radius_m)
        result.metadata.setdefault("apertures", []).append(
            {"radius_m": self.radius_m, "z_m": self.z_m}
        )
        return result
