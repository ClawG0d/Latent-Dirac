"""Mean-field space charge: the uniform-sphere model (parameterized tier).

The alive cloud is approximated by one uniform-density sphere fitted per
transport step (charge-weighted centroid, R = sqrt(5/3) * r_rms). The
interior field is linear, the exterior field is Coulombic, and the
self-magnetic field is neglected — valid for non-relativistic clouds
(beta << 1, the trap regime). No self-consistent iteration: rung 2 of
the declared collective-effects ladder. See the 2026-07-05 mean-field
space charge spec for the validity envelope.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField

from latent_dirac.core.constants import epsilon_0
from latent_dirac.fields.base import Field
from latent_dirac.state.particle_state import ParticleState

VALIDITY_NOTE = (
    "parameterized uniform-sphere mean field: electrostatic only "
    "(beta << 1), single-sphere fit, no self-consistency"
)


class UniformSphereSelfField(BaseModel, Field):
    """Field of a uniform-density sphere: linear inside, Coulomb outside."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    center_m: tuple[float, float, float]
    radius_m: float = PydanticField(gt=0)
    total_charge_c: float

    def E(self, x, t) -> np.ndarray:
        positions = np.asarray(x, dtype=float)
        single = positions.ndim == 1
        points = np.atleast_2d(positions)
        offset = points - np.asarray(self.center_m)
        r = np.linalg.norm(offset, axis=1)
        k = self.total_charge_c / (4.0 * np.pi * epsilon_0)
        # E = k * d / R^3 inside, k * d / r^3 outside; flooring the
        # exterior denominator at R is exact (that branch is only selected
        # for r > R) and keeps the discarded branch free of divide-by-zero
        scale = np.where(
            r <= self.radius_m,
            k / self.radius_m**3,
            k / np.maximum(r, self.radius_m) ** 3,
        )
        field = offset * scale[:, np.newaxis]
        return field[0] if single else field

    def B(self, x, t) -> np.ndarray:
        positions = np.asarray(x, dtype=float)
        if positions.ndim == 1:
            return np.zeros(3)
        return np.zeros_like(positions)


def fit_uniform_sphere(state: ParticleState) -> UniformSphereSelfField | None:
    """Fit the alive cloud; None when there is nothing to fit.

    Dead particles neither source nor shape the field. Returns None for
    fewer than two alive particles or a degenerate (zero-radius) cloud
    rather than a singular field.
    """
    alive = state.alive
    if int(alive.sum()) < 2:
        return None
    weights = state.weight[alive]
    positions = state.position_m[alive]
    center = np.average(positions, axis=0, weights=weights)
    r_rms_sq = float(np.average(((positions - center) ** 2).sum(axis=1), weights=weights))
    if r_rms_sq <= 0.0:
        return None
    radius = float(np.sqrt(5.0 * r_rms_sq / 3.0))
    total_charge = float(weights.sum() * state.species.charge_c)
    return UniformSphereSelfField(
        center_m=tuple(float(component) for component in center),
        radius_m=radius,
        total_charge_c=total_charge,
    )
