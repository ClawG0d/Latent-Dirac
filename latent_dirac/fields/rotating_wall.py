"""Rotating-wall drive: a rotating multipole transverse E field.

Fidelity tier: parameterized. This is the rotating single-particle E field
only — the trap-optics analogue of the ideal Penning well. In a real trap a
rotating wall compresses a plasma radially, but that is a collective,
collisional effect (rotating field + dissipation) requiring a PIC/collective
solver the NumPy + mean-field core does not have; the single-particle field
here drives the correct rotating force and E×B drift but does NOT
self-consistently compress a plasma. Plasma compression is not a figure of
merit and no energetics are claimed. See the 2026-07-07 rotating-wall spec.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from latent_dirac.fields.base import Field


class RotatingWallField(BaseModel, Field):
    """Transverse (x, y) rotating multipole E field, z-independent, B = 0.

    With ω = 2π·frequency_hz and θ = ω·t + phase_rad:

    - multipole 1 (dipole): uniform field rotating at ω,
      ``E = amplitude_v_m · [cos θ, sin θ, 0]``;
    - multipole 2 (quadrupole): a quadrupole pattern whose axes rotate,
      linear in transverse position, scaled so ``|E| = amplitude_v_m`` at
      ``radius_m``: with ``g = amplitude_v_m / radius_m``,
      ``E = g · [−(x cos θ + y sin θ), y cos θ − x sin θ, 0]`` — the gradient
      of the rotating quadrupole potential ``Φ = (g/2)[(x²−y²)cos θ +
      2xy sin θ]``, so it is curl- and divergence-free (a true quadrupole,
      not a radial field) with ``|E| = g·r``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    multipole: int
    amplitude_v_m: float
    radius_m: float = 0.02
    frequency_hz: float
    phase_rad: float = 0.0

    @field_validator("multipole")
    @classmethod
    def _valid_multipole(cls, value):
        if value not in (1, 2):
            raise ValueError("multipole must be 1 (dipole) or 2 (quadrupole)")
        return value

    @field_validator("amplitude_v_m", "frequency_hz", "radius_m")
    @classmethod
    def _positive(cls, value, info):
        if value <= 0.0:
            raise ValueError(f"{info.field_name} must be positive")
        return value

    def E(self, x, t) -> np.ndarray:
        x_array = np.asarray(x, dtype=float)
        single = x_array.ndim == 1
        points = np.atleast_2d(x_array)
        theta = 2.0 * np.pi * self.frequency_hz * np.atleast_1d(np.asarray(t, dtype=float)) + self.phase_rad
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        if self.multipole == 1:
            comp = self.amplitude_v_m * np.stack([cos_t, sin_t, np.zeros_like(cos_t)], axis=-1)
            # uniform in space; broadcast the per-time field to the N positions
            # (raises on a genuine (N != len(t)) mismatch rather than silently
            # returning the wrong shape)
            field = np.broadcast_to(comp, points.shape)
        else:  # multipole == 2
            g = self.amplitude_v_m / self.radius_m
            xx, yy = points[:, 0], points[:, 1]
            field = np.column_stack(
                [-g * (xx * cos_t + yy * sin_t), g * (yy * cos_t - xx * sin_t), np.zeros_like(xx)]
            )
        return field[0] if single else field

    def B(self, x, t) -> np.ndarray:
        x_array = np.asarray(x, dtype=float)
        if x_array.ndim == 1:
            return np.zeros(3)
        return np.zeros_like(x_array)
