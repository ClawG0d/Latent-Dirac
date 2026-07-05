"""Ideal Penning trap field: quadrupole electrostatic well plus axial B.

Fidelity tier: parameterized. The potential
``V = (V0 / (2 d^2)) (z^2 - (x^2 + y^2)/2)`` is the ideal quadrupole
(exactly Laplacian); real electrode stacks, space charge, and field
imperfections are out of scope — this is the trap-optics analog of the
hard-edge magnets. The field is global (no hard edge).
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, field_validator

from latent_dirac.core.species import ParticleSpecies
from latent_dirac.fields.base import Field


class PenningTrapField(BaseModel, Field):
    """Uniform axial B plus the ideal quadrupole electrostatic well.

    ``E = (V0 / d^2) * (x/2, y/2, -(z - center_z_m))``; positive
    ``q * v0_volt`` confines axially. Analytic eigenfrequencies are
    exposed by :meth:`eigenfrequencies`.
    """

    v0_volt: float
    d_m: float
    b_tesla: float
    center_z_m: float = 0.0

    @field_validator("d_m")
    @classmethod
    def _positive_d(cls, value):
        if value <= 0.0:
            raise ValueError("d_m must be positive")
        return value

    def E(self, x, t) -> np.ndarray:
        x_array = np.asarray(x, dtype=float)
        scale = self.v0_volt / self.d_m**2
        single = x_array.ndim == 1
        points = np.atleast_2d(x_array)
        field = np.column_stack(
            [
                0.5 * scale * points[:, 0],
                0.5 * scale * points[:, 1],
                -scale * (points[:, 2] - self.center_z_m),
            ]
        )
        return field[0] if single else field

    def B(self, x, t) -> np.ndarray:
        x_array = np.asarray(x, dtype=float)
        if x_array.ndim == 1:
            return np.array([0.0, 0.0, self.b_tesla])
        field = np.zeros_like(x_array)
        field[:, 2] = self.b_tesla
        return field

    def eigenfrequencies(self, species: ParticleSpecies) -> tuple[float, float, float]:
        """Analytic (omega_plus, omega_minus, omega_z) magnitudes in rad/s.

        Non-relativistic single-particle values, returned as positive
        magnitudes; the rotation sense is set by sign(q * b_tesla). Raises
        if the configuration does not confine the species.
        """

        omega_z_squared = species.charge_c * self.v0_volt / (species.mass_kg * self.d_m**2)
        if omega_z_squared <= 0.0:
            raise ValueError(
                "axially deconfining configuration: q * v0_volt must be positive for this species"
            )
        omega_c = abs(species.charge_c * self.b_tesla) / species.mass_kg
        discriminant = omega_c**2 - 2.0 * omega_z_squared
        if discriminant <= 0.0:
            raise ValueError("unstable Penning trap configuration: omega_c^2 must exceed 2 omega_z^2")
        root = math.sqrt(discriminant)
        omega_plus = 0.5 * (omega_c + root)
        # Vieta form avoids the catastrophic cancellation in (omega_c - root)
        omega_minus = 0.5 * omega_z_squared / omega_plus
        return omega_plus, omega_minus, math.sqrt(omega_z_squared)
