"""Idealized solenoid field."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, field_validator

from latent_dirac.fields.base import Field


class SolenoidField(BaseModel, Field):
    """Idealized hard-edge solenoid.

    The first implementation approximates the inside of the solenoid as a
    uniform Bz field within the specified length and radius. The outside field
    is zero. Fringe fields, coils, materials, and detailed magnet engineering
    are deliberately out of scope.
    """

    b_tesla: float
    radius_m: float
    length_m: float
    center_z_m: float = 0.0

    @field_validator("radius_m", "length_m")
    @classmethod
    def _positive(cls, value, info):
        if value <= 0.0:
            raise ValueError(f"{info.field_name} must be positive")
        return value

    def E(self, x, t) -> np.ndarray:
        x_array = np.asarray(x, dtype=float)
        if x_array.ndim == 1:
            return np.zeros(3)
        return np.zeros_like(x_array)

    def B(self, x, t) -> np.ndarray:
        x_array = np.asarray(x, dtype=float)
        if x_array.ndim == 1:
            return np.array([0.0, 0.0, self.b_tesla]) if self._inside(x_array) else np.zeros(3)

        radial = np.linalg.norm(x_array[:, :2], axis=1)
        z_inside = np.abs(x_array[:, 2] - self.center_z_m) <= 0.5 * self.length_m
        inside = (radial <= self.radius_m) & z_inside
        field = np.zeros_like(x_array)
        field[inside, 2] = self.b_tesla
        return field

    def _inside(self, position: np.ndarray) -> bool:
        radial = float(np.linalg.norm(position[:2]))
        z_inside = abs(float(position[2]) - self.center_z_m) <= 0.5 * self.length_m
        return radial <= self.radius_m and z_inside


class ThinSheetSolenoidField(BaseModel, Field):
    """Finite thin-current-sheet solenoid with a smooth first-order fringe.

    On-axis profile of an ideal thin cylindrical current sheet, with the
    first radial order of the axisymmetric expansion:

        b(z) = (B0/2) * (f(z+) - f(z-)),  f(u) = u / sqrt(u^2 + R^2)
        B_z  = b(z)
        B_r  = -(r/2) * b'(z)

    The pair is exactly divergence-free everywhere (it is the curl of
    A_phi = r b(z) / 2) but curl-free only to first order in r, so
    accuracy degrades off-axis — intended for r <~ radius_m. `b_tesla`
    is the sheet strength B0 = mu0 n I, which keeps the integrated
    on-axis strength exactly B0 * length_m (the hard-edge equivalent);
    the field at the center of a short solenoid is
    B0 * L / sqrt(L^2 + 4 R^2) < B0. Fidelity tier: parameterized.
    Design record: the 2026-07-06 thin-sheet solenoid profile spec.
    """

    b_tesla: float
    radius_m: float
    length_m: float
    center_z_m: float = 0.0

    @field_validator("radius_m", "length_m")
    @classmethod
    def _positive(cls, value, info):
        if value <= 0.0:
            raise ValueError(f"{info.field_name} must be positive")
        return value

    def E(self, x, t) -> np.ndarray:
        x_array = np.asarray(x, dtype=float)
        if x_array.ndim == 1:
            return np.zeros(3)
        return np.zeros_like(x_array)

    def B(self, x, t) -> np.ndarray:
        x_array = np.asarray(x, dtype=float)
        single = x_array.ndim == 1
        positions = x_array[np.newaxis, :] if single else x_array

        b, db_dz = self._on_axis_profile(positions[:, 2])
        field = np.empty_like(positions)
        field[:, 0] = -0.5 * positions[:, 0] * db_dz
        field[:, 1] = -0.5 * positions[:, 1] * db_dz
        field[:, 2] = b
        return field[0] if single else field

    def _on_axis_profile(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """On-axis b(z) and b'(z) of the thin sheet (elementary functions)."""

        r_sq = self.radius_m * self.radius_m
        zeta_plus = (z - self.center_z_m) + 0.5 * self.length_m
        zeta_minus = (z - self.center_z_m) - 0.5 * self.length_m
        root_plus = np.sqrt(zeta_plus * zeta_plus + r_sq)
        root_minus = np.sqrt(zeta_minus * zeta_minus + r_sq)
        b = 0.5 * self.b_tesla * (zeta_plus / root_plus - zeta_minus / root_minus)
        db_dz = 0.5 * self.b_tesla * r_sq * (root_plus**-3 - root_minus**-3)
        return b, db_dz
