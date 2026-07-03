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
