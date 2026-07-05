"""Idealized hard-edge magnetic dipole field."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from latent_dirac.fields.base import Field, broadcast_vector


class DipoleField(BaseModel, Field):
    """Idealized hard-edge magnetic dipole.

    The field is the configured uniform vector inside the longitudinal
    extent ``abs(z - center_z_m) <= 0.5 * length_m`` and zero outside. This
    is a parameterized beam optics model, not a magnet engineering model:
    fringe fields and pole geometry are deliberately out of scope.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    B_vector_t: np.ndarray
    length_m: float
    center_z_m: float = 0.0

    @field_validator("B_vector_t", mode="before")
    @classmethod
    def _as_vector(cls, value):
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,):
            raise ValueError("B_vector_t must have shape (3,)")
        return vector

    @field_validator("length_m")
    @classmethod
    def _positive_length(cls, value):
        if value <= 0.0:
            raise ValueError("length_m must be positive")
        return value

    def E(self, x, t) -> np.ndarray:
        x_array = np.asarray(x, dtype=float)
        if x_array.ndim == 1:
            return np.zeros(3)
        return np.zeros_like(x_array)

    def B(self, x, t) -> np.ndarray:
        x_array = np.asarray(x, dtype=float)
        if x_array.ndim == 1:
            if self._inside_z(float(x_array[2])):
                return self.B_vector_t.copy()
            return np.zeros(3)

        inside = np.abs(x_array[:, 2] - self.center_z_m) <= 0.5 * self.length_m
        field = np.zeros_like(x_array)
        field[inside] = broadcast_vector(self.B_vector_t, x_array[inside])
        return field

    def _inside_z(self, z_m: float) -> bool:
        return abs(z_m - self.center_z_m) <= 0.5 * self.length_m
