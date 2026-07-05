"""Idealized hard-edge magnetic quadrupole field."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from latent_dirac.fields.base import Field


class QuadrupoleField(BaseModel, Field):
    """Idealized hard-edge magnetic quadrupole.

    Inside the longitudinal extent ``abs(z - center_z_m) <= 0.5 * length_m``
    the field is the linear transverse profile ``B_x = gradient_t_m * y``,
    ``B_y = gradient_t_m * x``, ``B_z = 0``; outside it is zero. Changing
    the sign of ``gradient_t_m`` swaps the focusing and defocusing planes.
    This is a parameterized beam optics model, not a magnet engineering
    model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    gradient_t_m: float
    length_m: float
    center_z_m: float = 0.0

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
            if abs(float(x_array[2]) - self.center_z_m) <= 0.5 * self.length_m:
                return np.array(
                    [
                        self.gradient_t_m * x_array[1],
                        self.gradient_t_m * x_array[0],
                        0.0,
                    ]
                )
            return np.zeros(3)

        inside = np.abs(x_array[:, 2] - self.center_z_m) <= 0.5 * self.length_m
        field = np.zeros_like(x_array)
        field[inside, 0] = self.gradient_t_m * x_array[inside, 1]
        field[inside, 1] = self.gradient_t_m * x_array[inside, 0]
        return field
