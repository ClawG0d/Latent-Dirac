"""Uniform electromagnetic field."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator
from pydantic import Field as PydanticField

from latent_dirac.fields.base import Field, broadcast_vector


class UniformField(BaseModel, Field):
    """A spatially and temporally uniform electromagnetic field."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    E_vector_v_m: np.ndarray = PydanticField(default_factory=lambda: np.zeros(3))
    B_vector_t: np.ndarray = PydanticField(default_factory=lambda: np.zeros(3))

    @field_validator("E_vector_v_m", "B_vector_t", mode="before")
    @classmethod
    def _as_vector(cls, value):
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,):
            raise ValueError("field vector must have shape (3,)")
        return vector

    def E(self, x, t) -> np.ndarray:
        return broadcast_vector(self.E_vector_v_m, x)

    def B(self, x, t) -> np.ndarray:
        return broadcast_vector(self.B_vector_t, x)
