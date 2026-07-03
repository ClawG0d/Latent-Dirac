"""Trajectory storage for sampled particle histories."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Trajectory(BaseModel):
    """Optional trajectory record with positions and momenta over time."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    time_s: np.ndarray
    position_m: np.ndarray
    momentum_kg_m_s: np.ndarray

    @field_validator("time_s", "position_m", "momentum_kg_m_s", mode="before")
    @classmethod
    def _as_array(cls, value):
        return np.asarray(value, dtype=float)

    @model_validator(mode="after")
    def _validate_shapes(self):
        if self.position_m.shape != self.momentum_kg_m_s.shape:
            raise ValueError("position_m and momentum_kg_m_s must have matching shapes")
        if self.position_m.ndim != 3 or self.position_m.shape[-1] != 3:
            raise ValueError("position_m must have shape (T, N, 3)")
        if self.time_s.shape != (self.position_m.shape[0],):
            raise ValueError("time_s must have shape (T,)")
        return self
