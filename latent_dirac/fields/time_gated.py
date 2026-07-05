"""Time-gated field wrapper: the first field model that uses `t`.

Multiplies an inner field by a per-particle time gate
(`t_on_s <= t < t_off_s`). This is the mechanism behind dynamic trap
capture: a potential raised only after the bunch has entered. The gate is
ideal (instantaneous switching); ramp shapes and RF structures are later
field-library extensions.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from latent_dirac.fields.base import Field


class TimeGatedField(BaseModel, Field):
    """Inner field active only inside the half-open window [t_on_s, t_off_s)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inner: Field
    t_on_s: float
    t_off_s: float

    @model_validator(mode="after")
    def _validate_window(self):
        if self.t_off_s <= self.t_on_s:
            raise ValueError("t_off_s must be greater than t_on_s")
        return self

    def _gate(self, x, t) -> np.ndarray:
        times = np.asarray(t, dtype=float)
        return (times >= self.t_on_s) & (times < self.t_off_s)

    def _apply(self, values: np.ndarray, gate: np.ndarray) -> np.ndarray:
        if values.ndim == 1:
            return values if bool(gate) else np.zeros_like(values)
        gate = np.broadcast_to(np.asarray(gate), values.shape[:1])
        return np.where(gate[:, np.newaxis], values, 0.0)

    def E(self, x, t) -> np.ndarray:
        return self._apply(self.inner.E(x, t), self._gate(x, t))

    def B(self, x, t) -> np.ndarray:
        return self._apply(self.inner.B(x, t), self._gate(x, t))
