"""Composite electromagnetic field built from component field models."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from latent_dirac.fields.base import Field


class CompositeField(BaseModel, Field):
    """Sum of component electromagnetic fields.

    Returns the elementwise sum of every component's electric and magnetic
    contributions, enabling beamline-like configurations without changing
    the solver contract. Fidelity follows the component models; the
    composition itself is exact superposition.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fields: list[Field]

    def E(self, x, t) -> np.ndarray:
        return self._sum_component("E", x, t)

    def B(self, x, t) -> np.ndarray:
        return self._sum_component("B", x, t)

    def _sum_component(self, component: str, x, t) -> np.ndarray:
        x_array = np.asarray(x, dtype=float)
        total = np.zeros(3) if x_array.ndim == 1 else np.zeros_like(x_array)
        for field in self.fields:
            total = total + getattr(field, component)(x, t)
        return total
