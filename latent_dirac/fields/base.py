"""Field interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Field(ABC):
    """Electromagnetic field interface."""

    @abstractmethod
    def E(self, x, t) -> np.ndarray:
        """Return electric field in V/m at positions `x` and time `t`."""
        raise NotImplementedError

    @abstractmethod
    def B(self, x, t) -> np.ndarray:
        """Return magnetic field in tesla at positions `x` and time `t`."""
        raise NotImplementedError


def broadcast_vector(vector, x) -> np.ndarray:
    x_array = np.asarray(x, dtype=float)
    vector_array = np.asarray(vector, dtype=float)
    if vector_array.shape != (3,):
        raise ValueError("field vector must have shape (3,)")
    if x_array.ndim == 1:
        return vector_array.copy()
    return np.broadcast_to(vector_array, (x_array.shape[0], 3)).copy()
