"""Random number helper utilities."""

from __future__ import annotations

import numpy as np


def rng_from_seed(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)
