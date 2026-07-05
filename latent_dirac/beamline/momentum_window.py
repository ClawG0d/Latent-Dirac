"""Momentum-window acceptance."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, model_validator

from latent_dirac.beamline.element import BeamlineElement
from latent_dirac.state.particle_state import ParticleState


class MomentumWindow(BaseModel, BeamlineElement):
    """Accept particles whose momentum magnitude lies in a fixed SI interval."""

    min_momentum_kg_m_s: float
    max_momentum_kg_m_s: float

    def __init__(
        self,
        min_momentum_kg_m_s: float | None = None,
        max_momentum_kg_m_s: float | None = None,
        **data,
    ):
        if min_momentum_kg_m_s is not None:
            data["min_momentum_kg_m_s"] = min_momentum_kg_m_s
        if max_momentum_kg_m_s is not None:
            data["max_momentum_kg_m_s"] = max_momentum_kg_m_s
        super().__init__(**data)

    @model_validator(mode="after")
    def _validate_window(self):
        if self.min_momentum_kg_m_s < 0.0:
            raise ValueError("min_momentum_kg_m_s must be non-negative")
        if self.max_momentum_kg_m_s < self.min_momentum_kg_m_s:
            raise ValueError("max_momentum_kg_m_s must be >= min_momentum_kg_m_s")
        return self

    def apply(self, cloud: ParticleState) -> ParticleState:
        result = cloud.copy()
        momentum = np.linalg.norm(result.momentum_kg_m_s, axis=1)
        result.apply_alive_mask(
            (momentum >= self.min_momentum_kg_m_s)
            & (momentum <= self.max_momentum_kg_m_s)
        )
        result.metadata.setdefault("momentum_windows", []).append(
            {
                "min_momentum_kg_m_s": self.min_momentum_kg_m_s,
                "max_momentum_kg_m_s": self.max_momentum_kg_m_s,
            }
        )
        return result
