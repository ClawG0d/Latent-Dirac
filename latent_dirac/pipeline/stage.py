"""Pipeline stages and per-stage results."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from latent_dirac.state.particle_state import ParticleState


class StageResult(BaseModel):
    stage_name: str
    input_weighted_count: float
    output_weighted_count: float
    transmission: float
    losses: float
    metadata: dict = Field(default_factory=dict)


class Stage(BaseModel):
    """A named state-to-state operation with loss accounting.

    The pipeline layer owns the loss ledger: particles killed by this
    stage's action are stamped with the stage index in `lost_at_element`,
    so elements themselves stay ledger-agnostic.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    action: Callable[[ParticleState], ParticleState]

    def __init__(
        self,
        name: str | None = None,
        action: Callable[[ParticleState], ParticleState] | None = None,
        **data,
    ):
        if name is not None:
            data["name"] = name
        if action is not None:
            data["action"] = action
        super().__init__(**data)

    def run(
        self, state: ParticleState, stage_index: int | None = None
    ) -> tuple[ParticleState, StageResult]:
        input_count = state.weighted_count()
        alive_before = state.alive.copy()

        output_state = self.action(state)

        if np.any(~alive_before & output_state.alive):
            raise ValueError(
                f"stage {self.name!r} resurrected particles; actions must only "
                "AND the alive mask (see ParticleState.apply_alive_mask)"
            )

        newly_dead = alive_before & ~output_state.alive
        if stage_index is not None and np.any(newly_dead):
            ledger = output_state.lost_at_element
            output_state.lost_at_element = np.where(newly_dead, np.int32(stage_index), ledger)

        output_count = output_state.weighted_count()
        transmission = output_count / input_count if input_count > 0.0 else 0.0
        return output_state, StageResult(
            stage_name=self.name,
            input_weighted_count=input_count,
            output_weighted_count=output_count,
            transmission=transmission,
            losses=input_count - output_count,
            metadata={},
        )
