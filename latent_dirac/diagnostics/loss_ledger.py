"""Per-particle loss ledger reconstruction from the State channel."""

from __future__ import annotations

import numpy as np

from latent_dirac.pipeline.stage import StageResult
from latent_dirac.state.particle_state import ParticleState


def loss_ledger(
    final_state: ParticleState,
    stage_results: list[StageResult],
) -> dict[str, float]:
    """Reconstruct weighted losses per stage from `lost_at_element`.

    Returns a mapping of stage name to the weighted loss stamped at that
    stage, plus a `"surviving"` entry. The result must agree with the
    per-stage `StageResult.losses` accounting; a mismatch indicates a
    stage action manipulated `alive` outside the pipeline.
    """

    ledger: dict[str, float] = {}
    for stage_index, stage_result in enumerate(stage_results):
        if stage_result.stage_name == "surviving":
            raise ValueError('"surviving" is a reserved ledger key; rename the stage')
        lost_here = final_state.lost_at_element == stage_index
        ledger[stage_result.stage_name] = float(np.sum(final_state.weight[lost_here]))
    ledger["surviving"] = final_state.weighted_count()
    return ledger
