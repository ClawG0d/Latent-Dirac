"""Loss accounting helpers."""

from __future__ import annotations

from latent_dirac.pipeline.stage import StageResult


def total_losses(stage_results: list[StageResult]) -> float:
    return float(sum(stage.losses for stage in stage_results))


def losses_by_stage(stage_results: list[StageResult]) -> dict[str, float]:
    return {stage.stage_name: float(stage.losses) for stage in stage_results}
