"""Pipeline runner."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from latent_dirac.pipeline.stage import Stage, StageResult
from latent_dirac.state.particle_state import ParticleState


class PipelineResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_cloud: ParticleState
    stage_results: list[StageResult]


class PipelineRunner(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    stages: list[Stage]

    def run(self, state: ParticleState) -> PipelineResult:
        current = state
        stage_results: list[StageResult] = []
        for stage_index, stage in enumerate(self.stages):
            current, result = stage.run(current, stage_index=stage_index)
            stage_results.append(result)
        return PipelineResult(final_cloud=current, stage_results=stage_results)
