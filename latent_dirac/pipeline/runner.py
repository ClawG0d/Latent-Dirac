"""Pipeline runner."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from latent_dirac.pipeline.stage import Stage, StageResult
from latent_dirac.state.particle_cloud import ParticleCloud


class PipelineResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_cloud: ParticleCloud
    stage_results: list[StageResult]


class PipelineRunner(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    stages: list[Stage]

    def run(self, cloud: ParticleCloud) -> PipelineResult:
        current = cloud
        stage_results: list[StageResult] = []
        for stage in self.stages:
            current, result = stage.run(current)
            stage_results.append(result)
        return PipelineResult(final_cloud=current, stage_results=stage_results)
