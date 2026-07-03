"""Pipeline stages and per-stage results."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from latent_dirac.state.particle_cloud import ParticleCloud


class StageResult(BaseModel):
    stage_name: str
    input_weighted_count: float
    output_weighted_count: float
    transmission: float
    losses: float
    metadata: dict = Field(default_factory=dict)


class Stage(BaseModel):
    """A named cloud-to-cloud operation with loss accounting."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    action: Callable[[ParticleCloud], ParticleCloud]

    def __init__(
        self,
        name: str | None = None,
        action: Callable[[ParticleCloud], ParticleCloud] | None = None,
        **data,
    ):
        if name is not None:
            data["name"] = name
        if action is not None:
            data["action"] = action
        super().__init__(**data)

    def run(self, cloud: ParticleCloud) -> tuple[ParticleCloud, StageResult]:
        input_count = cloud.weighted_count()
        output_cloud = self.action(cloud)
        output_count = output_cloud.weighted_count()
        transmission = output_count / input_count if input_count > 0.0 else 0.0
        return output_cloud, StageResult(
            stage_name=self.name,
            input_weighted_count=input_count,
            output_weighted_count=output_count,
            transmission=transmission,
            losses=input_count - output_count,
            metadata={},
        )
