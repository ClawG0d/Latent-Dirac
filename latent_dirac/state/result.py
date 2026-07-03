"""Top-level simulation result containers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from latent_dirac.state.particle_cloud import ParticleCloud


class SimulationResult(BaseModel):
    """Final cloud plus structured run metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_cloud: ParticleCloud
    metadata: dict
