import numpy as np

from latent_dirac.beamline.aperture import Aperture
from latent_dirac.beamline.momentum_window import MomentumWindow
from latent_dirac.core.species import positron
from latent_dirac.pipeline.runner import PipelineRunner
from latent_dirac.pipeline.stage import Stage
from latent_dirac.state.particle_state import ParticleState


def test_pipeline_reports_per_stage_transmission_and_losses():
    cloud = ParticleState(
        species=positron,
        position_m=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.20, 0.0, 0.0],
                [0.15, 0.0, 0.0],
            ]
        ),
        momentum_kg_m_s=np.array(
            [
                [2.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        time_s=np.zeros(4),
        weight=np.ones(4),
        alive=np.ones(4, dtype=bool),
        particle_id=np.arange(4),
        parent_id=np.full(4, -1),
        metadata={},
    )

    result = PipelineRunner(
        stages=[
            Stage("aperture", Aperture(radius_m=0.1, z_m=0.0).apply),
            Stage("momentum window", MomentumWindow(1.0, 3.0).apply),
        ]
    ).run(cloud)

    assert [stage.stage_name for stage in result.stage_results] == ["aperture", "momentum window"]
    assert result.stage_results[0].input_weighted_count == 4.0
    assert result.stage_results[0].output_weighted_count == 2.0
    assert result.stage_results[0].losses == 2.0
    assert result.stage_results[0].transmission == 0.5
    assert result.stage_results[1].input_weighted_count == 2.0
    assert result.stage_results[1].output_weighted_count == 1.0
    assert result.stage_results[1].losses == 1.0
    assert result.final_cloud.weighted_count() == 1.0
