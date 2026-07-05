import numpy as np
import pytest

from latent_dirac.core.species import positron
from latent_dirac.diagnostics.loss_ledger import loss_ledger
from latent_dirac.pipeline.runner import PipelineRunner
from latent_dirac.pipeline.stage import Stage
from latent_dirac.state.particle_state import ParticleState


def make_state(count: int = 6) -> ParticleState:
    return ParticleState(
        species=positron,
        position_m=np.zeros((count, 3)),
        momentum_kg_m_s=np.full((count, 3), 1.0e-22),
        time_s=np.zeros(count),
        weight=np.full(count, 3.0),
        alive=np.ones(count, dtype=bool),
        particle_id=np.arange(count),
        parent_id=np.full(count, -1),
    )


def kill_first(n: int):
    def action(state: ParticleState) -> ParticleState:
        result = state.copy()
        mask = np.ones(result.alive.shape, dtype=bool)
        mask[:n] = False
        result.apply_alive_mask(mask)
        return result

    return action


def passthrough(state: ParticleState) -> ParticleState:
    return state


def test_stage_stamps_lost_at_element_with_stage_index():
    state = make_state()
    stage = Stage("collimator", kill_first(2))

    output, result = stage.run(state, stage_index=3)

    np.testing.assert_array_equal(output.lost_at_element[:2], [3, 3])
    np.testing.assert_array_equal(output.lost_at_element[2:], [-1, -1, -1, -1])
    assert result.losses == pytest.approx(6.0)


def test_earlier_stamp_is_never_overwritten():
    state = make_state()
    runner = PipelineRunner(
        stages=[
            Stage("first-cut", kill_first(2)),
            Stage("monitor", passthrough),
            Stage("second-cut", kill_first(4)),
        ]
    )
    result = runner.run(state)
    ledger_channel = result.final_cloud.lost_at_element

    np.testing.assert_array_equal(ledger_channel, [0, 0, 2, 2, -1, -1])


def test_resurrecting_particles_is_rejected():
    def necromancer(state: ParticleState) -> ParticleState:
        result = state.copy()
        result.alive = np.ones(result.alive.shape, dtype=bool)
        return result

    state = make_state()
    state.apply_alive_mask(np.array([False, True, True, True, True, True]))
    stage = Stage("necromancer", necromancer)

    with pytest.raises(ValueError, match="resurrect"):
        stage.run(state, stage_index=1)


def test_stage_run_without_index_does_not_stamp():
    state = make_state()
    output, _ = Stage("collimator", kill_first(2)).run(state)

    np.testing.assert_array_equal(output.lost_at_element, [-1] * 6)


def test_loss_ledger_rejects_reserved_stage_name():
    state = make_state()
    runner = PipelineRunner(stages=[Stage("surviving", kill_first(1))])
    result = runner.run(state)

    with pytest.raises(ValueError, match="surviving"):
        loss_ledger(result.final_cloud, result.stage_results)


def test_loss_ledger_matches_stage_results():
    state = make_state()
    runner = PipelineRunner(
        stages=[
            Stage("first-cut", kill_first(1)),
            Stage("second-cut", kill_first(3)),
        ]
    )
    result = runner.run(state)

    ledger = loss_ledger(result.final_cloud, result.stage_results)

    assert ledger["first-cut"] == pytest.approx(3.0)
    assert ledger["second-cut"] == pytest.approx(6.0)
    assert ledger["surviving"] == pytest.approx(9.0)
    for stage_result in result.stage_results:
        assert ledger[stage_result.stage_name] == pytest.approx(stage_result.losses)
