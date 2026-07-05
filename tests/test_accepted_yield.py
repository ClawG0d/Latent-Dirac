import numpy as np
import pytest

from latent_dirac.core.species import positron
from latent_dirac.diagnostics.accepted_yield import accepted_yield, accepted_yield_from_cloud
from latent_dirac.state.particle_state import ParticleState


def test_accepted_yield_divides_accepted_weight_by_primary_count():
    assert accepted_yield(accepted_weighted_count=25.0, primary_count=100.0) == 0.25


def test_accepted_yield_rejects_non_positive_primary_count():
    with pytest.raises(ValueError, match="primary_count"):
        accepted_yield(accepted_weighted_count=1.0, primary_count=0.0)


def test_accepted_yield_from_cloud_uses_alive_weighted_count():
    cloud = ParticleState(
        species=positron,
        position_m=np.zeros((3, 3)),
        momentum_kg_m_s=np.zeros((3, 3)),
        time_s=np.zeros(3),
        weight=np.array([1.0, 2.0, 4.0]),
        alive=np.array([True, False, True]),
        particle_id=np.arange(3),
        parent_id=np.full(3, -1),
        metadata={},
    )
    assert accepted_yield_from_cloud(cloud, primary_count=10.0) == 0.5
