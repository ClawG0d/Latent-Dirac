import numpy as np

from latent_dirac.core.species import positron
from latent_dirac.sources.positron_pair import PositronPairSource


def test_positron_pair_source_samples_weighted_positron_cloud():
    source = PositronPairSource(
        primary_count=1000,
        yield_eplus_per_primary=0.05,
        mean_energy_MeV=5.0,
        energy_spread_MeV=0.5,
        angular_rms_rad=0.02,
        source_sigma_m=1.0e-3,
        bunch_length_s=1.0e-12,
        macro_particles=128,
    )

    cloud = source.sample(np.random.default_rng(123))

    assert cloud.species == positron
    assert cloud.position_m.shape == (128, 3)
    assert cloud.momentum_kg_m_s.shape == (128, 3)
    assert np.isclose(cloud.weighted_count(), 50.0)
    assert cloud.metadata["model_type"] == "parameterized"
    assert "not full shower physics" in cloud.metadata["physics_note"]
