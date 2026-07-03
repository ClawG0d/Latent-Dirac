import numpy as np

from latent_dirac.core.species import positron
from latent_dirac.core.units import joule_to_ev
from latent_dirac.sources.positron_beta import BetaPlusPositronSource


def test_beta_plus_source_samples_isotropic_simplified_positron_cloud():
    source = BetaPlusPositronSource(
        half_life_s=120.0,
        beta_plus_branching_ratio=0.8,
        initial_activity_bq=2000.0,
        endpoint_energy_MeV=1.5,
        source_radius_m=0.02,
        macro_particles=256,
    )

    cloud = source.sample(np.random.default_rng(456))

    assert cloud.species == positron
    assert cloud.position_m.shape == (256, 3)
    assert np.all(np.linalg.norm(cloud.position_m, axis=1) <= 0.02)
    assert joule_to_ev(cloud.kinetic_energy_joule()).max() <= 1.5e6
    assert np.isclose(cloud.weighted_count(), 1600.0)
    assert cloud.metadata["model_type"] == "simplified"
    assert "approximate beta energy distribution" in cloud.metadata["physics_note"]
