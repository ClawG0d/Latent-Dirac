import numpy as np

from latent_dirac.core.species import antiproton
from latent_dirac.core.units import momentum_si_to_gev_c
from latent_dirac.sources.antiproton_surrogate import AntiprotonSurrogateSource


def test_antiproton_surrogate_source_samples_weighted_antiproton_cloud():
    source = AntiprotonSurrogateSource(
        primary_proton_count=10_000,
        yield_pbar_per_primary_in_acceptance=1.0e-4,
        central_momentum_GeV_c=3.0,
        momentum_spread_fraction=0.05,
        angular_rms_rad=0.01,
        source_sigma_m=1.0e-3,
        bunch_length_s=2.0e-12,
        macro_particles=128,
    )

    cloud = source.sample(np.random.default_rng(789))
    momentum = np.linalg.norm(cloud.momentum_kg_m_s, axis=1)

    assert cloud.species == antiproton
    assert cloud.position_m.shape == (128, 3)
    assert np.isclose(cloud.weighted_count(), 1.0)
    assert np.isclose(momentum_si_to_gev_c(momentum).mean(), 3.0, rtol=0.15)
    assert cloud.metadata["model_type"] == "surrogate"
    assert "not a detailed target model" in cloud.metadata["physics_note"]
