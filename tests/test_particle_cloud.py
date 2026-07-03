import numpy as np

from latent_dirac.core.species import positron
from latent_dirac.core.units import kinetic_energy_to_momentum_magnitude, mev_to_joule
from latent_dirac.state.particle_cloud import ParticleCloud


def make_cloud():
    p0 = kinetic_energy_to_momentum_magnitude(mev_to_joule(1.0), positron.mass_kg)
    return ParticleCloud(
        species=positron,
        position_m=np.zeros((3, 3)),
        momentum_kg_m_s=np.array([[p0, 0.0, 0.0], [0.0, p0, 0.0], [0.0, 0.0, p0]]),
        time_s=np.zeros(3),
        weight=np.array([1.0, 2.0, 4.0]),
        alive=np.array([True, False, True]),
        particle_id=np.array([0, 1, 2]),
        parent_id=np.array([-1, -1, -1]),
        metadata={"source": "test"},
    )


def test_particle_cloud_reports_weighted_count_and_kinematics():
    cloud = make_cloud()
    assert cloud.weighted_count() == 5.0
    assert cloud.gamma().shape == (3,)
    assert cloud.velocity().shape == (3, 3)
    assert np.all(np.linalg.norm(cloud.velocity(), axis=1) < 299_792_458.0)
    assert cloud.kinetic_energy_joule().shape == (3,)
    assert cloud.mean_kinetic_energy_joule() > 0.0


def test_particle_cloud_copy_is_deep_for_arrays_and_metadata():
    cloud = make_cloud()
    copied = cloud.copy()
    copied.position_m[0, 0] = 1.0
    copied.metadata["source"] = "changed"
    assert cloud.position_m[0, 0] == 0.0
    assert cloud.metadata["source"] == "test"


def test_particle_cloud_apply_alive_mask_combines_with_existing_alive_state():
    cloud = make_cloud()
    cloud.apply_alive_mask(np.array([True, True, False]))
    assert cloud.alive.tolist() == [True, False, False]
    assert cloud.weighted_count() == 1.0
