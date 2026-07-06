"""Tests for the residual-gas storage-lifetime loss element.

Spec: 2026-07-06 residual-gas storage lifetime. Stochastic per-particle
annihilation-on-residual-gas over a hold time; parameterized tier.
"""

from __future__ import annotations

import numpy as np
import pytest

from latent_dirac.core.species import antiproton, positron
from latent_dirac.scene.build import run_scene
from latent_dirac.scene.loader import scene_from_mapping


def loss_scene(mean_lifetime_s, hold_time_s, species=antiproton, count=4000, seed=17):
    return scene_from_mapping(
        {
            "schema_version": 1,
            "name": "storage-lifetime",
            "seed": seed,
            "solver": {"dt_s": 1e-9, "steps": 1},
            "source": {
                "type": "cold_uniform_sphere",
                "label": "cloud",
                "params": {
                    "species_name": species.name,
                    "macro_particles": count,
                    "radius_m": 1e-3,
                    "weight": 1.0,
                },
            },
            "elements": [
                {
                    "type": "residual_gas_loss",
                    "label": "storage",
                    "mean_lifetime_s": mean_lifetime_s,
                    "hold_time_s": hold_time_s,
                },
                {"type": "monitor", "label": "end"},
            ],
        }
    )


def survivors(result):
    cloud = result.pipeline_result.final_cloud
    return cloud.alive


def test_survival_fraction_matches_exponential():
    tau, hold = 100.0, 50.0
    result = run_scene(loss_scene(tau, hold, count=8000))
    alive = survivors(result)
    fraction = alive.mean()
    expected = np.exp(-hold / tau)
    # binomial std ~ sqrt(p(1-p)/N) ~ 0.006 here; 4 sigma bound
    assert abs(fraction - expected) < 0.025


def test_deterministic_given_seed():
    a = run_scene(loss_scene(100.0, 50.0, seed=5))
    b = run_scene(loss_scene(100.0, 50.0, seed=5))
    np.testing.assert_array_equal(survivors(a), survivors(b))

    c = run_scene(loss_scene(100.0, 50.0, seed=6))
    # different seed: same fraction band, different survivor set
    assert not np.array_equal(survivors(a), survivors(c))


def test_killed_particles_stamped_in_ledger():
    result = run_scene(loss_scene(100.0, 50.0))
    cloud = result.pipeline_result.final_cloud
    # storage is stage index 0
    killed = ~cloud.alive
    assert np.all(cloud.lost_at_element[killed] == 0)
    assert np.all(cloud.lost_at_element[cloud.alive] == -1)


def test_survivors_age_by_hold_time():
    tau, hold = 100.0, 30.0
    result = run_scene(loss_scene(tau, hold))
    cloud = result.pipeline_result.final_cloud
    # source starts every particle at t = 0
    np.testing.assert_allclose(cloud.time_s[cloud.alive], hold)
    np.testing.assert_allclose(cloud.time_s[~cloud.alive], 0.0)


def test_zero_hold_keeps_all_alive():
    result = run_scene(loss_scene(100.0, 0.0))
    assert survivors(result).all()


def test_long_hold_kills_nearly_all():
    result = run_scene(loss_scene(1.0, 20.0, count=4000))
    assert survivors(result).mean() < 0.01


@pytest.mark.parametrize("species", [antiproton, positron])
def test_species_agnostic(species):
    result = run_scene(loss_scene(100.0, 70.0, species=species, count=4000))
    fraction = survivors(result).mean()
    assert abs(fraction - np.exp(-0.7)) < 0.03


def test_dead_on_entry_stay_dead_and_are_not_restamped():
    # a preceding aperture kills some particles; the loss element must not
    # resurrect or re-stamp them
    scene = scene_from_mapping(
        {
            "schema_version": 1,
            "name": "aperture-then-storage",
            "seed": 3,
            "solver": {"dt_s": 1e-9, "steps": 1},
            "source": {
                "type": "cold_uniform_sphere",
                "label": "cloud",
                "params": {
                    "species_name": "antiproton",
                    "macro_particles": 2000,
                    "radius_m": 5e-3,
                    "weight": 1.0,
                },
            },
            "elements": [
                {"type": "aperture", "label": "collimator", "radius_m": 2e-3, "z_m": 0.0},
                {"type": "monitor", "label": "after-collimator"},
                {
                    "type": "residual_gas_loss",
                    "label": "storage",
                    "mean_lifetime_s": 100.0,
                    "hold_time_s": 50.0,
                },
            ],
        }
    )
    result = run_scene(scene)
    cloud = result.pipeline_result.final_cloud
    # identify aperture victims independently of the final stamp value, so a
    # re-stamping regression (0 -> storage's index) would actually be caught
    dead_after_aperture = ~result.monitors["after-collimator"].alive
    assert dead_after_aperture.any()  # some died at the collimator
    # they stay dead and keep their stage-0 stamp — never revived, never re-stamped
    assert not cloud.alive[dead_after_aperture].any()
    assert np.all(cloud.lost_at_element[dead_after_aperture] == 0)


def test_validation_rejects_bad_params():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        loss_scene(0.0, 10.0)  # mean_lifetime_s must be > 0
    with pytest.raises(ValidationError):
        loss_scene(100.0, -1.0)  # hold_time_s must be >= 0


def test_jax_backend_rejects_the_element():
    pytest.importorskip("jax")
    from latent_dirac.backends.jax_scene import run_scene_batched

    scene = loss_scene(100.0, 50.0, count=64)
    with pytest.raises(ValueError, match="not supported by the JAX backend"):
        run_scene_batched(scene, overrides={})
