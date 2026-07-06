"""Tests for the buffer_gas_cooling element (parameterized stand-in).

Spec: 2026-07-06 buffer-gas collisions, implementation slice 1. A
standalone cooling region: Poisson collisions over a hold time, each
either cooling (energy loss floored at (3/2)kT) or a Ps-formation loss.
"""

from __future__ import annotations

import numpy as np
import pytest

from latent_dirac.core.constants import k_B
from latent_dirac.scene.build import build_source, run_scene
from latent_dirac.scene.loader import scene_from_mapping

FLOOR_300K = 1.5 * k_B * 300.0  # (3/2) k_B T at 300 K, in joules


def cooling_scene(
    hold_time_s=1e-7,
    collision_rate_hz=2e8,
    energy_loss_ev=8.5,
    ps_fraction=0.005,
    gas_temperature_k=300.0,
    mean_energy_MeV=1e-4,
    macro_particles=500,
    seed=11,
    lead_aperture=False,
):
    elements = []
    if lead_aperture:
        elements.append({"type": "aperture", "label": "iris", "radius_m": 5e-4, "z_m": 0.0})
    elements.append(
        {
            "type": "buffer_gas_cooling",
            "label": "cooler",
            "hold_time_s": hold_time_s,
            "collision_rate_hz": collision_rate_hz,
            "energy_loss_ev": energy_loss_ev,
            "ps_fraction": ps_fraction,
            "gas_temperature_k": gas_temperature_k,
        }
    )
    elements.append({"type": "monitor", "label": "end"})
    return scene_from_mapping(
        {
            "schema_version": 1,
            "name": "buffer-gas",
            "seed": seed,
            "solver": {"dt_s": 1e-9, "steps": 1},
            "source": {
                "type": "positron_pair",
                "label": "pairs",
                "params": {
                    "primary_count": 10000,
                    "yield_eplus_per_primary": 0.02,
                    "mean_energy_MeV": mean_energy_MeV,
                    "energy_spread_MeV": mean_energy_MeV * 0.2,
                    "angular_rms_rad": 0.2,
                    "source_sigma_m": 1e-3,
                    "bunch_length_s": 1e-10,
                    "macro_particles": macro_particles,
                },
            },
            "elements": elements,
        }
    )


def test_cloud_cools_toward_the_floor():
    scene = cooling_scene()
    initial = build_source(scene).sample(np.random.default_rng(scene.seed))
    initial_mean = initial.kinetic_energy_joule().mean()

    result = run_scene(scene)
    final = result.pipeline_result.final_cloud
    ke = final.kinetic_energy_joule()[final.alive]
    # many collisions at 8.5 eV cool the ~100 eV cloud toward (3/2)kT; the
    # mean drops far below the injection energy and nothing goes sub-floor
    # (a Poisson tail of under-collided particles keeps a few above the
    # floor, so the "everyone parked at the floor" check lives in the
    # huge-hold test below)
    assert ke.mean() < 0.01 * initial_mean
    assert np.all(ke >= FLOOR_300K - 1e-30)  # never below the floor


def test_energy_floor_respected_with_huge_hold():
    scene = cooling_scene(hold_time_s=1e-6)  # 10x more collisions
    result = run_scene(scene)
    final = result.pipeline_result.final_cloud
    ke = final.kinetic_energy_joule()[final.alive]
    np.testing.assert_allclose(ke, FLOOR_300K, rtol=1e-6)


def test_ps_loss_is_ledgered():
    scene = cooling_scene(ps_fraction=0.05)
    result = run_scene(scene)
    final = result.pipeline_result.final_cloud
    killed = ~final.alive
    assert killed.any() and not killed.all()
    assert np.all(final.lost_at_element[killed] == 0)  # cooler is stage 0
    assert np.all(final.lost_at_element[final.alive] == -1)


def test_zero_hold_is_a_noop():
    scene = cooling_scene(hold_time_s=0.0)
    initial = build_source(scene).sample(np.random.default_rng(scene.seed))
    result = run_scene(scene)
    final = result.pipeline_result.final_cloud
    assert final.alive.all()
    np.testing.assert_allclose(
        final.kinetic_energy_joule(), initial.kinetic_energy_joule(), rtol=1e-12
    )


def test_cooling_preserves_momentum_direction():
    # a cooling collision only shrinks |p| at this tier; direction is kept
    scene = cooling_scene(ps_fraction=0.0)  # no losses, so ids align 1:1
    initial = build_source(scene).sample(np.random.default_rng(scene.seed))
    result = run_scene(scene)
    final = result.pipeline_result.final_cloud
    p0 = initial.momentum_kg_m_s
    p1 = final.momentum_kg_m_s
    # compare normalized directions where the initial momentum is nonzero
    n0 = np.linalg.norm(p0, axis=1)
    mask = n0 > 0
    u0 = p0[mask] / n0[mask, None]
    u1 = p1[mask] / np.linalg.norm(p1[mask], axis=1)[:, None]
    dots = np.einsum("ij,ij->i", u0, u1)
    np.testing.assert_allclose(dots, 1.0, atol=1e-9)


def test_deterministic_given_seed():
    a = run_scene(cooling_scene(seed=5))
    b = run_scene(cooling_scene(seed=5))
    np.testing.assert_array_equal(
        a.pipeline_result.final_cloud.alive, b.pipeline_result.final_cloud.alive
    )
    np.testing.assert_allclose(
        a.pipeline_result.final_cloud.momentum_kg_m_s,
        b.pipeline_result.final_cloud.momentum_kg_m_s,
    )


def test_dead_on_entry_untouched():
    scene = cooling_scene(lead_aperture=True, ps_fraction=0.05)
    result = run_scene(scene)
    final = result.pipeline_result.final_cloud
    dead_after_iris = final.lost_at_element == 0  # aperture is stage 0
    assert dead_after_iris.any()
    assert not final.alive[dead_after_iris].any()  # never revived


def test_schema_rejects_bad_params():
    from pydantic import ValidationError

    for bad in (
        {"ps_fraction": 1.5},
        {"ps_fraction": -0.1},
        {"collision_rate_hz": 0.0},
        {"energy_loss_ev": 0.0},
        {"hold_time_s": -1.0},
    ):
        with pytest.raises(ValidationError):
            cooling_scene(**bad)


def test_jax_backend_rejects_the_element():
    pytest.importorskip("jax")
    from latent_dirac.backends.jax_scene import run_scene_batched

    with pytest.raises(ValueError, match="JAX backend"):
        run_scene_batched(cooling_scene(macro_particles=32), overrides={})
