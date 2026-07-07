"""Tiered fp32-GPU vs fp64-CPU validation (G2 of the execution plan).

Runs only where a JAX GPU backend exists (the WSL 5070 Ti box); CI and
CPU-only checkouts skip. No performance numbers here — correctness
only. Design record:
docs/superpowers/specs/2026-07-06-gpu-float32-validation-design.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")


def _gpu_devices():
    try:
        return jax.devices("gpu")
    except RuntimeError:
        return []


pytestmark = pytest.mark.skipif(not _gpu_devices(), reason="no JAX GPU backend on this host")

from latent_dirac.backends.jax_scene import run_scene_batched  # noqa: E402
from latent_dirac.scene.build import run_scene  # noqa: E402
from latent_dirac.scene.loader import load_scene, scene_from_mapping  # noqa: E402

HELLO = Path("examples/scenes/hello_beamline.yaml")
LEDGER = Path("examples/scenes/antiproton_ledger.yaml")


def trap_scene_mapping() -> dict:
    return {
        "schema_version": 1,
        "name": "gpu-trap-line",
        "seed": 2033,
        "source": {
            "type": "beta_plus",
            "label": "src",
            "params": {
                "half_life_s": 8.21e7,
                "beta_plus_branching_ratio": 0.9,
                "initial_activity_bq": 3.7e8,
                "endpoint_energy_MeV": 2.0e-5,
                "source_radius_m": 0.0005,
                "macro_particles": 32,
            },
        },
        "solver": {"type": "relativistic_boris", "dt_s": 2.0e-11, "steps": 500},
        "elements": [
            {"type": "penning_trap", "label": "trap", "v0_volt": 50.0, "d_m": 0.005, "b_tesla": 0.05}
        ],
    }


def _endpoints(scene, device):
    with jax.default_device(device):
        batched = run_scene_batched(scene, overrides={})
    return batched


def test_strict_tier_x64_gpu_matches_cpu():
    # hardware/compiler divergence isolated from precision: with x64 the
    # GPU program must match the CPU JAX program to double roundoff
    with jax.experimental.enable_x64():
        scene = load_scene(HELLO)
        gpu = _endpoints(scene, _gpu_devices()[0])
        cpu = _endpoints(scene, jax.devices("cpu")[0])
    np.testing.assert_allclose(gpu.position_m, cpu.position_m, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(gpu.momentum_kg_m_s, cpu.momentum_kg_m_s, rtol=1e-12, atol=1e-33)
    np.testing.assert_array_equal(gpu.alive, cpu.alive)


@pytest.mark.parametrize(
    "scene_source",
    ["hello", "ledger", "trap"],
)
def test_trajectory_tier_fp32_gpu_tracks_fp64_reference(scene_source):
    scene = {
        "hello": lambda: load_scene(HELLO),
        "ledger": lambda: load_scene(LEDGER),
        "trap": lambda: scene_from_mapping(trap_scene_mapping()),
    }[scene_source]()

    reference = run_scene(scene).pipeline_result.final_cloud  # NumPy float64 truth
    with jax.experimental.disable_x64():
        gpu = _endpoints(scene, _gpu_devices()[0])

    position_scale = max(float(np.max(np.abs(reference.position_m))), 1e-9)
    position_error = float(np.max(np.abs(gpu.position_m - reference.position_m))) / position_scale
    assert position_error < 1e-4, position_error

    mass_c = reference.species.mass_kg * 299792458.0
    u_ref = reference.momentum_kg_m_s / mass_c
    u_gpu = gpu.momentum_kg_m_s / mass_c
    u_scale = max(float(np.max(np.abs(u_ref))), 1e-12)
    u_error = float(np.max(np.abs(u_gpu - u_ref))) / u_scale
    assert u_error < 1e-5, u_error


def test_observable_tier_accepted_counts():
    scene = load_scene(HELLO)
    reference = run_scene(scene).pipeline_result.final_cloud
    with jax.experimental.disable_x64():
        gpu = _endpoints(scene, _gpu_devices()[0])
    # cuts are robust for this scene; allow at most one boundary particle
    mismatches = int(np.sum(gpu.alive[0] != reference.alive))
    assert mismatches <= 1, mismatches


def test_conservation_tier_u_magnitude_drift_fp32():
    from latent_dirac.scene.build import build_source

    scene = load_scene(LEDGER)  # pure uniform-B transport before the cuts
    with jax.experimental.disable_x64():
        gpu = _endpoints(scene, _gpu_devices()[0])
    # Boris conserves |u| exactly in exact arithmetic through pure B (and
    # kills only freeze particles), so the sampled initial |u| is truth;
    # fp32 drift is a rounding walk and must stay tiny
    initial = build_source(scene).sample(np.random.default_rng(scene.seed))
    mass_c = initial.species.mass_kg * 299792458.0
    u_start = np.linalg.norm(initial.momentum_kg_m_s / mass_c, axis=1)
    u_final = np.linalg.norm(np.asarray(gpu.momentum_kg_m_s[0]) / mass_c, axis=1)
    drift = float(np.max(np.abs(u_final - u_start) / np.maximum(u_start, 1e-12)))
    assert drift < 1e-4, drift
