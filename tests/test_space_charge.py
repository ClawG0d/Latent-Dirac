"""Tests for the mean-field space-charge model.

Spec: 2026-07-05 mean-field space charge (closed-loop v1, item 4).
Parameterized uniform-sphere mean field; trap regime (beta << 1).
"""

from __future__ import annotations

import numpy as np
import pytest

from latent_dirac.core.constants import epsilon_0
from latent_dirac.core.species import antiproton, positron
from latent_dirac.fields.space_charge import (
    UniformSphereSelfField,
    fit_uniform_sphere,
)
from latent_dirac.scene.build import run_scene
from latent_dirac.scene.loader import scene_from_mapping
from latent_dirac.sources.base import particle_arrays
from latent_dirac.state.particle_state import ParticleState


def make_cold_sphere(
    count: int = 200, species=positron, radius_m: float = 1e-3, seed: int = 7
) -> ParticleState:
    rng = np.random.default_rng(seed)
    # uniform sampling inside a sphere
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, np.newaxis]
    r = radius_m * rng.uniform(0.0, 1.0, count) ** (1.0 / 3.0)
    return ParticleState(
        species=species,
        position_m=direction * r[:, np.newaxis],
        momentum_kg_m_s=np.zeros((count, 3)),
        time_s=np.zeros(count),
        weight=np.full(count, 1e6),
        **particle_arrays(count),
    )


def test_field_formula_interior_exterior_continuity():
    field = UniformSphereSelfField(
        center_m=(0.0, 0.0, 0.0), radius_m=2e-3, total_charge_c=1e-12
    )
    r = field.radius_m
    e_surface = field.E(np.array([[r, 0.0, 0.0]]), 0.0)[0, 0]
    e_half = field.E(np.array([[r / 2, 0.0, 0.0]]), 0.0)[0, 0]
    e_double = field.E(np.array([[2 * r, 0.0, 0.0]]), 0.0)[0, 0]

    analytic_surface = field.total_charge_c / (4 * np.pi * epsilon_0 * r**2)
    assert e_surface == pytest.approx(analytic_surface, rel=1e-12)
    assert e_half == pytest.approx(e_surface / 2, rel=1e-12)  # linear inside
    assert e_double == pytest.approx(e_surface / 4, rel=1e-12)  # 1/r^2 outside
    assert e_surface > 0  # outward for positive charge
    np.testing.assert_allclose(
        field.B(np.array([[r, 0.0, 0.0]]), 0.0), 0.0
    )


def test_fit_uniform_sphere_weighted_alive_only():
    state = make_cold_sphere(count=100)
    state.alive[:50] = False
    field = fit_uniform_sphere(state)

    alive_weight = state.weight[state.alive].sum()
    expected_q = alive_weight * state.species.charge_c
    assert field.total_charge_c == pytest.approx(expected_q, rel=1e-12)

    dead_only = make_cold_sphere(count=10)
    dead_only.alive[:] = False
    assert fit_uniform_sphere(dead_only) is None


@pytest.mark.parametrize("species", [positron, antiproton])
def test_cold_sphere_expands_regardless_of_charge_sign(species):
    scene = scene_from_mapping(sphere_scene_mapping(species, steps=60))
    result = run_scene(scene, record_trajectories=True)

    trajectory = result.trajectories["free-space"]
    rms = np.sqrt((trajectory**2).sum(axis=2).mean(axis=1))
    assert np.all(np.diff(rms) > 0), "same-species mean field must self-repel"


def test_surface_particle_leading_order_kick():
    # early-time displacement of the outermost particle ~ (qE/m) t^2 / 2;
    # t must stay well below 1/omega_p for leading order to hold
    from latent_dirac.scene.build import build_source

    mapping = sphere_scene_mapping(positron, steps=5, seed=11)
    mapping["solver"] = {"dt_s": 5e-11, "steps": 5}
    scene = scene_from_mapping(mapping)
    initial = build_source(scene).sample(np.random.default_rng(scene.seed))
    outer = int(np.argmax(np.linalg.norm(initial.position_m, axis=1)))
    field = fit_uniform_sphere(initial)
    e_here = field.E(initial.position_m[outer][np.newaxis, :], 0.0)[0]

    result = run_scene(scene, record_trajectories=True)
    trajectory = result.trajectories["free-space"]

    # kick-then-drift from rest: the discrete sum is a*dt^2*n(n+1)/2
    # (-> a*t^2/2 as n grows); using the exact sum also pins the kernel's
    # stepping structure
    n = 5
    dt = scene.solver.dt_s
    acceleration = positron.charge_c * e_here / positron.mass_kg
    displacement = trajectory[-1, outer] - trajectory[0, outer]
    expected = acceleration * dt**2 * n * (n + 1) / 2
    np.testing.assert_allclose(displacement, expected, rtol=0.02)


def test_total_momentum_conserved_under_self_field():
    scene = scene_from_mapping(sphere_scene_mapping(positron, steps=40))
    result = run_scene(scene)
    final = result.pipeline_result.final_cloud

    total_p = (final.momentum_kg_m_s * final.weight[:, np.newaxis]).sum(axis=0)
    # the fitted field exerts zero net force on interior particles by the
    # centroid definition; near-surface Coulomb tails leave a sampling
    # fluctuation, so assert "no systematic push", not exact zero
    # bound calibrated with ~3x headroom over sampled seeds (worst ~7e-3):
    # numpy Generator streams are not guaranteed stable across releases
    scale = (np.linalg.norm(final.momentum_kg_m_s, axis=1) * final.weight).sum()
    assert np.linalg.norm(total_p) < 2e-2 * scale


def test_trap_suppresses_expansion():
    # dt resolves omega_c (0.02 T -> omega_c*dt ~ 0.18) and omega_z;
    # confinement condition omega_p^2/3 << omega_z^2 holds by construction
    solver = {"dt_s": 5e-11, "steps": 500}
    free_mapping = sphere_scene_mapping(positron, steps=500)
    free_mapping["solver"] = solver
    trapped_mapping = sphere_scene_mapping(positron, steps=500)
    trapped_mapping["solver"] = solver
    trapped_mapping["elements"] = [
        {
            "type": "penning_trap",
            "label": "trap",
            "v0_volt": 40.0,
            "d_m": 5e-3,
            "b_tesla": 0.02,
            "space_charge": "uniform_sphere",
        }
    ]

    rms_free = final_rms(run_scene(scene_from_mapping(free_mapping)))
    rms_trapped = final_rms(run_scene(scene_from_mapping(trapped_mapping)))
    initial_rms = np.sqrt(3.0 / 5.0) * 1e-3  # uniform sphere, R = 1 mm

    assert rms_trapped < rms_free / 3
    assert rms_trapped < 2 * initial_rms  # confined, not blowing up


def test_jax_backend_rejects_space_charge():
    jnp = pytest.importorskip("jax")  # noqa: F841
    from latent_dirac.backends.jax_scene import run_scene_batched

    scene = scene_from_mapping(sphere_scene_mapping(positron, steps=4))
    with pytest.raises(ValueError, match="space_charge"):
        run_scene_batched(scene, overrides={})


def test_differentiable_objective_rejects_space_charge():
    pytest.importorskip("jax")
    from latent_dirac.backends.differentiable import make_differentiable_objective

    scene = scene_from_mapping(sphere_scene_mapping(positron, steps=4))
    with pytest.raises(ValueError, match="space_charge"):
        make_differentiable_objective(scene, variables=["free-space.E_vector_v_m[0]"])


def test_schema_rejects_unknown_model():
    from pydantic import ValidationError

    mapping = sphere_scene_mapping(positron, steps=4)
    mapping["elements"][0]["space_charge"] = "pic"
    with pytest.raises(ValidationError):
        scene_from_mapping(mapping)


# --- helpers -----------------------------------------------------------


def sphere_scene_mapping(species, steps: int, seed: int = 7) -> dict:
    return {
        "schema_version": 1,
        "name": f"space-charge-{species.name}",
        "seed": seed,
        "solver": {"dt_s": 2e-9, "steps": steps},
        "source": {
            "type": "cold_uniform_sphere",
            "label": "cold-sphere",
            "params": {
                "species_name": species.name,
                "macro_particles": 200,
                "radius_m": 1e-3,
                "weight": 1e3,
            },
        },
        "elements": [
            {
                "type": "uniform_field",
                "label": "free-space",
                "space_charge": "uniform_sphere",
            }
        ],
    }


def final_rms(result) -> float:
    cloud = result.pipeline_result.final_cloud
    return float(np.sqrt((cloud.position_m**2).sum(axis=1).mean()))
