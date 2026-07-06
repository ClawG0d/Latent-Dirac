"""Thin-sheet solenoid profile: field model, wiring, and Busch physics.

Design record: docs/superpowers/specs/2026-07-06-thin-sheet-solenoid-profile-design.md.
The first-order pair (B_z = b(z), B_r = -(r/2) b'(z)) is exactly
divergence-free (curl of A_phi = r b(z) / 2), so the canonical angular
momentum P_phi = (x p_y - y p_x) + q r^2 b(z) / 2 is an exact invariant
of the true dynamics — the strongest single check of both the field
implementation and its consistency under the Boris integrator.
"""

from __future__ import annotations

import numpy as np
import pytest

from latent_dirac.core.constants import ELECTRON_MASS_KG
from latent_dirac.core.species import positron
from latent_dirac.core.units import momentum_gev_c_to_si
from latent_dirac.fields.solenoid import SolenoidField, ThinSheetSolenoidField
from latent_dirac.scene.loader import scene_from_mapping
from latent_dirac.solvers.relativistic_boris import RelativisticBorisSolver
from latent_dirac.state.particle_state import ParticleState

B0_TESLA = 0.5
RADIUS_M = 0.03
LENGTH_M = 0.2
CENTER_Z_M = 0.3


def make_field(**overrides) -> ThinSheetSolenoidField:
    params = {
        "b_tesla": B0_TESLA,
        "radius_m": RADIUS_M,
        "length_m": LENGTH_M,
        "center_z_m": CENTER_Z_M,
    }
    params.update(overrides)
    return ThinSheetSolenoidField(**params)


def on_axis_profile(field: ThinSheetSolenoidField, z: np.ndarray) -> np.ndarray:
    positions = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    return field.B(positions, 0.0)[:, 2]


def test_long_solenoid_limit_recovers_sheet_strength():
    field = make_field(radius_m=0.01, length_m=1.0, center_z_m=0.0)
    b_center = field.B(np.array([0.0, 0.0, 0.0]), 0.0)[2]
    assert b_center == pytest.approx(B0_TESLA, rel=5e-4)


def test_integrated_on_axis_strength_matches_hard_edge():
    field = make_field()
    z = np.linspace(CENTER_Z_M - 40.0 * LENGTH_M, CENTER_Z_M + 40.0 * LENGTH_M, 200_001)
    # manual trapezoid: np.trapezoid needs numpy >= 2.0 while the
    # declared package floor is numpy >= 1.26
    b = on_axis_profile(field, z)
    integral = float(np.sum(0.5 * (b[1:] + b[:-1]) * np.diff(z)))
    assert integral == pytest.approx(B0_TESLA * LENGTH_M, rel=1e-4)


def test_divergence_free_everywhere():
    field = make_field()
    rng = np.random.default_rng(11)
    # in-bore, off-bore, and points pinned to the former hard edges
    points = np.vstack(
        [
            rng.uniform([-0.05, -0.05, 0.0], [0.05, 0.05, 0.6], size=(20, 3)),
            [[RADIUS_M, 0.0, CENTER_Z_M], [0.0, RADIUS_M, CENTER_Z_M - 0.5 * LENGTH_M]],
            [[0.02, 0.01, CENTER_Z_M + 0.5 * LENGTH_M]],
        ]
    )
    h = 1.0e-6
    for point in points:
        divergence = 0.0
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = h
            divergence += (field.B(point + offset, 0.0)[axis] - field.B(point - offset, 0.0)[axis]) / (
                2.0 * h
            )
        assert abs(divergence) < 1.0e-6


def test_fringe_funnels_into_the_bore():
    field = make_field()
    entrance_z = CENTER_Z_M - 0.5 * LENGTH_M
    below_entrance = field.B(np.array([0.01, 0.0, entrance_z - 0.05]), 0.0)
    above_exit = field.B(np.array([0.01, 0.0, CENTER_Z_M + 0.5 * LENGTH_M + 0.05]), 0.0)
    assert below_entrance[0] < 0.0  # converging toward the axis on the way in
    assert above_exit[0] > 0.0  # splaying back out past the exit
    assert below_entrance[2] > 0.0  # axial component already present in the fringe


def test_smooth_across_bore_radius_and_faces():
    field = make_field()
    radial_line = np.column_stack(
        [np.linspace(0.0, 2.0 * RADIUS_M, 401), np.zeros(401), np.full(401, CENTER_Z_M)]
    )
    axial_line = np.column_stack([np.full(401, 0.5 * RADIUS_M), np.zeros(401), np.linspace(0.0, 0.6, 401)])
    for line in (radial_line, axial_line):
        values = field.B(line, 0.0)
        jumps = np.max(np.abs(np.diff(values, axis=0)), axis=0)
        # a hard edge would jump by ~B0 between adjacent samples
        assert np.all(jumps < 0.05 * B0_TESLA)


def _single_positron(x_m: float, p_gev_c: float) -> ParticleState:
    return ParticleState(
        species=positron,
        position_m=np.array([[x_m, 0.0, 0.0]]),
        momentum_kg_m_s=np.array([[0.0, 0.0, momentum_gev_c_to_si(p_gev_c)]]),
        time_s=np.zeros(1),
        weight=np.ones(1),
        alive=np.ones(1, dtype=bool),
        particle_id=np.arange(1),
        parent_id=np.full(1, -1),
    )


def test_busch_rotation_and_canonical_angular_momentum():
    field = make_field(radius_m=0.05, length_m=0.4, center_z_m=0.3)
    r0 = 0.01
    cloud = _single_positron(r0, 0.002)  # 2 MeV/c, parallel to the axis
    cloud.position_m[0, 2] = -0.2  # weak-field region, 0.3 m before the face

    q = positron.charge_c
    # dt resolves the gyration ~600x so the integrator contribution is
    # negligible next to the evaluation skew documented at the assert
    stepper = RelativisticBorisSolver(dt_s=5.0e-13, steps=1)
    gamma = float(cloud.gamma()[0])

    def b_on_axis(z: float) -> float:
        return float(field.B(np.array([0.0, 0.0, z]), 0.0)[2])

    def canonical_p_phi(state: ParticleState) -> float:
        x, y, z = state.position_m[0]
        px, py = state.momentum_kg_m_s[0, :2]
        return float(x * py - y * px + 0.5 * q * (x * x + y * y) * b_on_axis(z))

    p_phi_start = canonical_p_phi(cloud)
    p_phi_scale = 0.5 * abs(q) * r0 * r0 * B0_TESLA
    # tolerance floor: Boris staggers momentum half a step behind position,
    # so evaluating P_phi from one state carries an O(dt * torque) skew
    # (~0.5% here) that dwarfs the true conservation drift; a sign or
    # factor error in the fringe would deviate at the 50%-of-scale level
    rate_errors = []
    max_p_perp = 0.0
    current = cloud
    for _ in range(5000):
        current = stepper.propagate(current, field)
        assert canonical_p_phi(current) == pytest.approx(p_phi_start, abs=1.0e-2 * p_phi_scale)
        x, y, z = current.position_m[0]
        r = np.hypot(x, y)
        max_p_perp = max(max_p_perp, float(np.hypot(*current.momentum_kg_m_s[0, :2])))
        # Busch (P_phi ~ 0): phi_dot = -q b / (2 gamma m) at ANY radius;
        # the residual P_phi / (gamma m r^2) correction from the nonzero
        # tail field at launch stays under (r0/r)^2 * 0.6% — keep r above
        # 0.5 r0 so it sits within the 5% tolerance
        if 0.25 <= z <= 0.45 and r > 0.5 * r0:
            vx, vy = current.velocity()[0, :2]
            phi_dot = (x * vy - y * vx) / (r * r)
            larmor = -q * b_on_axis(z) / (2.0 * gamma * ELECTRON_MASS_KG)
            rate_errors.append(phi_dot / larmor - 1.0)
        if z > 0.45:
            break
    assert len(rate_errors) > 20, "trajectory never sampled the interior at r > 0.5 r0"
    assert np.max(np.abs(rate_errors)) < 5.0e-2

    # transverse momentum was actually acquired through the fringe
    # (hard edge acquires exactly zero — see the companion test)
    assert max_p_perp > 0.25 * 0.5 * abs(q) * r0 * B0_TESLA


def test_hard_edge_artifact_no_transverse_kick():
    field = SolenoidField(b_tesla=B0_TESLA, radius_m=0.05, length_m=0.4, center_z_m=0.3)
    cloud = _single_positron(0.01, 0.002)
    cloud.position_m[0, 2] = -0.2
    result = RelativisticBorisSolver(dt_s=2.0e-12, steps=1200).propagate(cloud, field)
    # v x B = 0 for axis-parallel momentum in a purely axial field: the
    # documented hard-edge artifact this profile exists to remove
    assert np.all(result.momentum_kg_m_s[0, :2] == cloud.momentum_kg_m_s[0, :2])


SOURCE_PARAMS = {
    "primary_count": 10000,
    "yield_eplus_per_primary": 0.02,
    "mean_energy_MeV": 3.0,
    "energy_spread_MeV": 0.4,
    "angular_rms_rad": 0.03,
    "source_sigma_m": 0.001,
    "bunch_length_s": 1.0e-12,
    "macro_particles": 48,
}


def scene_mapping(profile: str | None) -> dict:
    solenoid: dict = {
        "type": "solenoid",
        "label": "sol",
        "b_tesla": 0.5,
        "radius_m": 0.03,
        "length_m": 0.2,
        "center_z_m": 0.1,
        "steps": 60,
    }
    if profile is not None:
        solenoid["profile"] = profile
    return {
        "schema_version": 1,
        "name": "thin-sheet-line",
        "seed": 2032,
        "source": {"type": "positron_pair", "label": "src", "params": dict(SOURCE_PARAMS)},
        "solver": {"type": "relativistic_boris", "dt_s": 2.0e-12, "steps": 60},
        "elements": [
            solenoid,
            # tight enough that the cut bites: the soft objective must sit
            # strictly inside (0, 1) for the gradient test
            {"type": "aperture", "label": "window", "radius_m": 0.002, "z_m": 0.03},
        ],
    }


def test_schema_profile_default_and_validation():
    default_scene = scene_from_mapping(scene_mapping(None))
    assert default_scene.elements[0].profile == "hard_edge"
    thin_scene = scene_from_mapping(scene_mapping("thin_sheet"))
    assert thin_scene.elements[0].profile == "thin_sheet"
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        scene_from_mapping(scene_mapping("soft_edge"))


def test_profiles_produce_different_transport():
    from latent_dirac.scene.build import run_scene

    hard = run_scene(scene_from_mapping(scene_mapping("hard_edge")))
    thin = run_scene(scene_from_mapping(scene_mapping("thin_sheet")))
    hard_final = hard.pipeline_result.final_cloud
    thin_final = thin.pipeline_result.final_cloud
    # SI momenta are ~1e-21 kg m/s: allclose's default atol=1e-8 would
    # declare everything equal, so compare with rtol only
    assert not np.allclose(hard_final.momentum_kg_m_s, thin_final.momentum_kg_m_s, rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("profile", ["thin_sheet", "hard_edge"])
def test_jax_parity_and_sweep(profile):
    # both profiles: the demo scenes all switched to thin_sheet, so this
    # is the suite's only element-wise hard-edge JAX-vs-NumPy comparison
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from latent_dirac.backends.jax_scene import run_scene_batched
    from latent_dirac.scene.build import run_scene

    scene = scene_from_mapping(scene_mapping(profile))
    reference = run_scene(scene).pipeline_result.final_cloud
    batched = run_scene_batched(scene, overrides={})
    np.testing.assert_array_equal(batched.alive[0], reference.alive)
    np.testing.assert_allclose(batched.position_m[0], reference.position_m, rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(batched.momentum_kg_m_s[0], reference.momentum_kg_m_s, rtol=1e-9, atol=1e-30)

    swept = run_scene_batched(scene, overrides={"sol.b_tesla": np.array([0.3, 0.5, 0.9])})
    assert swept.alive.shape[0] == 3


def test_differentiable_gradient_thin_sheet():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from latent_dirac.backends.differentiable import make_differentiable_objective

    scene = scene_from_mapping(scene_mapping("thin_sheet"))
    objective = make_differentiable_objective(scene, variables=["sol.b_tesla"], sharpness=50.0)
    inputs = {"sol.b_tesla": 0.5}
    value, grads = objective.value_and_grad(inputs)
    assert 0.0 < value < 1.0
    assert np.isfinite(grads["sol.b_tesla"])

    step = 1.0e-5
    numeric = (
        objective.value({"sol.b_tesla": 0.5 + step}) - objective.value({"sol.b_tesla": 0.5 - step})
    ) / (2.0 * step)
    assert grads["sol.b_tesla"] == pytest.approx(numeric, rel=1e-3, abs=1e-8)


def test_render_hover_carries_thin_sheet_fidelity():
    pytest.importorskip("plotly")
    from latent_dirac.scene.build import run_scene
    from latent_dirac.viz.scene_3d import render_scene_3d

    scene = scene_from_mapping(scene_mapping("thin_sheet"))
    figure = render_scene_3d(scene, run_scene(scene, record_trajectories=True))
    hovers = [trace.hovertext for trace in figure.data if getattr(trace, "hovertext", None)]
    assert any("thin-sheet" in hover for hover in hovers)
