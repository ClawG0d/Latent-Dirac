"""Tests for the Xsuite adapter (Spec: 2026-07-05 Xsuite adapter).

Requires the optional [xsuite] extra; skipped when xtrack is missing.
"""

from __future__ import annotations

import numpy as np
import pytest

xt = pytest.importorskip("xtrack")
pytest.importorskip("xsuite")  # line.track() loads its prebuilt kernels from here

from latent_dirac.adapters.xsuite.adapter import (  # noqa: E402
    ReferenceFrame,
    from_xtrack_particles,
    reference_from_state,
    to_xtrack_particles,
    xsuite_tracking_stage,
)
from latent_dirac.core.constants import c, e  # noqa: E402
from latent_dirac.core.species import antiproton, positron  # noqa: E402
from latent_dirac.sources.base import particle_arrays  # noqa: E402
from latent_dirac.state.particle_state import ParticleState  # noqa: E402


def make_forward_state(count: int = 8, species=positron, p0c_ev: float = 25e6) -> ParticleState:
    """A paraxial forward-going cloud around the reference momentum."""
    rng = np.random.default_rng(41)
    p0_si = p0c_ev * e / c
    momentum = np.zeros((count, 3))
    momentum[:, 0] = p0_si * rng.uniform(-1e-4, 1e-4, count)
    momentum[:, 1] = p0_si * rng.uniform(-1e-4, 1e-4, count)
    momentum[:, 2] = p0_si * rng.uniform(0.95, 1.05, count)
    position = np.zeros((count, 3))
    position[:, 0] = rng.uniform(-1e-3, 1e-3, count)
    position[:, 1] = rng.uniform(-1e-3, 1e-3, count)
    position[:, 2] = rng.uniform(-0.01, 0.01, count)
    state = ParticleState(
        species=species,
        position_m=position,
        momentum_kg_m_s=momentum,
        time_s=rng.uniform(0.0, 1e-9, count),
        weight=np.full(count, 3.0),
        **particle_arrays(count),
    )
    return state


def test_round_trip_closure():
    state = make_forward_state()
    state.alive[2] = False
    frame = ReferenceFrame(p0c_ev=25e6)

    particles = to_xtrack_particles(state, frame)
    back = from_xtrack_particles(particles, state.species, frame, template_state=state)

    np.testing.assert_allclose(back.position_m, state.position_m, rtol=0, atol=1e-15)
    np.testing.assert_allclose(back.momentum_kg_m_s, state.momentum_kg_m_s, rtol=1e-12)
    np.testing.assert_allclose(back.time_s, state.time_s, rtol=0, atol=1e-21)
    np.testing.assert_array_equal(back.weight, state.weight)
    np.testing.assert_array_equal(back.particle_id, state.particle_id)
    np.testing.assert_array_equal(back.alive, state.alive)


def test_species_mapping_signs_and_masses():
    frame = ReferenceFrame(p0c_ev=3.5e9)
    pbar_state = make_forward_state(species=antiproton, p0c_ev=3.5e9)
    pos_state = make_forward_state(species=positron, p0c_ev=3.5e9)

    pbar = to_xtrack_particles(pbar_state, frame)
    pos = to_xtrack_particles(pos_state, frame)

    assert pbar.q0 == -1
    assert pos.q0 == 1
    assert pbar.mass0 == pytest.approx(antiproton.mass_kg * c**2 / e, rel=1e-12)
    assert pos.mass0 == pytest.approx(positron.mass_kg * c**2 / e, rel=1e-12)


def test_reference_from_state_uses_weighted_mean():
    state = make_forward_state()
    frame = reference_from_state(state)
    p_mean = np.average(
        np.linalg.norm(state.momentum_kg_m_s, axis=1), weights=state.weight
    )
    assert frame.p0c_ev == pytest.approx(p_mean * c / e)


def test_off_reference_frame_still_round_trips():
    state = make_forward_state(p0c_ev=25e6)
    frame = ReferenceFrame(p0c_ev=30e6)  # deliberately off the cloud mean

    particles = to_xtrack_particles(state, frame)
    back = from_xtrack_particles(particles, state.species, frame, template_state=state)

    np.testing.assert_allclose(back.momentum_kg_m_s, state.momentum_kg_m_s, rtol=1e-12)
    np.testing.assert_allclose(back.time_s, state.time_s, rtol=0, atol=1e-21)


def test_drift_equivalence_paraxial():
    length = 1.0
    state = make_forward_state()
    frame = reference_from_state(state)
    line = xt.Line(elements=[xt.Drift(length=length)])
    line.build_tracker()

    particles = to_xtrack_particles(state, frame)
    line.track(particles, num_turns=1)
    tracked = from_xtrack_particles(
        particles, state.species, frame, template_state=state, line_length_m=length
    )

    # analytic straight line: x += L * px/pz (our Boris result in zero field)
    pz = state.momentum_kg_m_s[:, 2]
    expected_x = state.position_m[:, 0] + length * state.momentum_kg_m_s[:, 0] / pz
    expected_y = state.position_m[:, 1] + length * state.momentum_kg_m_s[:, 1] / pz
    # xtrack's expanded drift differs at O(px^2/p^2) ~ 1e-8 relative: the
    # documented paraxial handshake tolerance, not a discrepancy
    np.testing.assert_allclose(tracked.position_m[:, 0], expected_x, atol=1e-9)
    np.testing.assert_allclose(tracked.position_m[:, 1], expected_y, atol=1e-9)
    np.testing.assert_allclose(tracked.momentum_kg_m_s, state.momentum_kg_m_s, rtol=1e-12)


def test_tracking_advances_z_and_time():
    # regression: xtrack resets per-particle s to 0 at turn end, so z and
    # time must be rebuilt from at_turn, the line length, and zeta
    length = 2.0
    state = make_forward_state()
    frame = reference_from_state(state)
    line = xt.Line(elements=[xt.Drift(length=length)])
    line.build_tracker()

    stage = xsuite_tracking_stage("drift", line, frame)
    out, _ = stage.run(state, stage_index=0)

    np.testing.assert_allclose(
        out.position_m[:, 2], state.position_m[:, 2] + length, rtol=0, atol=1e-9
    )
    p = np.linalg.norm(state.momentum_kg_m_s, axis=1)
    energy = np.sqrt((p * c) ** 2 + (state.species.mass_kg * c**2) ** 2)
    beta = p * c / energy  # per-particle beta, not the reference beta0
    np.testing.assert_allclose(
        out.time_s - state.time_s, length / (beta * c), rtol=1e-6
    )


def test_backward_going_alive_particles_are_rejected():
    state = make_forward_state()
    state.momentum_kg_m_s[1, 2] *= -1.0
    frame = ReferenceFrame(p0c_ev=25e6)

    with pytest.raises(ValueError, match="forward"):
        to_xtrack_particles(state, frame)

    # a dead backward particle is tolerated: only alive rows are converted
    # meaningfully, and the dead row stays dead through the round trip
    state.alive[1] = False
    particles = to_xtrack_particles(state, frame)
    back = from_xtrack_particles(particles, state.species, frame, template_state=state)
    assert not back.alive[1]


def test_tracking_stage_stamps_ledger():
    state = make_forward_state(count=12)
    state.position_m[:, 0] = np.linspace(-4e-3, 4e-3, 12)
    frame = reference_from_state(state)
    line = xt.Line(
        elements=[xt.Drift(length=0.1), xt.LimitRect(min_x=-2e-3, max_x=2e-3)]
    )
    line.build_tracker()

    stage = xsuite_tracking_stage("xsuite-line", line, frame)
    out_state, result = stage.run(state, stage_index=7)

    killed = np.abs(state.position_m[:, 0]) > 2e-3
    assert np.any(killed) and not np.all(killed)
    np.testing.assert_array_equal(out_state.alive, ~killed)
    assert np.all(out_state.lost_at_element[killed] == 7)
    assert np.all(out_state.lost_at_element[~killed] == -1)
    assert result.stage_name == "xsuite-line"


def test_dead_on_entry_stay_dead_through_tracking():
    state = make_forward_state(count=4)
    state.alive[1] = False
    state.lost_at_element[1] = 3
    frame = reference_from_state(state)
    line = xt.Line(elements=[xt.Drift(length=0.5)])
    line.build_tracker()

    stage = xsuite_tracking_stage("drift", line, frame)
    out_state, _ = stage.run(state, stage_index=9)

    assert not out_state.alive[1]
    assert out_state.lost_at_element[1] == 3  # not restamped
    assert np.all(out_state.alive[[0, 2, 3]])
