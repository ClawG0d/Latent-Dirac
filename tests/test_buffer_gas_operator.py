"""Tests for the null-collision (Skullerud) buffer-gas operator (T5, slice 2).

The operator applies collisions to a cloud over a hold time using a
curated cross-section table: elastic scatters (redirect, negligible ΔE),
inelastic channels remove their threshold energy (floored at the thermal
(3/2)kT), and loss channels (positronium formation, annihilation,
ionization) kill the particle. Candidate collisions arrive as a Poisson
process at rate ν_max, so the realized rate reproduces n·σ·v exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from latent_dirac.collisions.cross_sections import CrossSectionTable
from latent_dirac.collisions.operator import (
    _speed_from_energy_ev,
    buffer_gas_collide,
    inelastic_energy_after,
)
from latent_dirac.core.constants import ELEMENTARY_CHARGE_C, k_B
from latent_dirac.core.species import positron
from latent_dirac.core.units import kinetic_energy_to_momentum_magnitude
from latent_dirac.state.particle_state import ParticleState

N_GAS = 2.4e19  # ~0.1 Pa N2 at 300 K


def _cloud(energy_ev, n=200, species=positron):
    ke_j = energy_ev * ELEMENTARY_CHARGE_C
    p_mag = float(kinetic_energy_to_momentum_magnitude(ke_j, species.mass_kg))
    momentum = np.tile([0.0, 0.0, p_mag], (n, 1))
    return ParticleState(
        species=species,
        position_m=np.zeros((n, 3)),
        momentum_kg_m_s=momentum,
        time_s=np.zeros(n),
        weight=np.ones(n),
        alive=np.ones(n, dtype=bool),
        particle_id=np.arange(n),
        parent_id=np.full(n, -1),
    )


def _table(channels, thresholds, energies_ev=(0.1, 5.0, 20.0, 100.0)):
    energies = np.asarray(energies_ev, dtype=float)
    chan = {name: np.full(energies.shape, sig) for name, sig in channels.items()}
    return CrossSectionTable(
        energies_ev=energies,
        channels=chan,
        thresholds_ev=dict(thresholds),
        provenance={"gas": "N2", "fidelity_tier": "parameterized"},
        fidelity_tier="parameterized",
    )


def test_inelastic_energy_after_removes_the_threshold():
    # a single inelastic collision removes exactly the channel threshold...
    assert inelastic_energy_after(20.0, 8.5, floor_ev=0.04) == pytest.approx(11.5)
    # ...unless that would drop below the thermal floor, where it clamps
    assert inelastic_energy_after(8.6, 8.5, floor_ev=0.04) == pytest.approx(0.1)
    assert inelastic_energy_after(8.0, 8.5, floor_ev=0.04) == pytest.approx(0.04)


def test_hot_cloud_cools_over_the_hold():
    cloud = _cloud(50.0, n=300)
    before = cloud.mean_kinetic_energy_joule()
    table = _table({"elastic": 5.0e-20, "electronic": 8.0e-21}, {"elastic": 0.0, "electronic": 8.5})
    out = buffer_gas_collide(
        cloud, table, n_gas_m3=N_GAS, hold_time_s=1.0e-5,
        gas_temperature_k=300.0, rng=np.random.default_rng(0),
    )
    after = out.mean_kinetic_energy_joule()
    assert after < before  # cooled
    assert after >= 0.0
    assert np.all(out.kinetic_energy_joule()[out.alive] >= 0.0)


def test_cools_to_the_thermal_floor_with_plenty_of_collisions():
    # with many inelastic collisions the cloud parks at (3/2)kT — and this
    # is now independent of any numerical substep knob (C1 regression)
    cloud = _cloud(50.0, n=300)
    floor_j = 1.5 * k_B * 300.0
    table = _table({"electronic": 5.0e-20}, {"electronic": 8.5})
    out = buffer_gas_collide(
        cloud, table, n_gas_m3=N_GAS, hold_time_s=1.0e-4,
        gas_temperature_k=300.0, rng=np.random.default_rng(0),
    )
    ke = out.kinetic_energy_joule()[out.alive]
    np.testing.assert_allclose(ke.mean(), floor_j, rtol=0.05)


def test_realized_collision_rate_reproduces_n_sigma_v():
    # positronium-only: a particle dies on its FIRST real collision, so the
    # killed fraction = 1 - exp(-nu_real*hold), nu_real = n*sigma*v(E0). This
    # is the null-collision method's defining criterion (C1 regression).
    e0, n = 20.0, 4000
    sigma_ps, hold = 5.0e-20, 2.0e-7
    table = _table({"positronium": sigma_ps}, {"positronium": 0.0})
    out = buffer_gas_collide(
        _cloud(e0, n=n), table, n_gas_m3=N_GAS, hold_time_s=hold,
        gas_temperature_k=300.0, rng=np.random.default_rng(0),
    )
    v0 = float(_speed_from_energy_ev(e0, positron.mass_kg))
    expected_killed = 1.0 - np.exp(-N_GAS * sigma_ps * v0 * hold)
    killed_frac = 1.0 - out.alive.mean()
    assert abs(killed_frac - expected_killed) < 0.03


def test_elastic_only_conserves_energy_but_redirects():
    cloud = _cloud(30.0, n=300)
    before = cloud.mean_kinetic_energy_joule()
    table = _table({"elastic": 5.0e-20}, {"elastic": 0.0})
    out = buffer_gas_collide(
        cloud, table, n_gas_m3=N_GAS, hold_time_s=1.0e-5,
        gas_temperature_k=300.0, rng=np.random.default_rng(1),
    )
    after = out.mean_kinetic_energy_joule()
    assert after == pytest.approx(before, rel=1e-9)  # elastic: energy conserved
    # direction thermalised: transverse momentum appeared (was pure +z)
    pt = np.linalg.norm(out.momentum_kg_m_s[:, :2], axis=1)
    assert np.any(pt > 0.0)


def test_loss_channel_kills_particles():
    cloud = _cloud(20.0, n=300)
    table = _table({"positronium": 5.0e-20}, {"positronium": 8.8})
    out = buffer_gas_collide(
        cloud, table, n_gas_m3=N_GAS, hold_time_s=1.0e-5,
        gas_temperature_k=300.0, rng=np.random.default_rng(2),
    )
    assert out.alive.sum() < 300  # positronium formation removed some
    assert out.weighted_count() < 300


def test_killed_particles_do_not_age():
    # a killed particle stays at its entry time (matches residual_gas_loss and
    # the parameterized cooler); survivors age by the full hold (I4 regression)
    cloud = _cloud(20.0, n=400)
    table = _table({"positronium": 5.0e-20}, {"positronium": 8.8})
    out = buffer_gas_collide(
        cloud, table, n_gas_m3=N_GAS, hold_time_s=1.0e-6,
        gas_temperature_k=300.0, rng=np.random.default_rng(1),
    )
    killed = ~out.alive
    assert killed.any() and out.alive.any()
    np.testing.assert_allclose(out.time_s[killed], 0.0)
    np.testing.assert_allclose(out.time_s[out.alive], 1.0e-6)


def test_deterministic_under_a_fixed_seed():
    table = _table({"elastic": 5.0e-20, "electronic": 8.0e-21}, {"elastic": 0.0, "electronic": 8.5})
    kw = dict(n_gas_m3=N_GAS, hold_time_s=1.0e-5, gas_temperature_k=300.0)
    a = buffer_gas_collide(_cloud(50.0), table, rng=np.random.default_rng(7), **kw)
    b = buffer_gas_collide(_cloud(50.0), table, rng=np.random.default_rng(7), **kw)
    assert np.array_equal(a.alive, b.alive)
    assert np.allclose(a.momentum_kg_m_s, b.momentum_kg_m_s)


def test_no_gas_no_collisions():
    # zero density -> nothing happens (a guard against a divide-by-zero nu_max)
    cloud = _cloud(50.0, n=100)
    table = _table({"elastic": 5.0e-20, "electronic": 8.0e-21}, {"elastic": 0.0, "electronic": 8.5})
    out = buffer_gas_collide(
        cloud, table, n_gas_m3=0.0, hold_time_s=1.0e-5,
        gas_temperature_k=300.0, rng=np.random.default_rng(0),
    )
    assert out.alive.all()
    assert np.allclose(out.momentum_kg_m_s, cloud.momentum_kg_m_s)


def test_thermal_floor_from_gas_temperature():
    # a cloud in a warm gas is not driven below (3/2) k_B T
    cloud = _cloud(9.0, n=300)
    floor_j = 1.5 * k_B * 300.0
    table = _table({"electronic": 5.0e-20}, {"electronic": 8.5})
    out = buffer_gas_collide(
        cloud, table, n_gas_m3=N_GAS, hold_time_s=1.0e-4,
        gas_temperature_k=300.0, rng=np.random.default_rng(3),
    )
    ke = out.kinetic_energy_joule()[out.alive]
    assert np.all(ke >= floor_j - 1e-30)
