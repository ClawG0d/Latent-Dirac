"""Null-collision (Skullerud) Monte Carlo operator for buffer-gas cooling.

Given a cloud and a curated cross-section table, apply collisions over a
hold time. The null-collision method uses a constant candidate rate
ν_max = max_E [n_gas · σ_total(E) · v(E)] over the tabulated grid; candidate
collisions arrive as a Poisson process at that rate (inter-arrival times
drawn from Exponential(1/ν_max)), and each candidate is a real collision
with probability ν(E)/ν_max, with the channel picked in proportion to
σ_i(E). Channel kinematics:

- elastic: isotropic redirect, negligible energy change;
- inelastic (any channel with a threshold that is not a loss channel):
  remove the channel threshold energy, floored at the thermal (3/2)kT of
  the gas, then redirect;
- loss (positronium formation, annihilation, ionization): the positron
  is removed (killed); the scene pipeline stamps the loss ledger.

Drawing the time to the next candidate (rather than one Bernoulli trial
per fixed substep) makes the realized collision rate reproduce n·σ·v
exactly — there is no numerical substep knob to tune. Pure and seeded (a
caller-supplied ``numpy.random.Generator``). Stochastic and
energy-changing, so — like the parameterized ``buffer_gas_cooling`` — it
lives on the NumPy pipeline only; the JAX backend rejects it.
"""

from __future__ import annotations

import numpy as np

from latent_dirac.collisions.cross_sections import CrossSectionTable
from latent_dirac.core.constants import ELEMENTARY_CHARGE_C, SPEED_OF_LIGHT_M_PER_S, k_B
from latent_dirac.core.units import kinetic_energy_to_momentum_magnitude
from latent_dirac.state.particle_state import ParticleState

#: Channels that remove the positron rather than cool it.
LOSS_CHANNELS = frozenset({"positronium", "annihilation", "ionization"})
#: The single channel treated as pure elastic redirection.
ELASTIC_CHANNEL = "elastic"


def inelastic_energy_after(energy_ev: float, threshold_ev: float, *, floor_ev: float) -> float:
    """Kinetic energy (eV) after one inelastic collision: remove the threshold, floor at kT."""
    return max(energy_ev - threshold_ev, floor_ev)


def _speed_from_energy_ev(energy_ev, mass_kg: float):
    """Relativistic speed (m/s) at a kinetic energy, elementwise."""
    ke_j = np.asarray(energy_ev, dtype=float) * ELEMENTARY_CHARGE_C
    p_mag = kinetic_energy_to_momentum_magnitude(np.clip(ke_j, 0.0, None), mass_kg)
    gamma = ke_j / (mass_kg * SPEED_OF_LIGHT_M_PER_S**2) + 1.0
    return p_mag / (gamma * mass_kg)


def _sigmas_at(table: CrossSectionTable, channel_names, energy_ev: float):
    """σ (m²) for every channel at an energy — linear interpolation, 0 outside range."""
    return [
        float(np.interp(energy_ev, table.energies_ev, table.channels[c], left=0.0, right=0.0))
        for c in channel_names
    ]


def _isotropic_direction(rng: np.random.Generator) -> np.ndarray:
    """A unit vector drawn uniformly on the sphere."""
    cos_theta = rng.uniform(-1.0, 1.0)
    sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta)))
    phi = rng.uniform(0.0, 2.0 * np.pi)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])


def buffer_gas_collide(
    state: ParticleState,
    table: CrossSectionTable,
    *,
    n_gas_m3: float,
    hold_time_s: float,
    gas_temperature_k: float,
    rng: np.random.Generator,
) -> ParticleState:
    """Apply null-collision buffer-gas collisions to a cloud over a hold time."""
    result = state.copy()
    mass = result.species.mass_kg
    floor_ev = 1.5 * k_B * gas_temperature_k / ELEMENTARY_CHARGE_C

    # nu_max = max over the tabulated grid of n_gas * sigma_total(E) * v(E).
    channel_names = list(table.channels.keys())
    total_on_grid = np.sum([table.channels[c] for c in channel_names], axis=0)
    v_on_grid = _speed_from_energy_ev(table.energies_ev, mass)
    nu_max = float(np.max(n_gas_m3 * total_on_grid * v_on_grid))
    if nu_max <= 0.0 or hold_time_s <= 0.0:
        return result  # no gas / no time: nothing collides

    mean_gap = 1.0 / nu_max
    ke_ev = result.kinetic_energy_joule() / ELEMENTARY_CHARGE_C
    survive = result.alive.copy()

    for i in np.flatnonzero(result.alive):
        energy = float(ke_ev[i])
        direction = result.momentum_kg_m_s[i].copy()
        norm = float(np.linalg.norm(direction))
        if norm > 0.0:
            direction = direction / norm
        collided = False
        killed = False

        # candidate collisions arrive as a Poisson process of rate nu_max
        t = float(rng.exponential(mean_gap))
        while t < hold_time_s:
            sigmas = _sigmas_at(table, channel_names, energy)
            sig_total = sum(sigmas)
            if sig_total > 0.0:
                speed = float(_speed_from_energy_ev(energy, mass))
                nu_real = n_gas_m3 * sig_total * speed
                if rng.random() < nu_real / nu_max:  # real (vs null) collision
                    r = rng.random() * sig_total  # pick a channel in proportion to sigma_i(E)
                    acc = 0.0
                    chosen = channel_names[-1]
                    for name, sig in zip(channel_names, sigmas, strict=True):
                        acc += sig
                        if r <= acc:
                            chosen = name
                            break
                    if chosen in LOSS_CHANNELS:
                        killed = True
                        break
                    if chosen != ELASTIC_CHANNEL:
                        energy = inelastic_energy_after(
                            energy, table.thresholds_ev[chosen], floor_ev=floor_ev
                        )
                    direction = _isotropic_direction(rng)
                    collided = True
            t += float(rng.exponential(mean_gap))

        if killed:
            survive[i] = False
            continue
        if collided:
            new_mag = float(kinetic_energy_to_momentum_magnitude(energy * ELEMENTARY_CHARGE_C, mass))
            result.momentum_kg_m_s[i] = direction * new_mag

    # killed particles do not age (match residual_gas_loss / parameterized cooling)
    aged = result.alive & survive
    result.time_s = result.time_s + aged * hold_time_s
    result.apply_alive_mask(survive)
    return result
