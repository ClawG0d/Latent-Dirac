"""Unit conversion and relativistic kinematics helpers.

All functions use SI units internally unless the function name states an
external convention such as eV or GeV/c.
"""

from __future__ import annotations

import numpy as np

from latent_dirac.core.constants import c, e


def ev_to_joule(energy_ev):
    return np.asarray(energy_ev) * e


def joule_to_ev(energy_joule):
    return np.asarray(energy_joule) / e


def mev_to_joule(energy_mev):
    return ev_to_joule(np.asarray(energy_mev) * 1.0e6)


def gev_to_joule(energy_gev):
    return ev_to_joule(np.asarray(energy_gev) * 1.0e9)


def momentum_gev_c_to_si(momentum_gev_c):
    return gev_to_joule(momentum_gev_c) / c


def momentum_si_to_gev_c(momentum_kg_m_s):
    return joule_to_ev(np.asarray(momentum_kg_m_s) * c) / 1.0e9


def _momentum_magnitude(momentum_kg_m_s):
    momentum = np.asarray(momentum_kg_m_s, dtype=float)
    if momentum.ndim > 0 and momentum.shape[-1] == 3:
        return np.linalg.norm(momentum, axis=-1)
    return np.abs(momentum)


def gamma_from_momentum(momentum_kg_m_s, mass_kg: float):
    p_mag = _momentum_magnitude(momentum_kg_m_s)
    return np.sqrt(1.0 + (p_mag / (mass_kg * c)) ** 2)


def velocity_from_momentum(momentum_kg_m_s, mass_kg: float):
    momentum = np.asarray(momentum_kg_m_s, dtype=float)
    gamma = gamma_from_momentum(momentum, mass_kg)
    if momentum.ndim > 0 and momentum.shape[-1] == 3:
        return momentum / (gamma[..., np.newaxis] * mass_kg)
    return momentum / (gamma * mass_kg)


def kinetic_energy_from_momentum(momentum_kg_m_s, mass_kg: float):
    gamma = gamma_from_momentum(momentum_kg_m_s, mass_kg)
    return (gamma - 1.0) * mass_kg * c**2


def kinetic_energy_to_momentum_magnitude(kinetic_energy_joule, mass_kg: float):
    energy = np.asarray(kinetic_energy_joule, dtype=float)
    if np.any(energy < 0.0):
        raise ValueError("kinetic_energy_joule must be non-negative")
    gamma = energy / (mass_kg * c**2) + 1.0
    return mass_kg * c * np.sqrt(np.maximum(gamma**2 - 1.0, 0.0))
