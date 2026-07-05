"""Pure-function transport kernels (NumPy float64 reference backend).

Kernels work in dimensionless momentum ``u = p / (m c)``: float32 with SI
momenta underflows (a 3 MeV positron has p² ≈ 3.5e-42 (kg·m/s)², below the
float32 smallest normal), so SI exists only at the State boundary. The
functions are pure — full-array computation, no in-place writes, no
data-dependent control flow — so the Phase 3 JAX backend can reuse the
algebra by swapping `np` for `jnp`.
"""

from __future__ import annotations

import numpy as np

from latent_dirac.core.constants import SPEED_OF_LIGHT_M_PER_S


def boris_step(
    position_m: np.ndarray,
    u: np.ndarray,
    time_s: np.ndarray,
    alive: np.ndarray,
    dt_s: float,
    charge_c: float,
    mass_kg: float,
    e_field: np.ndarray,
    b_field: np.ndarray,
    xp=np,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance one relativistic Boris step; dead particles are frozen.

    `u` is the dimensionless momentum p/(m c); `e_field`/`b_field` are the
    SI fields evaluated at `position_m`. Returns new
    `(position_m, u, time_s)` without mutating any input.

    `xp` selects the array namespace: `numpy` (reference backend, default)
    or `jax.numpy` — the same algebra runs on both, so the physics is
    written exactly once.
    """

    c = SPEED_OF_LIGHT_M_PER_S
    half_electric_kick = (charge_c * dt_s / (2.0 * mass_kg * c)) * e_field

    u_minus = u + half_electric_kick
    gamma_minus = xp.sqrt(1.0 + xp.sum(u_minus * u_minus, axis=1))
    t_vec = (charge_c * dt_s / (2.0 * mass_kg)) * b_field / gamma_minus[:, np.newaxis]
    t_mag2 = xp.sum(t_vec * t_vec, axis=1)
    s_vec = 2.0 * t_vec / (1.0 + t_mag2)[:, np.newaxis]

    u_prime = u_minus + xp.cross(u_minus, t_vec)
    u_plus = u_minus + xp.cross(u_prime, s_vec)
    u_new = u_plus + half_electric_kick

    gamma_new = xp.sqrt(1.0 + xp.sum(u_new * u_new, axis=1))
    velocity = u_new * (c / gamma_new)[:, np.newaxis]
    position_new = position_m + velocity * dt_s
    time_new = time_s + dt_s

    keep = alive[:, np.newaxis]
    return (
        xp.where(keep, position_new, position_m),
        xp.where(keep, u_new, u),
        xp.where(alive, time_new, time_s),
    )


def momentum_to_dimensionless(momentum_kg_m_s: np.ndarray, mass_kg: float) -> np.ndarray:
    """SI momentum -> u = p/(m c) at the State boundary."""

    return momentum_kg_m_s / (mass_kg * SPEED_OF_LIGHT_M_PER_S)


def dimensionless_to_momentum(u: np.ndarray, mass_kg: float) -> np.ndarray:
    """u = p/(m c) -> SI momentum at the State boundary."""

    return u * (mass_kg * SPEED_OF_LIGHT_M_PER_S)
