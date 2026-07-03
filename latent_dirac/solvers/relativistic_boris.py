"""Relativistic Boris pusher for charged macro-particles."""

from __future__ import annotations

import numpy as np
from pydantic import field_validator

from latent_dirac.core.units import gamma_from_momentum
from latent_dirac.fields.base import Field
from latent_dirac.solvers.base import Solver
from latent_dirac.state.particle_cloud import ParticleCloud


class RelativisticBorisSolver(Solver):
    """Advance momentum and position using the relativistic Boris method."""

    dt_s: float
    steps: int

    @field_validator("dt_s")
    @classmethod
    def _positive_dt(cls, value):
        if value <= 0.0:
            raise ValueError("dt_s must be positive")
        return value

    @field_validator("steps")
    @classmethod
    def _positive_steps(cls, value):
        if value <= 0:
            raise ValueError("steps must be positive")
        return value

    def propagate(self, cloud: ParticleCloud, field: Field) -> ParticleCloud:
        result = cloud.copy()
        q = result.species.charge_c
        m = result.species.mass_kg
        dt = self.dt_s

        for _ in range(self.steps):
            live = result.alive
            if not np.any(live):
                break

            x = result.position_m[live]
            p = result.momentum_kg_m_s[live]
            t_now = result.time_s[live]

            electric = field.E(x, t_now)
            magnetic = field.B(x, t_now)

            p_minus = p + q * electric * (0.5 * dt)
            gamma_minus = gamma_from_momentum(p_minus, m)
            t_vec = q * magnetic * (0.5 * dt) / (m * gamma_minus[:, np.newaxis])
            t_mag2 = np.sum(t_vec * t_vec, axis=1)
            s_vec = 2.0 * t_vec / (1.0 + t_mag2)[:, np.newaxis]

            p_prime = p_minus + np.cross(p_minus, t_vec)
            p_plus = p_minus + np.cross(p_prime, s_vec)
            p_new = p_plus + q * electric * (0.5 * dt)

            gamma_new = gamma_from_momentum(p_new, m)
            velocity = p_new / (gamma_new[:, np.newaxis] * m)

            result.momentum_kg_m_s[live] = p_new
            result.position_m[live] = x + velocity * dt
            result.time_s[live] = t_now + dt

        return result
