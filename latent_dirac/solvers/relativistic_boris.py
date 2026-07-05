"""Relativistic Boris pusher for charged macro-particles."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from pydantic import field_validator

from latent_dirac.fields.base import Field
from latent_dirac.solvers.base import Solver
from latent_dirac.solvers.kernels import (
    boris_step,
    dimensionless_to_momentum,
    momentum_to_dimensionless,
)
from latent_dirac.state.particle_state import ParticleState


class RelativisticBorisSolver(Solver):
    """Advance momentum and position using the relativistic Boris method.

    The solver converts SI momentum to dimensionless u = p/(m c) once at
    the State boundary, runs exactly `steps` pure-kernel iterations (dead
    particles are frozen inside the kernel, never skipped by control
    flow), and converts back to SI afterwards.
    """

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

    def propagate(self, state: ParticleState, field: Field) -> ParticleState:
        mass_kg = state.species.mass_kg
        charge_c = state.species.charge_c

        position = state.position_m
        u = momentum_to_dimensionless(state.momentum_kg_m_s, mass_kg)
        time_s = state.time_s
        alive = state.alive

        for _ in range(self.steps):
            e_field = field.E(position, time_s)
            b_field = field.B(position, time_s)
            position, u, time_s = boris_step(
                position,
                u,
                time_s,
                alive,
                dt_s=self.dt_s,
                charge_c=charge_c,
                mass_kg=mass_kg,
                e_field=e_field,
                b_field=b_field,
            )

        return replace(
            state,
            position_m=position,
            momentum_kg_m_s=dimensionless_to_momentum(u, mass_kg),
            time_s=time_s,
            alive=alive.copy(),
            weight=state.weight.copy(),
            particle_id=state.particle_id.copy(),
            parent_id=state.parent_id.copy(),
            lost_at_element=state.lost_at_element.copy(),
            metadata=deepcopy(state.metadata),
        )
