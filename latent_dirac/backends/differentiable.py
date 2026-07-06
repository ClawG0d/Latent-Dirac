"""Differentiable capture objective via soft acceptance relaxation.

The hard accepted fraction is a step function of scene parameters, so its
gradient is zero almost everywhere. For gradient-based design this module
relaxes the acceptance stages into smooth per-particle survival weights
(sigmoids of the cut margins). The relaxation is an optimization device,
not a physics claim: the hard NumPy/JAX pipelines remain the source of
truth, and the soft objective converges to the hard `accepted_fraction`
as `sharpness` grows.

The element loop mirrors `jax_scene._make_simulator` semantics (with one
documented exception, below); in soft mode nothing is frozen — survival
is bookkeeping, and survivors' trajectories are identical to the hard
pipeline's.

One intentional divergence from that mirror: `residual_gas_loss`. Its
hard form is a stochastic per-particle kill (no static-program form, so
the batched simulator rejects it); here it enters as its EXPECTED
survival factor exp(-hold_time_s / mean_lifetime_s) — a smooth,
differentiable multiplier. This is the general principle for stochastic
losses: hard = a random draw, soft = its expectation. It lets a design
loop jointly optimize capture efficiency against storage survival (the
antimatter-native objective: not most captured, but most still alive
after the hold).

Known gradient artifact: the sigmoid widths scale with the optimized
parameters themselves (`radius/sharpness`, `span/sharpness`), so in
regions where the hard objective is exactly flat (a cut far from the
beam) the soft gradient carries a small width-coupling component that can
even point the wrong way. It vanishes as sharpness grows; always validate
optima with the hard pipeline.
"""

from __future__ import annotations

import numpy as np

from latent_dirac.backends.evaluator import parse_variables
from latent_dirac.backends.jax_scene import (
    _FIELD_FNS,
    _TRANSPORT_TYPES,
    _base_params,
    _import_jax,
)
from latent_dirac.core.constants import ELEMENTARY_CHARGE_C, SPEED_OF_LIGHT_M_PER_S
from latent_dirac.scene.build import build_source
from latent_dirac.scene.schema import Scene
from latent_dirac.solvers.kernels import boris_step


class DifferentiableObjective:
    """Soft accepted fraction with gradients w.r.t. named scene variables.

    Variables use the evaluator vocabulary (`"label.param"`,
    `"label.vec[i]"`). The objective value is the survival-weighted
    accepted fraction of the relaxed acceptance chain — an approximation
    for optimization; validate optima with the hard pipeline.
    """

    def __init__(self, scene: Scene, variables: list[str], sharpness: float = 200.0):
        if sharpness <= 0.0:
            raise ValueError("sharpness must be positive")

        for element in scene.elements:
            if getattr(element, "space_charge", None) is not None:
                # checked before parse_variables so this message wins over
                # the generic JAX-backend rejection raised further down
                raise ValueError(
                    f"element {element.label!r} enables space_charge, which the "
                    "differentiable objective does not support; use the NumPy pipeline"
                )

        jax, jnp = _import_jax()
        self._variables, _ = parse_variables(scene, variables)
        self._order = list(self._variables)

        label_to_index = {element.label: index for index, element in enumerate(scene.elements)}
        self._slots = {}
        for variable, (key, component) in self._variables.items():
            label, _, param = key.partition(".")
            self._slots[variable] = (label_to_index[label], param, component)

        base_params = _base_params(scene)
        base_jnp = [{name: jnp.asarray(value) for name, value in entry.items()} for entry in base_params]

        rng = np.random.default_rng(scene.seed)
        initial = build_source(scene).sample(rng)
        mass_kg = initial.species.mass_kg
        charge_c = initial.species.charge_c
        c = SPEED_OF_LIGHT_M_PER_S

        position0 = jnp.asarray(initial.position_m)
        u0 = jnp.asarray(initial.momentum_kg_m_s / (mass_kg * c))
        time0 = jnp.asarray(initial.time_s)
        alive0 = jnp.asarray(initial.alive)
        weight = jnp.asarray(initial.weight)
        total_weight = float(np.sum(initial.weight))

        dt_s = scene.solver.dt_s
        lax = jax.lax
        sigmoid = jax.nn.sigmoid

        def objective(theta):
            params = [dict(entry) for entry in base_jnp]
            for position_in_theta, variable in enumerate(self._order):
                index, param, component = self._slots[variable]
                if component is None:
                    params[index][param] = theta[position_in_theta]
                else:
                    params[index][param] = params[index][param].at[component].set(theta[position_in_theta])

            position, u, time_s = position0, u0, time0
            survival = jnp.asarray(alive0, position0.dtype)
            for index, element in enumerate(scene.elements):
                element_params = params[index]
                if element.type in _TRANSPORT_TYPES:
                    steps = element.steps if element.steps is not None else scene.solver.steps
                    field_fn = _FIELD_FNS[element.type]

                    def step_fn(carry, _, field_fn=field_fn, element_params=element_params):
                        pos, u_now, t_now = carry
                        e_field, b_field = field_fn(jnp, pos, t_now, element_params)
                        return boris_step(
                            pos,
                            u_now,
                            t_now,
                            alive0,
                            dt_s=dt_s,
                            charge_c=charge_c,
                            mass_kg=mass_kg,
                            e_field=e_field,
                            b_field=b_field,
                            xp=jnp,
                        ), None

                    (position, u, time_s), _ = lax.scan(step_fn, (position, u, time_s), None, length=steps)
                elif element.type == "aperture":
                    radius = element_params["radius_m"]
                    radial = jnp.sqrt(position[:, 0] ** 2 + position[:, 1] ** 2)
                    # abs keeps degenerate regimes (radius <= 0) collapsing
                    # toward zero survival, matching the hard pipeline
                    width = jnp.abs(radius) / sharpness
                    survival = survival * sigmoid((radius - radial) / width)
                elif element.type == "momentum_window":
                    to_u = 1.0e9 * ELEMENTARY_CHARGE_C / (c * mass_kg * c)
                    u_min = element_params["p_min_gev_c"] * to_u
                    u_max = element_params["p_max_gev_c"] * to_u
                    u_mag = jnp.sqrt(jnp.sum(u * u, axis=1))
                    width = jnp.abs(u_max - u_min) / sharpness
                    survival = survival * sigmoid((u_mag - u_min) / width)
                    survival = survival * sigmoid((u_max - u_mag) / width)
                elif element.type == "residual_gas_loss":
                    # expected survival of the stochastic residual-gas kill;
                    # a uniform, differentiable factor in hold_time and lifetime.
                    # Floor tau like the aperture branch abs()es its width: the
                    # schema forbids tau <= 0 at construction, but if tau is an
                    # optimization variable the optimizer can transit through 0,
                    # where exp(-hold/tau) would return a NaN gradient.
                    tau = jnp.maximum(element_params["mean_lifetime_s"], 1e-30)
                    hold = element_params["hold_time_s"]
                    survival = survival * jnp.exp(-hold / tau)
                elif element.type == "monitor":
                    continue
                else:  # pragma: no cover - _base_params rejects unsupported types
                    raise ValueError(f"unsupported element type: {element.type!r}")

            return jnp.sum(weight * survival) / total_weight

        self._value_fn = jax.jit(objective)
        self._value_and_grad_fn = jax.jit(jax.value_and_grad(objective))
        self._jnp = jnp

    def _theta(self, inputs: dict[str, float]):
        missing = set(self._order) - set(inputs)
        if missing:
            raise ValueError(f"missing variables: {sorted(missing)}")
        unexpected = set(inputs) - set(self._order)
        if unexpected:
            raise ValueError(f"unexpected variables: {sorted(unexpected)}")
        return self._jnp.asarray([float(inputs[name]) for name in self._order])

    def value(self, inputs: dict[str, float]) -> float:
        """Soft accepted fraction at the given variable values."""

        return float(self._value_fn(self._theta(inputs)))

    def value_and_grad(self, inputs: dict[str, float]) -> tuple[float, dict[str, float]]:
        """Soft accepted fraction and its gradient, keyed by variable."""

        value, grads = self._value_and_grad_fn(self._theta(inputs))
        return float(value), {name: float(grads[position]) for position, name in enumerate(self._order)}


def make_differentiable_objective(
    scene: Scene,
    variables: list[str],
    sharpness: float = 200.0,
) -> DifferentiableObjective:
    """Build a differentiable soft-acceptance objective for the scene."""

    return DifferentiableObjective(scene, variables, sharpness=sharpness)
