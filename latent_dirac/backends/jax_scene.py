"""Batched scene execution on JAX: one launch, n_configs beamlines.

The scene is compiled into a single JAX program: a static Python loop over
elements builds the trace, transports run as `lax.scan` over time steps
calling the shared `boris_step` kernel with `xp=jax.numpy`, and acceptance
stages update the alive mask and stamp the loss ledger with the same
semantics as the NumPy pipeline. `vmap` maps the program over overridden
element parameters.

JAX is an optional dependency: `pip install "latent-dirac[jax]"`. Source
sampling stays on NumPy (RNG strategy from the positioning spec); arrays
cross to JAX at the State boundary in dimensionless momentum, so the
program is float32-safe by construction. Validation mode enables x64 and
compares element-wise against the NumPy float64 reference pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from latent_dirac.core.constants import ELEMENTARY_CHARGE_C, SPEED_OF_LIGHT_M_PER_S
from latent_dirac.scene.build import build_source
from latent_dirac.scene.schema import FIELD_ELEMENT_TYPES, Scene
from latent_dirac.solvers.kernels import boris_step

_SWEEPABLE_PARAMS = {
    "uniform_field": ("B_vector_t", "E_vector_v_m"),
    "solenoid": ("b_tesla", "radius_m", "length_m", "center_z_m"),
    "dipole": ("B_vector_t", "length_m", "center_z_m"),
    "quadrupole": ("gradient_t_m", "length_m", "center_z_m"),
    "penning_trap": ("v0_volt", "d_m", "b_tesla", "center_z_m"),
    "drift": (),
    "aperture": ("radius_m",),
    "momentum_window": ("p_min_gev_c", "p_max_gev_c"),
    "monitor": (),
    # numeric params are liftable, but the batched simulator still rejects
    # this element (stochastic kill has no static-program form); the
    # differentiable objective consumes them as the smooth expected-survival
    # factor exp(-hold/tau). See _make_simulator and backends/differentiable.py.
    "residual_gas_loss": ("mean_lifetime_s", "hold_time_s"),
}

_TRANSPORT_TYPES = (*FIELD_ELEMENT_TYPES, "drift")


def _import_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        raise ImportError(
            'The JAX backend requires jax. Install it with `pip install "latent-dirac[jax]"`.'
        ) from exc
    return jax, jnp


@dataclass
class BatchedSceneResult:
    """Final states for every configuration, as NumPy arrays."""

    position_m: np.ndarray  # (B, N, 3)
    momentum_kg_m_s: np.ndarray  # (B, N, 3)
    time_s: np.ndarray  # (B, N)
    alive: np.ndarray  # (B, N)
    lost_at_element: np.ndarray  # (B, N) int32, -1 = alive
    weight: np.ndarray  # (N,), shared across configurations
    accepted_weighted: np.ndarray  # (B,)
    accepted_fraction: np.ndarray  # (B,)
    trajectories: np.ndarray | None = None  # (B, S, N, 3) strided snapshots


def _base_params(scene: Scene) -> list[dict[str, np.ndarray]]:
    params = []
    for element in scene.elements:
        if getattr(element, "space_charge", None) is not None:
            raise ValueError(
                f"element {element.label!r} enables space_charge, which the JAX "
                "backend does not support: the state-dependent mean field breaks "
                "the static-program assumption; use the NumPy pipeline"
            )
        if element.type not in _SWEEPABLE_PARAMS:
            raise ValueError(
                f"element type {element.type!r} (label {element.label!r}) is not supported "
                "by the JAX backend yet; use the NumPy pipeline for this scene"
            )
        entry = {
            name: np.asarray(getattr(element, name), dtype=float) for name in _SWEEPABLE_PARAMS[element.type]
        }
        if getattr(element, "t_on_s", None) is not None:
            entry["t_on_s"] = np.asarray(element.t_on_s, dtype=float)
            entry["t_off_s"] = np.asarray(element.t_off_s, dtype=float)
        params.append(entry)
    return params


def _apply_gate(jnp, time_s, params, e_field, b_field):
    if "t_on_s" not in params:
        return e_field, b_field
    gate = (time_s >= params["t_on_s"]) & (time_s < params["t_off_s"])
    return (
        jnp.where(gate[:, None], e_field, 0.0),
        jnp.where(gate[:, None], b_field, 0.0),
    )


def _uniform_field(jnp, positions, time_s, params):
    count = positions.shape[0]
    e_field = jnp.broadcast_to(params["E_vector_v_m"], (count, 3))
    b_field = jnp.broadcast_to(params["B_vector_t"], (count, 3))
    return _apply_gate(jnp, time_s, params, e_field, b_field)


def _solenoid_field(jnp, positions, time_s, params):
    radial = jnp.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)
    inside = (radial <= params["radius_m"]) & (
        jnp.abs(positions[:, 2] - params["center_z_m"]) <= 0.5 * params["length_m"]
    )
    b_z = jnp.where(inside, params["b_tesla"], 0.0)
    zeros = jnp.zeros_like(b_z)
    return jnp.zeros_like(positions), jnp.stack([zeros, zeros, b_z], axis=1)


def _thin_sheet_solenoid_field(jnp, positions, time_s, params):
    # same algebra as fields.solenoid.ThinSheetSolenoidField: exactly
    # divergence-free first-order pair B_z = b(z), B_r = -(r/2) b'(z)
    r_sq = params["radius_m"] * params["radius_m"]
    zeta_plus = positions[:, 2] - params["center_z_m"] + 0.5 * params["length_m"]
    zeta_minus = positions[:, 2] - params["center_z_m"] - 0.5 * params["length_m"]
    root_plus = jnp.sqrt(zeta_plus * zeta_plus + r_sq)
    root_minus = jnp.sqrt(zeta_minus * zeta_minus + r_sq)
    b = 0.5 * params["b_tesla"] * (zeta_plus / root_plus - zeta_minus / root_minus)
    db_dz = 0.5 * params["b_tesla"] * r_sq * (root_plus**-3 - root_minus**-3)
    b_x = -0.5 * positions[:, 0] * db_dz
    b_y = -0.5 * positions[:, 1] * db_dz
    return jnp.zeros_like(positions), jnp.stack([b_x, b_y, b], axis=1)


def _dipole_field(jnp, positions, time_s, params):
    inside = jnp.abs(positions[:, 2] - params["center_z_m"]) <= 0.5 * params["length_m"]
    b_field = inside[:, None] * jnp.broadcast_to(params["B_vector_t"], positions.shape)
    return jnp.zeros_like(positions), b_field


def _quadrupole_field(jnp, positions, time_s, params):
    inside = jnp.abs(positions[:, 2] - params["center_z_m"]) <= 0.5 * params["length_m"]
    b_x = jnp.where(inside, params["gradient_t_m"] * positions[:, 1], 0.0)
    b_y = jnp.where(inside, params["gradient_t_m"] * positions[:, 0], 0.0)
    return jnp.zeros_like(positions), jnp.stack([b_x, b_y, jnp.zeros_like(b_x)], axis=1)


def _penning_trap_field(jnp, positions, time_s, params):
    scale = params["v0_volt"] / params["d_m"] ** 2
    e_field = jnp.stack(
        [
            0.5 * scale * positions[:, 0],
            0.5 * scale * positions[:, 1],
            -scale * (positions[:, 2] - params["center_z_m"]),
        ],
        axis=1,
    )
    b_z = jnp.broadcast_to(params["b_tesla"], positions.shape[:1])
    zeros = jnp.zeros_like(b_z)
    return _apply_gate(jnp, time_s, params, e_field, jnp.stack([zeros, zeros, b_z], axis=1))


def _drift_field(jnp, positions, time_s, params):
    zeros = jnp.zeros_like(positions)
    return zeros, zeros


_FIELD_FNS = {
    "uniform_field": _uniform_field,
    "solenoid": _solenoid_field,
    "dipole": _dipole_field,
    "quadrupole": _quadrupole_field,
    "penning_trap": _penning_trap_field,
    "drift": _drift_field,
}


def _field_fn_for(element):
    """Field function for an element; static profile dispatch for solenoids.

    Shared with the differentiable-objective mirror so profile dispatch
    lives in exactly one place.
    """

    if element.type == "solenoid" and element.profile == "thin_sheet":
        return _thin_sheet_solenoid_field
    return _FIELD_FNS[element.type]


def _make_simulator(scene: Scene, jax, jnp, mass_kg: float, charge_c: float, record: bool = False):
    # NOTE: backends/differentiable.py mirrors this element loop with soft
    # acceptance; when adding an element type here, extend the mirror too
    lax = jax.lax
    dt_s = scene.solver.dt_s
    c = SPEED_OF_LIGHT_M_PER_S

    def simulate(params, position, u, time_s, alive, ledger):
        history = [position[jnp.newaxis]] if record else None
        for index, element in enumerate(scene.elements):
            element_params = params[index]
            if element.type in _TRANSPORT_TYPES:
                steps = element.steps if element.steps is not None else scene.solver.steps
                field_fn = _field_fn_for(element)

                def step_fn(carry, _, field_fn=field_fn, element_params=element_params, alive=alive):
                    pos, u_now, t_now = carry
                    e_field, b_field = field_fn(jnp, pos, t_now, element_params)
                    new_carry = boris_step(
                        pos,
                        u_now,
                        t_now,
                        alive,
                        dt_s=dt_s,
                        charge_c=charge_c,
                        mass_kg=mass_kg,
                        e_field=e_field,
                        b_field=b_field,
                        xp=jnp,
                    )
                    return new_carry, (new_carry[0] if record else None)

                (position, u, time_s), emitted = lax.scan(step_fn, (position, u, time_s), None, length=steps)
                if record:
                    history.append(emitted)
            elif element.type == "aperture":
                radial = jnp.sqrt(position[:, 0] ** 2 + position[:, 1] ** 2)
                keep = radial <= element_params["radius_m"]
                newly_dead = alive & ~keep
                ledger = jnp.where(newly_dead, jnp.int32(index), ledger)
                alive = alive & keep
            elif element.type == "momentum_window":
                # bounds converted to dimensionless u so float32 stays safe
                to_u = 1.0e9 * ELEMENTARY_CHARGE_C / (c * mass_kg * c)
                u_min = element_params["p_min_gev_c"] * to_u
                u_max = element_params["p_max_gev_c"] * to_u
                u_mag = jnp.sqrt(jnp.sum(u * u, axis=1))
                keep = (u_mag >= u_min) & (u_mag <= u_max)
                newly_dead = alive & ~keep
                ledger = jnp.where(newly_dead, jnp.int32(index), ledger)
                alive = alive & keep
            elif element.type == "monitor":
                continue  # holds its ledger index; batched snapshots are a later extension
            elif element.type == "residual_gas_loss":
                # Intentional divergence from the differentiable mirror: the
                # hard model is a stochastic per-particle kill, which has no
                # static-program form. The differentiable objective models the
                # EXPECTED survival exp(-hold/tau) instead (a smooth factor).
                raise ValueError(
                    f"element {element.label!r} (residual_gas_loss) cannot be batched by the "
                    "JAX backend: stochastic annihilation kill has no static-program form. "
                    "Use the NumPy pipeline, or the differentiable objective for its "
                    "expected-survival relaxation."
                )
            else:  # pragma: no cover - schema union prevents this
                raise ValueError(f"unsupported element type for the JAX backend: {element.type!r}")
        if record:
            return position, u, time_s, alive, ledger, jnp.concatenate(history, axis=0)
        return position, u, time_s, alive, ledger

    return simulate


def base_parameters(scene: Scene) -> dict[str, np.ndarray]:
    """Sweepable parameters as a flat `"label.param" -> base value` map."""

    params = _base_params(scene)
    return {
        f"{element.label}.{name}": value
        for element, entry in zip(scene.elements, params, strict=True)
        for name, value in entry.items()
    }


class BatchedSceneProgram:
    """Compile a scene once, run it many times with different overrides.

    The source is sampled once (deterministic from the scene seed) and
    the vmapped simulator is staged for jit at construction; JAX compiles
    on the first `run` and caches per batch shape, so each distinct batch
    size compiles once and every further `run` with that shape reuses it.
    This is the path repeated callers (parameter scans, optimizers)
    should use.

    Dtype policy: outputs follow JAX's dtype configuration. Without
    `jax_enable_x64`, float64 inputs are downcast and results come back
    float32 - safe for the dimensionless internals, but avoid computing
    float32 SI-momentum magnitudes downstream (|p|^2 underflows); enable
    x64 for float64-validated results.
    """

    def __init__(
        self,
        scene: Scene,
        override_keys: tuple[str, ...] | list[str] = (),
        rng: np.random.Generator | None = None,
        record_stride: int | None = None,
    ):
        if record_stride is not None and record_stride < 1:
            raise ValueError("record_stride must be at least 1")
        self._record_stride = record_stride
        jax, jnp = _import_jax()
        self._jnp = jnp
        self._scene = scene

        params = _base_params(scene)
        label_to_index = {element.label: index for index, element in enumerate(scene.elements)}
        in_axes = [dict.fromkeys(entry, None) for entry in params]
        self._slots: dict[str, tuple[int, str, tuple[int, ...]]] = {}
        for key in override_keys:
            label, _, param = key.partition(".")
            if label not in label_to_index:
                raise ValueError(f"override {key!r}: no element labeled {label!r} in the scene")
            index = label_to_index[label]
            if param not in params[index]:
                raise ValueError(
                    f"override {key!r}: element {label!r} has no sweepable parameter {param!r} "
                    f"(sweepable: {sorted(params[index])})"
                )
            in_axes[index][param] = 0
            self._slots[key] = (index, param, params[index][param].shape)
        self._base = params

        rng = np.random.default_rng(scene.seed) if rng is None else rng
        initial = build_source(scene).sample(rng)
        self._mass_kg = initial.species.mass_kg
        self._weight = np.asarray(initial.weight)
        self._initial = (
            jnp.asarray(initial.position_m),
            jnp.asarray(initial.momentum_kg_m_s / (self._mass_kg * SPEED_OF_LIGHT_M_PER_S)),
            jnp.asarray(initial.time_s),
            jnp.asarray(initial.alive),
            jnp.asarray(initial.lost_at_element),
        )

        simulate = _make_simulator(
            scene,
            jax,
            jnp,
            self._mass_kg,
            initial.species.charge_c,
            record=record_stride is not None,
        )
        if self._slots:
            self._fn = jax.jit(jax.vmap(simulate, in_axes=(in_axes, None, None, None, None, None)))
        else:
            self._fn = jax.jit(simulate)

    def run(self, values: dict[str, np.ndarray] | None = None) -> BatchedSceneResult:
        jnp = self._jnp
        values = dict(values) if values is not None else {}

        missing = set(self._slots) - set(values)
        if missing:
            raise ValueError(f"missing override values for {sorted(missing)}")
        unexpected = set(values) - set(self._slots)
        if unexpected:
            raise ValueError(f"unexpected override values for {sorted(unexpected)}")

        params = [dict(entry) for entry in self._base]
        batch_size: int | None = None
        for key, array_values in values.items():
            index, param, base_shape = self._slots[key]
            array = np.asarray(array_values, dtype=float)
            if array.ndim != len(base_shape) + 1 or array.shape[1:] != base_shape:
                raise ValueError(
                    f"override {key!r} must have shape (batch, *{base_shape}), got {array.shape}"
                )
            if array.shape[0] == 0:
                raise ValueError(f"override {key!r}: batch size must be at least 1, got 0")
            if batch_size is None:
                batch_size = array.shape[0]
            elif array.shape[0] != batch_size:
                raise ValueError(f"override {key!r}: batch size {array.shape[0]} does not match {batch_size}")
            params[index][param] = array

        jax_params = [{name: jnp.asarray(value) for name, value in entry.items()} for entry in params]
        raw = self._fn(jax_params, *self._initial)
        if self._slots:
            outputs = tuple(np.asarray(value) for value in raw)
        else:
            outputs = tuple(np.asarray(value)[np.newaxis] for value in raw)
        return self._package(outputs)

    def _package(self, outputs) -> BatchedSceneResult:
        trajectories = None
        if self._record_stride is not None:
            *outputs, history = outputs
            # full per-step emission strided host-side; memory at emission is
            # B x T x N x 3 doubles - fine for demo scales, streaming for
            # extreme scales is a later design
            trajectories = history[:, :: self._record_stride]
        position_out, u_out, time_out, alive_out, ledger_out = outputs
        accepted_weighted = np.sum(self._weight[np.newaxis, :] * alive_out, axis=1)
        total_weight = float(np.sum(self._weight))
        return BatchedSceneResult(
            position_m=position_out,
            momentum_kg_m_s=u_out * (self._mass_kg * SPEED_OF_LIGHT_M_PER_S),
            time_s=time_out,
            alive=alive_out.astype(bool),
            lost_at_element=ledger_out.astype(np.int32),
            weight=self._weight,
            accepted_weighted=accepted_weighted,
            accepted_fraction=accepted_weighted / total_weight if total_weight > 0.0 else accepted_weighted,
            trajectories=trajectories,
        )


def run_scene_batched(
    scene: Scene,
    overrides: dict[str, np.ndarray] | None = None,
    rng: np.random.Generator | None = None,
    record_stride: int | None = None,
) -> BatchedSceneResult:
    """Run the scene once per configuration in a single JAX program.

    `overrides` maps `"<element label>.<param>"` to an array with a leading
    batch axis; all overrides must share the batch size. With no overrides
    the scene runs as a single configuration (batch size 1).

    This is a one-shot convenience wrapper around `BatchedSceneProgram`;
    repeated callers should build the program once and call `run` (see the
    dtype policy documented there).
    """

    overrides = dict(overrides) if overrides is not None else {}
    program = BatchedSceneProgram(scene, override_keys=tuple(overrides), rng=rng, record_stride=record_stride)
    return program.run(overrides)
