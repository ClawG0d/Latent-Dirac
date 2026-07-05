# State/Model/Control Refactor Design (Spec 2c)

## Objective

Replace the pydantic `ParticleCloud` with a pytree-compatible `ParticleState`
dataclass, rewrite the Boris pusher as a pure-function kernel in
dimensionless momentum, and land the per-particle loss ledger. This is a
pure refactor: no new physics, and it is the structural prerequisite for the
Phase 3 JAX backend. One-time break under 0.x semantics; no deprecation
shims.

Constraints locked by the positioning spec
(`2026-07-05-platform-positioning-and-roadmap-design.md`):

- Container layering: pydantic owns Model/scene schema only. Simulation
  State must be a pytree-compatible container (dataclass), never pydantic.
- Kernels are pure functions using dimensionless momentum `u = p/(m c)`;
  SI exists only at State boundaries. Rationale: float32 with SI momenta
  underflows (3 MeV positron p² ≈ 3.5e-42 < float32 smallest normal).
- The per-particle loss channel `lost_at_element: int32` (-1 = alive) is
  the ledger's data model under static array shapes.

## Design

### `ParticleState` (`latent_dirac/state/particle_state.py`)

A mutable dataclass with the same attribute and method surface as the old
`ParticleCloud` (`species`, `position_m`, `momentum_kg_m_s`, `time_s`,
`weight`, `alive`, `particle_id`, `parent_id`, `metadata`,
`weighted_count()`, `gamma()`, `velocity()`, `kinetic_energy_joule()`,
`mean_kinetic_energy_joule()`, `copy()`, `apply_alive_mask()`), plus:

- `lost_at_element: np.ndarray` int32 `(N,)`, defaulting to -1. Particles
  are never deleted; losses are stamped, keeping array shapes static.
- `tree_flatten()` / `tree_unflatten()` in the JAX pytree convention
  (array fields as leaves; `species` and `metadata` as static aux data).
  JAX itself is not imported. Phase 3 registration will need a thin
  wrapper: unflatten must bypass `__post_init__` validation (JAX may
  unflatten with tracer/placeholder leaves) and the mutable `metadata`
  dict is not hashable for jit cache keys.
- `__post_init__` validation replicating the old shape/dtype/weight checks
  (fail-fast is unchanged).

`ParticleCloud` and `latent_dirac/state/particle_cloud.py` are deleted.
`Trajectory` (a recording artifact, not on the hot path) and solver/source
configuration classes stay pydantic — they are Model-layer objects.

### Pure Boris kernel (`latent_dirac/solvers/kernels.py`)

`boris_step(position_m, u, time_s, alive, dt_s, charge_c, mass_kg,
e_field, b_field)` returns new `(position_m, u, time_s)`:

- dimensionless momentum `u = p/(m c)`; `gamma = sqrt(1 + |u|²)`;
  velocity `= u c / gamma`. Algebra is the standard Boris rotation,
  identical to the previous SI formulation up to floating-point rounding.
- Full-array computation with `np.where(alive, new, old)` freezing dead
  particles. No fancy-indexed in-place writes, no data-dependent early
  exit — the two patterns that are illegal under `jit`/`lax.scan`.
- Pure NumPy float64 reference implementation; the Phase 3 JAX backend
  swaps `np` for `jnp` without touching the algebra.

`RelativisticBorisSolver.propagate(state, field)` keeps its signature: it
converts SI momentum to `u` once before the step loop, evaluates the field
at all particle positions each step (dead particles are frozen by the
kernel), always runs exactly `steps` iterations, and converts back to SI
at the boundary. Trajectories recorded by callers step the solver as
before.

### Loss ledger stamping

- `Stage.run(state, stage_index)` stamps
  `lost_at_element[newly_dead] = stage_index` where
  `newly_dead = alive_before & ~alive_after`. Elements themselves stay
  ledger-agnostic; the pipeline layer owns stage indexing.
- `PipelineRunner` passes the index; scene labels remain the stage names,
  so the ledger is label-addressable.
- New diagnostic `loss_ledger(final_state, stage_results)` returns weighted
  losses per stage name (plus survivors) reconstructed purely from
  `lost_at_element`, and must agree with the per-stage `StageResult.losses`
  accounting.

## Migration (one-time break)

All `ParticleCloud` imports move to `ParticleState` (identical call sites):
sources (including the `particle_arrays` helper, which now also returns
`lost_at_element`), beamline elements, pipeline, diagnostics, io, scene
build, viz backends, 4 examples, both asset generator tools, and the
affected tests. `tests/test_particle_cloud.py` is replaced by
`tests/test_particle_state.py`. README example outputs and the WebP/3D
assets are regenerated (the dimensionless algebra can shift last digits).

## Validation

- Container: construction/validation parity with the old tests, copy
  independence, mask AND semantics, ledger default -1,
  `tree_flatten`/`tree_unflatten` round-trip.
- Kernel: kinetic-energy preservation in a uniform B field, Larmor radius
  against the analytic value (existing analytic tests continue to pass
  after migration), dead particles frozen, no early exit (fixed iteration
  count).
- Ledger: stamped indices match the stage that killed each particle; the
  reconstructed ledger equals `StageResult.losses` per stage; later stages
  never overwrite an earlier stamp.
- Full suite green, ruff clean, core stays viz-free, demos run, assets
  regenerate.

## Non-Goals

- JAX backend, vmap batching (Phase 3)
- Coupler implementation (the `SolverBase` contract stays; the Coupler
  remains a documented slot in `docs/architecture.md`)
- new physics of any kind
