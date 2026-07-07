# GPU float32 validation: tolerance policy and pytree registration

Date: 2026-07-06. Status: accepted. G2 of the execution plan
(2026-07-06-execution-plan-gpu-to-phase4-design.md).

## Scope

Correctness only — no performance numbers leave this stage (those are
G3, docs/benchmarks.md). Two deliverables: a real JAX pytree
registration for `ParticleState`, and a tiered validation suite that
runs the float32 GPU lane against the float64 CPU reference.

## ParticleState pytree registration

`ParticleState.tree_flatten/tree_unflatten` exist but the docstring
records why naive registration fails: `tree_unflatten` re-runs
`__post_init__` (which coerces dtypes and validates shapes — both
illegal on tracer leaves), and the mutable `metadata` dict in the
static aux is unhashable for jit cache keys.

Design (`latent_dirac/backends/pytree_state.py`):

- `register_particle_state_pytree()` registers dedicated
  flatten/unflatten functions (idempotent; not imported for side
  effects anywhere — callers opt in).
- Unflatten builds the instance via `object.__new__` and sets fields
  directly — no `__post_init__`, so tracer and placeholder leaves
  pass through untouched.
- **Metadata does not cross the boundary**: flatten's static aux is
  the species only (frozen pydantic model, hashable); unflattened
  states carry `metadata={}`. Rationale: metadata is provenance and
  diagnostics, not simulation state — the engineering rule already
  bans it from hot paths, and any hashing scheme for a mutable dict
  is either wrong (identity hash: cache misses per instance) or lossy
  anyway. Callers who need metadata re-attach it outside the jit
  boundary. This is documented in the function and the class
  docstring.

## Tolerance tiers (fp32 GPU vs fp64 CPU reference)

The float32 lane works in dimensionless u = p/(mc) by design; SI only
at State boundaries. Expected error sources: fp32 rounding
(~6e-8/step, random-walk accumulation), fused-multiply-add and
reduction-order differences on GPU. Tiers:

1. **Strict tier (x64-on-GPU)**: with `jax_enable_x64`, the GPU run
   must match the CPU JAX run bitwise-or-near (rtol 1e-12): isolates
   hardware/compiler divergence from precision loss. Any failure here
   is a bug, not a tolerance question.
2. **Trajectory tier (fp32)**: per-scene, per-element-type cases
   (uniform, thin-sheet solenoid, dipole, quadrupole, Penning trap;
   the scenes the JAX backend supports). Endpoint positions:
   rtol 1e-4 relative to the trajectory scale (Larmor radius or
   element aperture), after O(100–1000) steps. Dimensionless momenta:
   rtol 1e-5.
3. **Observable tier (fp32)**: accepted fraction and weighted counts
   must match the fp64 reference exactly for robust cuts (margins
   ≫ fp32 noise) — scenes are chosen so no particle sits within 1e-3
   relative of a cut boundary; where that cannot be arranged, the
   tolerance is one particle.
4. **Conservation tier (fp32, absolute)**: |u| drift in pure-B
   transport < 1e-4 relative over the case length (Boris conserves
   |u| exactly in exact arithmetic; drift measures fp32 rounding
   walk).

Test mechanics: `tests/test_gpu_float32_validation.py` — module
importorskips jax, and each GPU case skips unless
`jax.devices("gpu")` is non-empty, so CI (CPU-only) stays green and
the WSL box runs the real thing. The fp32-vs-fp64 comparisons reuse
`run_scene_batched` with dtype control via `jax_enable_x64` toggles
per test (config isolation via context managers).

## Out of scope

Field maps / batched monitors in the JAX backend (roadmap items),
multi-GPU, and any timing. The mirror pair is untouched.
