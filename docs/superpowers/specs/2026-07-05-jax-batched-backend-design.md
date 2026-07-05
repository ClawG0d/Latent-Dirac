# JAX Batched Backend Design (Spec 3a)

## Objective

Deliver the first Phase 3 milestone: run a declarative scene as a single
JAX program, batched over configurations with `vmap`, with the NumPy
float64 pipeline as the bit-level reference. This turns the
magnetic-control-sweep story into an API: one launch, `n_configs`
beamlines.

The 2c refactor is the foundation: the Boris kernel is already a pure
function in dimensionless momentum (float32-safe by construction), and
`ParticleState` is pytree-compatible.

## Design

### Single-source kernel

`latent_dirac.solvers.kernels.boris_step` gains an `xp` array-namespace
parameter (default `numpy`). The same algebra runs on `jax.numpy` — no
duplicated physics. The NumPy call sites are unchanged.

### JAX scene runner (`latent_dirac/backends/jax_scene.py`)

- JAX is an optional extra (`pip install "latent-dirac[jax]"`); the module
  raises a clear ImportError otherwise. The core stays JAX-free.
- Supported scene vocabulary v1: `uniform_field`, `solenoid`, `dipole`,
  `quadrupole`, `drift` (analytic hard-edge field functions in
  `jax.numpy`), `aperture`, `momentum_window` (mask updates), `monitor`
  (pass-through; batched snapshots are a later extension). `field_map`
  scenes are rejected with a clear error until the interpolating backend
  lands.
- Program shape: a static Python loop over scene elements builds the
  traced program; transports are `lax.scan` over time steps calling
  `boris_step(..., xp=jnp)`; acceptance stages update `alive` and stamp
  `lost_at_element` with the element index (identical semantics to the
  NumPy pipeline, including monitor stages holding their index).
- Momentum-window bounds convert to dimensionless u at the boundary, so
  the comparison is float32-safe (SI momentum magnitudes underflow in
  float32; |u| is O(1)).
- Source sampling stays on NumPy (RNG strategy from the positioning
  spec); arrays convert to JAX at the State boundary.

### Batched API

```python
from latent_dirac.backends.jax_scene import run_scene_batched

result = run_scene_batched(
    scene,
    overrides={"transport-field.B_vector_t": b_vectors},  # (B, 3)
)
result.accepted_fraction  # (B,)
result.alive              # (B, N)
result.lost_at_element    # (B, N)
```

- `overrides` maps `"<element label>.<param>"` to an array with a leading
  batch axis; non-overridden parameters broadcast. Unknown labels or
  params raise `ValueError` (fail-fast).
- Implementation: parameters live in a per-element pytree; `vmap` maps
  only the overridden leaves (`in_axes` mirrors the pytree); the mapped
  function is `jit`-compiled.
- Outputs return as NumPy arrays.

### Precision policy

Runtime default follows JAX's dtype configuration (float32 unless the
caller enables x64). Validation mode: tests enable `jax_enable_x64` and
compare element-wise against the NumPy float64 pipeline. Performance
numbers, when they are eventually published, must follow the honesty
discipline (settings + hardware); this spec publishes none.

## Validation

- Parity (x64): `run_scene_batched` with B=1 reproduces `run_scene` on the
  ledger scene and the dipole/quad scene — identical `alive` and
  `lost_at_element`, positions/momenta allclose at tight tolerance.
- Sweep correctness: a uniform-field By sweep matches a per-config NumPy
  loop for accepted fractions and final positions.
- Fail-fast: unknown override label/param and field-map scenes raise.
- The full suite stays green without JAX installed (importorskip).

## Non-Goals

- GPU benchmarks and performance claims (need Linux CUDA hardware)
- field-map interpolation in JAX, batched monitor snapshots
- Xopt evaluator (next spec), interactive viewer, trajectory streaming
