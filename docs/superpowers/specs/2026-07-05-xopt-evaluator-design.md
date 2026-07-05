# Xopt-Compatible Evaluator Design (Spec 3b)

## Objective

Plug the batched JAX backend into the accelerator community's de facto
tuning-optimization ecosystem (Xopt/Badger) — without inventing our own
optimizer, per the positioning research. Deliverable: a scene evaluator in
Xopt's calling convention (`dict[str, float] -> dict[str, float]`) plus a
batched variant that evaluates a whole generation of candidates in one
JAX launch.

Xopt itself is *not* a dependency: the evaluator is a plain callable that
any Xopt `Evaluator(function=...)` can wrap. An integration test runs only
when xopt happens to be installed.

## Design

### `BatchedSceneProgram` (`latent_dirac/backends/jax_scene.py`)

Compile once, run many — closes the per-call recompilation gap flagged in
the 3a review:

- `BatchedSceneProgram(scene, override_keys, rng=None)` samples the source
  once (deterministic from the scene seed), builds the params pytree and
  `in_axes` for the given override keys, and stages the vmapped program
  for jit (compilation happens lazily on the first `run` per batch shape
  and is cached).
- `program.run(values: dict[key, (B, ...) array]) -> BatchedSceneResult`
  reuses the cached compilation; only the override values change. Same
  batch size across calls avoids retracing; a different batch size
  retraces once (standard JAX behavior, documented).
- `run_scene_batched` becomes a thin wrapper (build a program, run once) —
  public behavior unchanged.

### Evaluator (`latent_dirac/backends/evaluator.py`)

```python
from latent_dirac.backends.evaluator import make_scene_evaluator

evaluate = make_scene_evaluator(
    scene,
    variables=["capture-solenoid.b_tesla", "sweep-field.B_vector_t[1]"],
)
evaluate({"capture-solenoid.b_tesla": 0.9, "sweep-field.B_vector_t[1]": 0.2})
# -> {"accepted_fraction": ..., "accepted_weighted": ...}

evaluate.batch({"capture-solenoid.b_tesla": np.array([...]), ...})
# -> {"accepted_fraction": (B,) array, ...}  — one JAX launch per generation
```

- Variables are scalar scene parameters (`label.param`) or single vector
  components (`label.vec[i]`); components merge into full-vector overrides
  on top of the scene's base values. Unknown labels/params/indices raise
  at construction (fail-fast).
- `evaluate(inputs)` is the Xopt calling convention; missing or extra
  input keys raise. `evaluate.batch(inputs)` evaluates B candidates in
  one launch — the intended path for generation-based optimizers.
- Objectives: `accepted_fraction` and `accepted_weighted` (the loss-ledger
  aggregates already computed by the backend).
- No xopt import anywhere in the package.

## Validation

- Program reuse: two `run` calls with different values give results
  matching fresh `run_scene_batched` calls.
- Evaluator parity: single-point evaluation matches the NumPy pipeline's
  accepted fraction for the same parameter value (x64).
- Vector-component variables reconstruct the correct full-vector override.
- `batch` matches per-point evaluation.
- Fail-fast: unknown variable at construction; wrong/missing keys at call.
- Optional: an end-to-end Xopt `random` generator smoke test behind
  `importorskip("xopt")` (not installed in CI).

## Non-Goals

- shipping xopt as a dependency or extra; Badger plugin packaging
- gradient-based optimization through the program (autodiff objectives are
  a later spec on top of this one)
- multi-objective bookkeeping beyond the two standard outputs
