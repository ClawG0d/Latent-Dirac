# Differentiable Capture Objective Design (Spec 3d)

## Objective

First landing of the "differentiable antimatter facility design"
narrative, scoped exactly where the positioning research put it: gradients
of the *capture chain* (trap-bound, low-energy) — distinct from existing
differentiable beam-dynamics prior art (Cheetah), and integrated with the
same scene/variable vocabulary as the Xopt evaluator rather than a
parallel system.

Deliverable: `d(soft accepted fraction) / d(scene parameters)` through the
whole transport program via JAX autodiff.

## Design

### Soft relaxation (honestly labeled)

The hard accepted fraction is a step function of scene parameters (a
particle is either cut or not), so its gradient is zero almost everywhere.
For optimization we relax the acceptance stages into smooth survival
weights `s ∈ [0, 1]` per particle:

- aperture: `s *= sigmoid((radius - r) / w)` with width
  `w = radius / sharpness`
- momentum window: `s *= sigmoid((u - u_min)/w_u) * sigmoid((u_max - u)/w_u)`
  with `w_u = (u_max - u_min) / sharpness`
- transports are already smooth (pure Boris kernel); in soft mode no
  particle is frozen — the survival weight is bookkeeping, and in the
  `sharpness → ∞` limit the soft objective converges to the hard
  `accepted_fraction` because survivors' trajectories are identical.

The relaxation is an optimization device, not a physics claim: the hard
NumPy/JAX pipelines remain the source of truth, and every docstring says
so (honesty discipline).

### API (`latent_dirac/backends/differentiable.py`)

```python
objective = make_differentiable_objective(
    scene,
    variables=["hello-solenoid.b_tesla", "hello-aperture.radius_m"],
    sharpness=200.0,
)
value = objective.value(inputs)                  # soft accepted fraction
value, grads = objective.value_and_grad(inputs)  # grads keyed by variable
```

- Variables reuse the evaluator's vocabulary (`label.param`,
  `label.vec[i]`) and fail-fast validation; the parsing helper is shared
  with `SceneEvaluator`, not duplicated.
- Composition of variable values into the params pytree happens in
  traceable JAX ops (`.at[i].set`), so `jax.value_and_grad` differentiates
  through it into every transport step.
- Single configuration (no vmap); jit + grad staged once per objective.

## Validation

- Gradient matches central finite differences of the soft objective
  (x64, tight tolerance).
- `sharpness → ∞`: soft value approaches the hard `accepted_fraction`
  from the batched backend on the hello scene.
- Physical sanity: `d(yield)/d(aperture radius) > 0`.
- A few steps of deterministic gradient ascent on the hello scene increase
  the soft objective.
- Fail-fast variable validation shared with the evaluator keeps working.

## Non-Goals

- optimizing through field maps (JAX field-map interpolation not landed)
- stochastic-source gradients (the source sample is fixed per objective,
  matching the evaluator's determinism)
- publishing optimization benchmarks or performance numbers
