# Solver Backends

## Reference backend: relativistic Boris (NumPy, float64)

`RelativisticBorisSolver` is the reference backend and the source of
truth. It advances relativistic momentum and position for the whole
cloud using gamma and velocity from momentum, and the Lorentz force from
the electric and magnetic fields. It is validated against uniform-field
analytic behavior (kinetic-energy preservation and the Larmor radius).

The push itself is a pure-function kernel (`solvers.kernels.boris_step`)
in dimensionless momentum u = p/(m c); SI appears only at `ParticleState`
boundaries (float32 with SI momenta underflows at MeV scale, so this is
load-bearing). The kernel is array-namespace generic (`xp = numpy` or
`jax.numpy`), so every backend below runs the same physics.

## JAX batched backend

`latent_dirac.backends.jax_scene` compiles a declarative scene into one
JAX program (`lax.scan` in time, mask-and-ledger acceptance) and `vmap`s
it over overridden element parameters:
`run_scene_batched(scene, overrides={"label.param": values})`. It is
enforced bit-for-bit against the NumPy float64 pipeline in CI. Optional
`[jax]` extra. Stochastic and state-dependent elements (space charge,
`residual_gas_loss`, `matter_slab`, `xsuite_lattice`) are rejected — they
have no static-program form; use the NumPy pipeline for those.

## Differentiable objective

`latent_dirac.backends.differentiable.make_differentiable_objective`
relaxes the hard acceptance chain into smooth per-particle survival
weights (sigmoids of the cut margins) and returns gradients of the soft
accepted fraction w.r.t. named scene variables via JAX autodiff. Storage
survival enters as the expected factor exp(-hold/tau), so capture and
storage-survival optimize jointly. The relaxation is an optimization
device — the hard pipelines remain the source of truth, and optima should
be validated against them.

New backends can be added behind the same solver interface; the pure
`xp`-generic kernel keeps the physics shared across all of them.
