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

## GPU lane (WSL2, float32)

The float32 GPU lane runs the same `boris_step` kernel on
`jax[cuda12]`; float64 truth stays on the CPU NumPy reference (consumer
Blackwell cuts fp64 throughput, and the dimensionless u = p/(mc)
internals are float32-safe by design). Correctness is enforced by the
tiered suite in `tests/test_gpu_float32_validation.py` (strict x64
GPU-vs-CPU equality, trajectory, observable, and |u|-conservation
tiers) — it runs wherever `jax.devices("gpu")` is non-empty and skips
elsewhere, so CI stays CPU-only. Design record:
`docs/superpowers/specs/2026-07-06-gpu-float32-validation-design.md`.

Reference environment (the project's WSL2 box, RTX 5070 Ti):

```bash
# repo must live on ext4 (~), never /mnt/c — the vendored tree crawls over 9P
git clone https://github.com/ClawG0d/Latent-Dirac ~/Latent-Dirac
cd ~/Latent-Dirac && python3 -m venv .venv
.venv/bin/pip install -e ".[dev,jax]"
.venv/bin/pip install "jax[cuda12]==0.9.1"   # see the pin note below
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -m pytest tests/test_gpu_float32_validation.py -v
```

Pin note: jax/jaxlib 0.10.2 miscompiles this project's `lax.scan`
programs on Blackwell (sm_120) — same-precision GPU-vs-CPU divergence,
NaNs, and an MLIR verifier crash for x64 — while 0.9.1 passes every
tier (fp32 GPU agrees with fp32 CPU to ~2e-7). Details in the design
record. `XLA_PYTHON_CLIENT_PREALLOCATE=false` avoids benign
preallocation errors when Windows holds part of the VRAM.

Performance numbers live only in `docs/benchmarks.md` (full labels:
GPU model, WSL2, CUDA/driver, jax/jaxlib versions, integrator,
timestep, particle count, batch size, fidelity tier).
