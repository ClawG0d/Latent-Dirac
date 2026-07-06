# Latent Dirac

Latent Dirac is an open, interactive simulation platform for antimatter
factories — positron and antiproton facilities from source through
transport to capture — built to turn facility design iteration from a
wall-clock problem into a compute problem.

Declarative scenes describe the beamline. Batched solvers sweep whole
configuration families in one launch. A per-particle ledger accounts for
every antiparticle, because antiparticles are extraordinarily expensive.

## How the platform is organized

A declarative **scene interface** (YAML/JSON schema, CLI, 3D rendering)
drives a set of **solvers** — one authoritative component per physics
domain — that all exchange the same `ParticleState`: a pytree-compatible
state carrying an alive mask, a per-particle loss ledger, and result
provenance. The **compute** layer pairs a NumPy float64 reference
pipeline with a JAX backend (jit, vmap, autodiff) and the vendored
vanilla Geant4 engine for particle–matter physics.

Every physics model declares one of five fidelity tiers — placeholder,
parameterized, surrogate, table-based, or externally calibrated — and no
comparative performance claim is made without an open, reproducible
benchmark.

## Where to go next

- [Architecture](architecture.md) — the layers and how they fit together.
- [Scene schema](scene_schema.md) — the declarative beamline description.
- [Physics scope](physics_scope.md) — what is modeled and at what fidelity.
- [Source models](source_models.md) — positron and antiproton source terms.
- [Solver backends](solver_backends.md) — the NumPy reference and JAX batch.
- [Rendering](rendering.md) — 3D visualization of scenes and trajectories.
- [Validation plan](validation_plan.md) — how models are checked.
- [Safety scope](safety_scope.md) — the canonical red lines.
- [License strategy](license_strategy.md) — first-party and vendored terms.
- [Roadmap](roadmap.md) — the phased plan.

The full source, demos, and design records live in the
[GitHub repository](https://github.com/ClawG0d/Latent-Dirac).
