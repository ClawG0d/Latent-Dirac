# Solver-zoo composition design

Date: 2026-07-05
Status: adopted (docs-only; no code, schema, or safety-scope changes)

## Decision

Latent Dirac's platform pillar is made concrete as a **solver zoo behind
one spine**: independent solver components, each authoritative over one
physics domain, composed by the scene schema and exchanging particles
through `ParticleState` at stage boundaries. The reference point is
Genesis (the robotics simulation platform), with one deliberate
correction to the analogy:

- In Genesis, Taichi is a homogeneous compute substrate, and every
  solver (rigid body, MPM, SPH, FEM) is first-party code written on it.
- In Latent Dirac, the Taichi role is played by the NumPy/JAX kernel
  substrate. The CERN-ecosystem tools (Geant4, Xsuite, WarpX,
  Garfield++) are *not* the substrate: they are members of the solver
  zoo — heterogeneous, engine-backed components behind adapters.

Genesis could rewrite all of its physics onto one substrate because its
domains are re-implementable by a small team. Latent Dirac deliberately
does not rewrite particle–matter physics: in-house shower physics is a
canonical safety-scope exclusion, and the vendored engine's three
decades of experimental validation pedigree are the reason to use it at
all (see the engine positioning spec). The platform is therefore a
hybrid by design:

- physics that is re-implementable and benefits from vmap/grad lives in
  first-party solvers on the substrate (vacuum transport, traps,
  mean-field collective effects);
- physics whose value is validation pedigree enters as engine-backed
  solvers behind adapters (matter interaction, lattice tracking, PIC,
  detector response).

"Like Taichi" therefore translates to the user experience, not to the
architecture: engines are demoted from tools the user must learn to
implementation details behind the scene schema. Users write scenes; the
platform routes each stage to whichever solver is authoritative there.
API invisibility never extends to provenance: engine-derived results
keep the four-tuple (engine version, physics list, dataset versions,
patch list) and the naming rules of the engine positioning spec.

## The zoo

| Component  | Authority domain                   | Form        | Backing                                       | Status |
| ---------- | ---------------------------------- | ----------- | --------------------------------------------- | ------ |
| Source     | positron / antiproton source terms | sampler     | first-party (pair, beta-plus, surrogate) + engine yield-table replay | shipped (first engine table landed with yieldgen); more per M3 |
| Transport  | vacuum EM transport                | stepper     | first-party Boris kernel (NumPy + JAX)        | shipped |
| Lattice    | decelerator rings, transfer lines  | stepper     | Xsuite adapter                                | planned (closed-loop v1) |
| Matter     | targets, degraders, annihilation   | transformer | vendored vanilla Geant4 v11.4.2               | builds via recipe (M1'-lite); offline tables only, no runtime coupling |
| Collective | in-trap space charge               | stepper     | first-party mean-field v1, later WarpX        | planned (mean-field v1 in closed-loop v1) |
| Detector   | detector response                  | transformer | parameterized model first, Garfield++ later   | planned |
| Analysis   | persistent output, ecosystem exchange | sink     | openPMD + ROOT via uproot                     | planned (closed-loop v1) |

Forms:

- **sampler** — produces an initial `ParticleState` from a seeded RNG.
- **stepper** — advances `ParticleState` step by step; first-party
  steppers live on the substrate and are batchable and differentiable.
- **transformer** — an engine boundary: particle cloud in, particle
  cloud plus events out. Not steppable at our timestep granularity, not
  differentiable.
- **sink** — consumes state and events, produces files and reports.

## Interface contract

- The exchange currency at every component boundary is `ParticleState`
  (SoA pytree, SI at boundaries), plus event records that flow into the
  per-particle ledger. The ledger spans component boundaries — "which
  stage killed this antiproton" stays answerable across engines; no
  tool in the reference chain answers this on its own.
- Every component output carries a fidelity tier; engine-backed output
  additionally carries the engine four-tuple.
- The coupling model is **staged handoff** (start-to-end simulation),
  not intra-step co-simulation: the factory chain is sequential
  (source → target → decelerator → trap → annihilation/detection →
  analysis). This is deliberately weaker coupling than Genesis's
  intra-step solver coupling, and it is what makes engine heterogeneity
  tractable — each adapter lands and is validated independently.
- Gradients flow through first-party steppers and stop at transformers.
  The recovery path is distillation: engine runs produce yield tables
  (M3) and later surrogates, which re-enter the differentiable
  substrate as `table_based` / `externally calibrated` sources. The
  design loop stays differentiable through distilled engines; the
  engines stay the truth anchor.

## Implementation order (adopted)

This spec changes documents only; the adopted code order is:

1. **Closed-loop v1** (Python side): openPMD output (the deferred 2e) →
   ROOT I/O via uproot → Xsuite adapter (the first adapter to become
   real: `ParticleState` ↔ `xtrack.Particles` round-trip plus a
   decelerator-style validation case; the
   `test_only_placeholder_adapters_are_present` gate flips in the same
   change) → native mean-field space charge (new physics: fidelity tier
   and validity envelope declared explicitly).
2. **Engine track M1'** in parallel on the WSL2 box (build recipes;
   WSL2 is a real Linux target, so recipes are built and validated
   there directly). The first deliverables landed the same day this
   spec was adopted (M1'-lite build recipe + M3-lite yieldgen table;
   see the engine-yieldgen demo spec).
3. **GPU lane** after closed-loop v1: float32 backend validation
   against the float64 reference, then the honest benchmark suite.
4. **Interactive viewer** after that.

## Honesty guardrails

- Any public zoo table carries a status column; planned components are
  named as planned, never implied as shipped.
- The cautionary precedent is Genesis's own launch, whose headline
  speed claims were publicly disputed within days. The positioning
  tests already forbid comparative performance wording without an open
  benchmark; the zoo narrative changes nothing there.
- GPU numbers from the project's CUDA box (RTX 5070 Ti under WSL2) are
  publishable only with full labels: GPU model, WSL2, CUDA/driver
  versions, integrator, timestep, particle count, batch size, fidelity
  tier. WSL2 is part of the label, not a footnote. The float32 lane is
  the intended fit for consumer Blackwell, whose FP64 throughput is
  heavily reduced relative to FP32; the NumPy float64 reference stays
  the truth anchor.
- Geant4 naming rules from the engine positioning spec apply verbatim
  to any zoo narrative copy.

## Alternatives considered and rejected

- **Full rewrite on one substrate (the literal Genesis move)** —
  excluded by the safety scope (in-house shower physics) and by the
  engine decision record: validation pedigree cannot be rewritten.
- **Casting the CERN tools as the substrate** — a category error: they
  are heterogeneous engines with incompatible state and execution
  models, no shared data layout, and no differentiability. The
  substrate is NumPy/JAX; the engines are zoo members.
- **Component-per-repository split** — declined; monorepo plus extras,
  consistent with the vendored-engine decision. Component boundaries
  are import boundaries and interface contracts, not repository
  boundaries.

## Consequences for documents (this change)

- README gains "The Solver Zoo" section (component matrix with status).
- The roadmap gains the solver-zoo view and the closed-loop v1 order.
- AGENTS.md names the composition architecture and updates its
  "Next:" line; CLAUDE.md and ONBOARDING.md record the environment
  split (macOS planning box, WSL2 test/run box, retained
  Windows-native checkout).
- No safety-scope changes; the canonical exclusion list is untouched.
