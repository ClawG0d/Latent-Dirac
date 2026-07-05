# Platform Positioning and Roadmap Design

## Objective

Reposition Latent Dirac from "source-to-acceptance modeling skeleton" to an
**open interactive simulation platform for antimatter factories**, and define
the phased roadmap that gets there. This spec records the positioning
decision, the research behind it, the honesty discipline that keeps the
narrative credible, and the design constraints that later phases must respect.

## Decision Record (2026-07-05)

- Business shape: positrons near-term, antiprotons long-term, end-to-end
  "antimatter factory" narrative.
- Core value: open ecosystem platform plus technical storytelling. Target
  users: the company's own engineering/physics team, academic research
  groups, investors and the public.
- Positioning option chosen: **platform narrative first** — interactive 3D
  and visual impact lead, physics fidelity is layered in explicitly
  tier-by-tier.
- GPU route: **JAX** (vmap for batched sweeps, autodiff for future
  differentiable design), with a NumPy float64 reference backend kept for
  correctness comparisons. Kernels stay functionally pure so other backends
  can be added later.
- Sequencing: architecture before GPU.

## Research Summary

Six research passes covered Genesis, NVIDIA Newton/Warp/Isaac Sim,
ANSYS/COMSOL/CST/SIMION, CAD and geometry pipelines, and the
antimatter/accelerator simulation landscape. Conclusions that shaped this
spec:

1. The existing ecosystem is split into five non-interoperating segments
   (Geant4 family, MAD-X/xsuite, PIC codes, SIMION/COMSOL, Molflow+). No
   open Python end-to-end pipeline exists for trap-based low-energy
   antimatter facilities. The closest existing tool is RF-Track (CERN),
   which has no trap physics, no batched GPU sweeps, and no per-particle
   loss ledger.
2. The architecture recipe has independently converged across Genesis,
   Newton, and xsuite: headless core with optional viewers; a static/dynamic
   split of the data model (Model/State/Control); pluggable solver
   interfaces; single-source kernels compiled to multiple backends;
   declarative serializable scene descriptions; SoA particle pools with
   alive masks.
3. GPU value comes from batched throughput, not single-particle speed.
   Charged-particle beams are a natural large-batch independent-trajectory
   workload. Performance storytelling must stay honest: publish settings,
   never imply single-trajectory speedups.
4. Do not rewrite upstream physics. Field-map import is the industry's de
   facto exchange interface with commercial FEM tools; Geant4 stays behind
   an adapter; openPMD is the ticket into academic analysis workflows; CAD
   import is deferred because parameterized electrode/beamline primitives
   cover the near-term need.
5. Hard physics constraints for later phases: the trap timescale gap
   (~10 ps cyclotron period vs seconds of storage) requires a
   guiding-center/secular solver in addition to the Boris pusher; Geant4 is
   unreliable below ~1 keV so moderation needs semi-empirical
   parameterizations (Makhov profiles); differentiable beam dynamics has
   prior art (Cheetah), so the differentiable niche is trap capture and the
   low-energy chain; optimization should integrate with Xopt instead of
   reinventing optimizers; there is no machine-readable open library of
   positron cross sections and moderator yields — curating one is a moat.
6. Business reality: academic users do not pay for licenses. The viable
   path is a permissive open core plus design studies, consulting, and
   (later) hosted services. The north-star metric is adoption as the design
   tool of one or two flagship or under-construction facilities.

## Positioning Statement

Latent Dirac is an open interactive simulation platform for antimatter
factories: declarative scene descriptions of positron/antiproton facilities
(source → transport → capture), batch-parallel simulation, sweeps and
optimization, and interactive 3D visualization. The slogan: **turn
antimatter facility design iteration from a wall-clock problem into a
compute problem.**

Three narrative pillars:

1. **Platform, not just a tracker**: Scene/Entity declarative description,
   pluggable solvers, optional viewers.
2. **Throughput**: JAX batched sweeps (n_configs × n_particles in one
   launch).
3. **Ledger**: loss accounting as a full life-cycle ledger for every
   antiparticle — antiparticles are extraordinarily expensive, and no
   existing tool offers this as a first-class concept.

## Honesty Discipline

The reference cautionary tale is a robotics simulator whose headline
throughput number collapsed by two orders of magnitude once benchmark
settings were normalized. The antimatter community is smaller and more
expert; inflated claims would be fatal. Rules:

- The README separates design intent from current status ("architected
  for X / currently implements Y"), keeping the Implemented / Not
  implemented yet lists.
- Every performance number ships with reproducible settings: integrator,
  timestep, particle count, batch size, physics approximation tier, and
  hardware. Benchmark code is open and runs in CI.
- Every solver and source model declares its fidelity tier: placeholder,
  parameterized, surrogate, table-based, or externally calibrated.
  Fidelity labels will also be rendered into 3D scenes.
- `tests/test_project_positioning.py` enforces this mechanically: current
  status sections, fidelity tier declarations, verbatim safety-scope
  exclusions in the README, and no comparative performance wording without
  a benchmark reference.

## Safety Boundary

All exclusions in `docs/safety_scope.md` remain unchanged. The digital-twin
direction adds one explicit boundary: offline forward simulation, replay of
measured data, and historical parameter calibration only — no real-time
control loops and no interfaces that write back to a facility.

## JAX Numerical and Platform Constraints (locked before Phase 2c)

- **Dimensionless kernel units.** float32 with SI internal units underflows:
  a 3 MeV positron has p² ≈ 3.5e-42 (kg·m/s)², below the float32 smallest
  normal 1.18e-38, and eV-scale trap particles flush to zero. Kernels use
  dimensionless variables (p/mc, t·ω_c, x·ω_c/c); SI exists only at State
  boundaries.
- **Container layering rule.** pydantic owns Model/scene schema (static
  configuration, fail-fast validation). State uses a pytree-compatible
  container (dataclass/NamedTuple registered with
  `jax.tree_util.register_pytree_node`). The current pydantic
  `ParticleCloud` cannot be reused as the State container.
- **Validation mode.** `jax_enable_x64=True` plus the NumPy float64
  reference backend, compared element-wise in CI (runs on CPU).
- **Hardware reality.** Local macOS development uses CPU JAX for
  correctness; performance numbers come only from Linux CUDA hardware and
  are labeled as such. jax-metal is not a dependency.
- **Environment check (verified 2026-07-05):** jaxlib 0.10.2 publishes
  CPython 3.14 wheels including `macosx_11_0_arm64`, so the current venv
  (Python 3.14.3) is supported for CPU development.
- **RNG strategy.** Source sampling stays on NumPy (CPU); only transport
  kernels move to JAX, avoiding a wholesale migration to the `jax.random`
  key system.

## Roadmap

### Phase 2 — Architecture foundation with visuals first

Split into independently deliverable specs:

- **2-0 Open-source hygiene** (first): GitHub Actions pytest matrix
  (3.10–3.14, macOS + Linux), ruff, CONTRIBUTING.md with a fidelity
  declaration duty for new physics, README CI badge, PyPI 0.1.0 release via
  trusted publishing (the `pip install "latent-dirac[viz]"` command in
  docs/rendering.md must become real), 0.x versioning semantics, CHANGELOG
  from 0.2.0, docs site choice (mkdocs-material).
- **2a Field model library** (spec exists:
  `2026-07-04-field-model-library-design.md`): CompositeField, DipoleField,
  QuadrupoleField.
- **2b Scene schema + minimal 3D** (visual flagship, may run parallel to
  2c): declarative serializable scene (JSON/YAML with a top-level
  `schema_version`, fail-fast validation, named element labels). Acceptance
  criterion: every schema element type has a 3D representation; one command
  goes from a YAML scene to a plotly 3D beamline with tracked trajectories.
  Design constraint to pre-bake: every numeric parameter must be liftable
  into a batch-dimension array (the Phase 3 vmap prerequisite). Includes the
  60-second hello-beamline.
- **2c State/Model/Control refactor** (pure refactor, no new physics):
  pytree State container (SoA, alive mask, and a per-particle loss channel
  `lost_at_element: int32` with -1 meaning alive — the ledger's data model
  under static shapes); the Boris pusher rewritten as a pure function
  `boris_step(state, field_params, dt)` still on NumPy; a unified SolverBase
  with a Coupler pipeline slot; dimensionless unit boundaries. Migration is
  a one-time break (0.x semantics, no deprecation shims) across the 4
  examples, `tools/generate_demo_webp.py`, the dependent tests, and
  regenerated README animations.
- **2d FieldMap** (separate spec; explicitly revises the non-goals of the
  2026-07-04 field spec): regular-grid field container with trilinear
  interpolation, starting with exactly one import format (COMSOL
  regular-grid CSV); CST and SIMION formats come later.
- **2e openPMD**: deferred until Phase 3 wrap-up / JOSS submission.

### Phase 3 — GPU batch and the interactive platform (3–6 months)

- JAX backend: pure-function kernels, vmap over particles × configs,
  `lax.scan` over time; NumPy float64 reference comparisons in CI.
- Batched sweep API (`scene.build(n_configs=B)` semantics) and an
  Xopt-compatible evaluator as an optional `[opt]` extra.
- Interactive 3D viewer: plotly 3D first, then a web viewer; USD export kept
  open for later.
- Flagship demo: thousands of configurations in a 3D beam-envelope
  animation. The spec must first design trajectory downsampling/streaming —
  full (T, N, 3) storage at that scale does not fit in memory.
- Honest benchmark suite: analytic cases (Larmor radius, magnetic mirror,
  E×B drift, relativistic cyclotron frequency) in CI. External comparisons
  (RF-Track, published experimental data) only after verifying license and
  data availability/redistribution rights.

### Phase 4 — Digital twin and physics fill-in (6–12 months)

- Differentiable chain via JAX autodiff: d(capture efficiency)/d(field
  parameters) with sigmoid-relaxed soft apertures; a "differentiable
  antimatter facility design" demo focused on trap capture and the
  low-energy chain.
- Trap physics step one: a Penning-Malmberg trap element (axial
  electrostatic well plus uniform B) and Surko buffer-gas Monte Carlo
  collisions (N2/CF4 cross sections curated as a standalone open dataset
  with provenance).
- Timescale gap: a guiding-center/secular long-timescale solver, because a
  Boris pusher cannot reach seconds of storage.
- Geant4 adapter made real (high-energy ground truth; first interface is
  implantation/moderation yield tables) plus semi-empirical moderation
  parameterizations (Makhov profiles).
- Digital twin: offline replay of measured data and historical parameter
  calibration only; domain randomization for uncertainty quantification.

### Ecosystem and community (continuous)

- Apache-2.0, mkdocs-material docs site, JOSS/PRAB paper for citations.
- North star: adoption by one or two flagship or under-construction
  facilities, not GitHub stars.
- Governance: company-led at first with a neutral-home path reserved;
  CLA/IP design before accepting institutional contributions.
- Commercialization stays long-term (design studies, hosted GUI); the space
  propulsion/fusion narrative is brand material, not a revenue assumption.

## Non-Goals

This positioning phase does not implement the scene schema, the JAX backend,
trap physics, or any adapter. It changes documentation, tests, and README
assets only. It does not modify `docs/safety_scope.md` exclusions.

## Acceptance Criteria

- `tests/test_project_positioning.py` enforces the honesty discipline and
  passes.
- README, AGENTS.md, docs/roadmap.md, docs/architecture.md,
  docs/physics_scope.md, and docs/safety_scope.md reflect this spec with the
  design-intent/current-status separation.
- A 3D trajectory README asset generated from a real simulation exists.
- The full test suite passes.
