# Roadmap

The positioning behind this roadmap is recorded in
[the platform positioning spec](https://github.com/ClawG0d/Latent-Dirac/blob/master/docs/superpowers/specs/2026-07-05-platform-positioning-and-roadmap-design.md).

## Solver-zoo view (adopted 2026-07-05)

The platform composes solver components behind one scene/state spine;
see [the solver-zoo spec](https://github.com/ClawG0d/Latent-Dirac/blob/master/docs/superpowers/specs/2026-07-05-solver-zoo-composition-design.md).
Component → next milestone:

- Source → first engine yield table shipped (`engine/yieldgen`);
  positron/moderation tables per M3
- Transport → shipped (NumPy float64 reference + JAX batch)
- Lattice → adapter shipped (conversion + tracking Stage) and the
  `xsuite_lattice` scene element shipped (declarative `xtrack.Line` by
  path; needs the `[xsuite]` extra)
- Matter → adapter shipped (M2) and the `matter_slab` scene element
  shipped (M2b: declarative slabs; transformer binary injected via
  LATENT_DIRAC_G4_TRANSFORMER); GDML translation of full scenes later
- Collective → mean-field v1 shipped (uniform-sphere, parameterized);
  WarpX adapter later
- Detector → parameterized model first, Garfield++ later
- Analysis → shipped: openPMD write (2e, `[openpmd]` extra) and ROOT
  round-trip via uproot (`[root]` extra)

### Closed-loop v1 (in order)

openPMD output — **done** (write-only particle output,
`latent_dirac.io.openpmd_io`) → ROOT I/O via uproot — **done**
(`latent_dirac.io.root_io`: SI-unit TTrees + JSON sidecar, write→read
round-trip) → Xsuite adapter — **done**
(`latent_dirac.adapters.xsuite.adapter`: `ParticleState` ↔
`xtrack.Particles` with an explicit reference frame, tracking Stage
with ledger stamping; the placeholder gate flipped into the
adapter-status test) → native mean-field space charge — **done**
(`space_charge: uniform_sphere` on uniform-field and Penning-trap
elements; parameterized tier, beta << 1, per-step refit, NumPy pipeline
only). Closed-loop v1 is complete. Engine-track M1' proceeds in
parallel; the GPU lane (float32 backend validation, then the honest
benchmark suite published to docs/benchmarks.md) is next — the
session-level execution plan through M3/M4 into Phase 4 is recorded in
[the 2026-07-06 execution-plan spec](https://github.com/ClawG0d/Latent-Dirac/blob/master/docs/superpowers/specs/2026-07-06-execution-plan-gpu-to-phase4-design.md).
The interactive viewer's first slice (the Plotly animated viewer)
shipped.

## Phase 1 (done)

Architecture skeleton and minimal working demos: core constants and species,
particle cloud state, parameterized positron and surrogate antiproton
sources, uniform and idealized solenoid fields, relativistic Boris solver,
aperture and momentum-window acceptance, pipeline loss accounting,
accepted-yield diagnostics, placeholder adapters.

## Phase 2 — architecture foundation, visuals first

Split into independently deliverable specs:

- **2-0 open-source hygiene**: CI matrix, ruff, CONTRIBUTING with fidelity
  declaration duty, PyPI release and 0.x versioning, CHANGELOG, docs site.
- **2a field model library**: CompositeField, DipoleField, QuadrupoleField.
- **2b scene schema + minimal 3D** (visual flagship): declarative
  serializable scenes with `schema_version` and named element labels; every
  element type has a 3D representation; one command from a YAML scene to a
  plotly 3D beamline with trajectories; numeric parameters designed to be
  liftable into batch-dimension arrays.
- **2c State/Model/Control refactor**: pytree State container with an alive
  mask and a per-particle `lost_at_element` loss channel; the Boris pusher
  as a pure function on the NumPy backend; unified SolverBase with a Coupler
  slot; dimensionless unit boundaries.
- **2d FieldMap import**: regular-grid field container with trilinear
  interpolation, COMSOL regular-grid CSV first. RF fields are a further
  field-library extension after field maps.
- **2e openPMD output**: done (write-only particle output behind the
  `[openpmd]` extra; see the solver-zoo view above).

## Phase 3 — GPU batch and the interactive platform

- JAX backend (vmap over configurations, `lax.scan` in time) with NumPy
  float64 reference comparisons in CI — **done** (Spec 3a:
  `latent_dirac.backends.jax_scene.run_scene_batched`, optional `[jax]`
  extra; field maps and batched monitor snapshots still pending)
- `latent-dirac` CLI and hello-beamline — **done** (Spec 3c: `run` and
  `render` subcommands, console script, hello scene)
- Xopt-compatible sweep evaluator — **done** (Spec 3b:
  `latent_dirac.backends.evaluator.make_scene_evaluator`; plain-callable
  convention, no xopt dependency; `evaluate.batch` runs a generation of
  candidates in one launch via the precompiled `BatchedSceneProgram`)
- interactive 3D viewer (plotly first, then web); USD export kept open
- flagship batched-sweep 3D demo — **done** (Spec 3e: `record_stride`
  strided trajectory recording in the batched program; streaming for
  extreme scales remains a later design)
- honest benchmark suite: analytic cases in CI; external comparisons only
  after license and data-availability checks

## Factory-chain demos — done

Decay emission, surrogate target production, electrostatic deceleration
with time-gated dynamic capture (`TimeGatedField`), and the annihilation
endpoint (at-rest two-photon kinematics, no energetics). RF deceleration
and real target physics remain below.

## Geant4 engine track

Positioning and rules in
[the engine positioning spec](https://github.com/ClawG0d/Latent-Dirac/blob/master/docs/superpowers/specs/2026-07-05-geant4-engine-positioning-design.md):
vanilla Geant4 v11.4.2 vendored in-repo as a read-only baseline, a
companion acceleration library via the fast-simulation hooks, and a
controlled patch protocol (frozen until its infrastructure exists).

- **M0 — vendored baseline and positioning** — done (this spec: vendor
  commit, tooling/licensing compliance, agent-doc rules, safety-scope
  rewrite with the canonical exclusion list pinned by tests)
- **M1' — engine build recipes** — first recipe done (`engine/README.md`:
  WSL/Linux minimal-physics build, no visualization/UI, datasets fetched
  at build time); containerized CI outside the Python matrix still
  pending
- **M2 — adapter made real** — done (the Matter adapter:
  `Geant4MatterAdapter` drives `engine/transformer` over the phase-space
  file contract; FTFP_BERT slab transform with annihilation into the
  loss ledger; physically validated thin-foil/stopping-block brackets —
  see the 2026-07-05 matter-adapter spec). The `matter_slab` scene
  element (M2b) is done — declarative slabs in a scene, the transformer
  binary injected at run time via LATENT_DIRAC_G4_TRANSFORMER (never in
  the YAML); GDML translation of full scenes remains a later extension
- **M3 — yield-table pipeline** — first deliverable done
  (`engine/yieldgen`: proton-on-iridium FTFP_BERT production, CSV
  contract with the provenance four-tuple, consumed by the
  `antiproton_yield_table` table-based source and the engine-backed
  target demo); positron/moderation tables and the surrogate source's
  graduation toward `externally calibrated` still pending
- **M4 — companion acceleration library**: first-party C++ in `engine/`
  attached through `G4VFastSimulationModel`; EM domain first;
  performance claims only against open vanilla-Geant4 benchmarks

## Phase 4 — digital twin and physics fill-in

- differentiable capture chain via autodiff with soft-aperture relaxation
  — **done early** (Spec 3d: `make_differentiable_objective`, gradients
  validated against finite differences; field-map objectives still pending)
- Penning-Malmberg trap element — **field model done** (Spec 4a: ideal
  quadrupole well + axial B with validated eigenfrequencies; electrode
  geometries stay on the field-map route); Surko buffer-gas Monte Carlo
  collisions — **parameterized + table-based operators done** (the
  `buffer_gas_cooling` element: a constant-rate parameterized mode, and a
  table-based mode driven by the null-collision (Skullerud) operator in
  `latent_dirac/collisions/` over a provenance-checked, energy-dependent
  cross-section table — elastic/inelastic-threshold/loss channels at gas
  density n = P/(k_B T), with a cross-section provenance block in the
  report). What remains is a **real, DOI-cited positron cross-section
  dataset** (the shipped N2 table is a clearly-labeled synthetic
  placeholder, so the element stays `parameterized` until a curated table
  lands) and operator-splitting with the trap field — see the 2026-07-06
  buffer-gas collisions and table-based landing specs
- storage lifetime — **parameterized model done** (the `residual_gas_loss`
  element: stochastic annihilation on residual gas over a hold time,
  `mean_lifetime_s` a direct input, ledgered per particle; the
  differentiable objective consumes it as the expected-survival factor
  exp(-hold/tau), so capture and storage survival optimize jointly; the
  cross-section-derived tau = 1/(n sigma v) form needs the same curated
  sigma(v) dataset as the buffer-gas work above — pending)
- guiding-center/secular solver for long-timescale trap storage
- moderation physics: implantation/moderation yield tables via the
  engine track (M3) plus semi-empirical moderation parameterizations
- offline digital twin: replay of measured data and historical calibration
  only; domain randomization for uncertainty quantification

## Continuous — ecosystem and community

- mkdocs-material documentation site, JOSS/PRAB paper
- adoption by flagship or under-construction facilities as the north star
- governance: neutral-home path reserved; CLA/IP design before accepting
  institutional contributions
