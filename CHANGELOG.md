# Changelog

Latent Dirac uses 0.x semantics: minor releases may break APIs without
deprecation shims. Notable changes are recorded here starting from 0.2.0.

## Unreleased (0.2.0)

- Added the `matter_slab` scene element (engine-track M2b): a declarative
  slab of NIST material tracked by the vendored Geant4 Matter adapter
  from within a scene. Physics config (`material`, `thickness_mm`,
  `entry_z_m`, geometry envelope) lives in the scene; the machine-specific
  transformer binary is injected at run time via
  `LATENT_DIRAC_G4_TRANSFORMER` (with optional
  `LATENT_DIRAC_G4_PATH_STYLE`), never stored in the YAML — so scenes
  construct and render with no engine present, and only running the slab
  stage needs the binary. The adapter body is unchanged; `scene_3d` draws
  the slab (fidelity: engine transformer); the JAX backend rejects it.
  Fidelity tier: engine transformer (vanilla Geant4 v11.4.2, FTFP_BERT).
  Design record:
  `docs/superpowers/specs/2026-07-06-matter-slab-scene-element-design.md`.

- The differentiable capture objective now handles `residual_gas_loss`:
  the stochastic hard kill enters the soft objective as its expected
  survival `exp(-hold_time_s / mean_lifetime_s)` — a smooth, uniform,
  differentiable factor (in both parameters). This makes the
  antimatter-native target optimizable: not most captured, but most
  still alive after the hold. The batched simulator continues to reject
  the element (stochastic kill has no static-program form); the
  divergence is intentional and documented at both mirror sites.

- Added the `residual_gas_loss` scene element (storage lifetime): while
  held in a region at finite vacuum, trapped antiparticles annihilate on
  residual gas with per-particle exponential survival
  `exp(-hold_time_s / mean_lifetime_s)`. Killed particles enter the loss
  ledger (seeded, reproducible), survivors age by the hold time. First
  antimatter-native loss physics beyond geometry; fidelity tier
  parameterized (`mean_lifetime_s` is a direct input — the
  cross-section-derived `tau = 1/(n sigma v)` form is a future upgrade
  needing curated sigma(v) data). NumPy pipeline only; the JAX backend
  rejects it. Design record:
  `docs/superpowers/specs/2026-07-06-residual-gas-storage-lifetime-design.md`.

- **The Geant4 Matter adapter is real (engine-track M2).**
  `Geant4MatterAdapter` (`latent_dirac/adapters/geant4/adapter.py`,
  placeholder retired) tracks a `ParticleState` cloud through a
  NIST-material slab via the new `engine/transformer` application
  (vanilla Geant4, FTFP_BERT): energy loss, multiple scattering, and
  antiproton annihilation come back as engine phase space for survivors
  and ledger deaths for absorbed particles, with the provenance
  four-tuple namespaced under `matter` (the source's own provenance is
  preserved). Exchange is subprocess + phase-space CSV files (id-keyed,
  completion-marker guarded; WSL bridge helper for Windows hosts) —
  never in-process. The adapter fails fast on particles outside the
  engine transverse aperture or world envelope. Physically validated
  brackets: 3.6 GeV/c antiprotons through 1 mm Al survive with textbook
  dE/dx and MSC; 100 MeV antiprotons into 50 mm Al all stop and
  annihilate. A `matter_slab` scene element is deferred to M2b.

- Added mean-field space charge (closed-loop v1 item 4, completing
  closed-loop v1): `space_charge: uniform_sphere` on `uniform_field`
  and `penning_trap` elements enables a parameterized uniform-sphere
  self-field (`latent_dirac.fields.space_charge`), refit from the
  alive cloud every transport step — electrostatic only (beta << 1),
  single-sphere fit, no self-consistency; dead particles neither
  source nor feel it; the JAX backend and the differentiable objective
  reject it explicitly. Plus the `cold_uniform_sphere` prepared-cloud
  source (placeholder tier) and the scene-report validity line.
  Physics pinned by tests: interior-linear/exterior-Coulomb field,
  sign-independent self-repulsion, leading-order surface kick,
  no net self-push, trap-suppressed expansion.

- **README restructured Genesis-style.** One-line tagline, numbered
  table of contents, a "What is Latent Dirac?" section led by the new
  four-layer architecture diagram (`assets/architecture.svg`: scene
  interface → solvers → `ParticleState` → compute), merged
  Quick Installation / Using the API sections, new Contributing and
  License-and-Acknowledgments sections. The solver layer is now called
  "Solvers" in the README (the dated solver-zoo spec keeps its name as
  a historical record). Demo sections are byte-identical to before.

- The README no longer mirrors the safety-scope exclusion list or the
  vendored-engine section (owner decision): the canonical list stays
  pinned across `docs/safety_scope.md`, `AGENTS.md`, and
  `tests/test_project_positioning.py`, and the README keeps a
  test-enforced link to `docs/safety_scope.md`. Geant4 attribution
  remains in `NOTICE`.

- The Xsuite adapter is real (`latent_dirac.adapters.xsuite.adapter`,
  new optional `[xsuite]` extra): `ParticleState` ↔ `xtrack.Particles`
  conversion around an explicit `ReferenceFrame` (p0c never silently
  inferred; zeta ↔ per-particle time mapping), and
  `xsuite_tracking_stage` wrapping an `xtrack.Line` as a pipeline
  Stage so xtrack losses stamp the per-particle ledger (xtrack's
  reordering of lost particles is undone by id). The xsuite placeholder
  is deleted; the placeholder gate test became the adapter-status test
  (geant4/root placeholders still enforced). Closed-loop v1 item 3.

- Added ROOT I/O via uproot (`latent_dirac.io.root_io`, new optional
  `[root]` extra; pure Python, no ROOT installation): labeled
  `ParticleState` snapshots as flat SI-unit TTrees plus a JSON
  species/metadata sidecar per label, `write_scene_result` for
  monitors + final cloud, and `read_particle_state` for a full
  write→read round-trip (custom species survive; engine provenance
  travels in the sidecar). Closed-loop v1 item 2.

- Added openPMD particle output (`latent_dirac.io.openpmd_io`, new
  optional `[openpmd]` extra): `write_particle_states` writes labeled
  `ParticleState` snapshots as openPMD iterations (SI unit metadata,
  loss-ledger channel, alive mask, per-particle time as records;
  charge/mass as constants), `write_scene_result` writes every monitor
  plus the final cloud. Engine provenance four-tuples in state metadata
  are lifted to flat species attributes. Write-only; closed-loop v1
  item 1 per the solver-zoo spec.

- **Positioning: solver-zoo composition.** The platform narrative is
  organized as a solver zoo behind one scene/state spine: first-party
  steppers on the NumPy/JAX substrate, engine-backed transformers
  behind adapters (vendored Geant4 now; Xsuite, WarpX, and Garfield++
  as roadmap items), `ParticleState` as the exchange currency, and the
  per-particle ledger spanning solver boundaries. The README gains the
  component matrix; the roadmap gains the zoo view and the closed-loop
  v1 order (openPMD → uproot → Xsuite adapter → mean-field space
  charge). Design record:
  `docs/superpowers/specs/2026-07-05-solver-zoo-composition-design.md`.
  Docs only; no code changes.

- **Engine track first deliverables (M1'-lite + M3-lite).** The vendored
  vanilla Geant4 v11.4.2 tree now builds via the documented WSL/Linux
  recipe (`engine/README.md`: minimal-physics configuration, no
  visualization/UI, datasets fetched at build time), and the first
  first-party engine application ships: `engine/yieldgen` (proton on an
  iridium stand-in target, FTFP_BERT reference physics list, MT) records
  the phase space of antiprotons exiting the target into a versioned CSV
  yield table carrying the provenance four-tuple. A new table-based
  source (`AntiprotonYieldTableSource`, scene type
  `antiproton_yield_table`) replays such tables through the existing
  pipeline; the engine-backed target-production demo scene consumes a
  committed table. Exchange stays offline (no runtime coupling);
  `latent_dirac/adapters/` remains placeholder-only.

- **Positioning: Geant4 engine track.** The complete vanilla Geant4
  v11.4.2 source tree is vendored in-repo at `geant4-v11.4.2/` as a
  read-only engine baseline (Geant4 Software License; attribution in
  `NOTICE`; excluded from lint, tests, and the Python distribution).
  The safety scope was rewritten for the engine era: shower physics and
  energy deposition are delegated to the vendored vanilla engine as
  diagnostics, while weaponization, energetic-release applications,
  facility control, high-yield recipes, activation, and shielding design
  stay excluded; the canonical exclusion list is now pinned verbatim by
  `tests/test_project_positioning.py` across `docs/safety_scope.md`,
  `AGENTS.md`, and the README. Design record:
  `docs/superpowers/specs/2026-07-05-geant4-engine-positioning-design.md`.
  No build or runtime coupling ships yet; adapters remain placeholders.

- Added the Antimatter Factory Chain demos (decay emission with
  beta-spectrum coloring, surrogate target production with drawn-only
  annotations, electrostatic deceleration with time-gated dynamic trap
  capture, and the annihilation endpoint), plus the capabilities behind
  them: `TimeGatedField` and gate parameters on `uniform_field` /
  `penning_trap` scene elements (both backends, parity-tested), and the
  `annihilation_plate` element recording at-rest two-photon kinematics.
  The safety scope was amended minimally: annihilation *energetics*
  remain excluded; annihilation is modeled only as a ledgered loss
  endpoint with kinematic two-photon emission for visualization.

- Added strided trajectory recording to the batched JAX program
  (`record_stride` on `BatchedSceneProgram`/`run_scene_batched`,
  parity-tested against the NumPy pipeline) and the "one launch, 24
  beamlines" batched-sweep README demo.

- Added the ideal Penning trap: `PenningTrapField` (quadrupole
  electrostatic well + axial B, analytic `eigenfrequencies` with the
  invariance relations validated in tests) and the `penning_trap` scene
  element, supported by the NumPy pipeline, the JAX backend, and the
  differentiable objective.

- Added the differentiable capture objective
  (`latent_dirac.backends.differentiable.make_differentiable_objective`):
  sigmoid-relaxed acceptance stages with gradients of the soft accepted
  fraction w.r.t. named scene variables, validated against finite
  differences and converging to the hard `accepted_fraction` as
  sharpness grows.
- Added the `latent-dirac` CLI (`run` prints the scene report, `render`
  writes the interactive 3D HTML) and the hello-beamline scene;
  `scene_report` moved from `examples/` into
  `latent_dirac.diagnostics.scene_report`.
- Added the Xopt-compatible scene evaluator
  (`latent_dirac.backends.evaluator.make_scene_evaluator`): scalar and
  vector-component variables (`"label.param"`, `"label.vec[i]"`),
  Xopt's dict-to-dict calling convention with no xopt dependency, and
  `evaluate.batch` for one-launch candidate generations on the new
  `BatchedSceneProgram` (built once; JAX compiles on first run and
  caches per batch shape).
- Added the JAX batched backend (`latent_dirac.backends.jax_scene`,
  optional `[jax]` extra): a declarative scene compiles into one JAX
  program (`lax.scan` transports, mask-and-ledger acceptance) and `vmap`
  maps it over overridden element parameters
  (`run_scene_batched(scene, overrides={"label.param": values})`).
  The shared Boris kernel is now array-namespace generic (`xp=numpy` or
  `jax.numpy`), and parity with the NumPy float64 pipeline is enforced
  in CI.
- Rebuilt every README demo as a 3D animation rendered from real
  simulation runs through a shared matplotlib pipeline (`tools/mpl3d.py`,
  `tools/generate_scene_demo_webps.py`). Most demos are now defined by
  declarative scenes under `examples/scenes/`; new demos cover the Wien
  velocity filter, a dipole+quadrupole beamline, a field-map-driven
  magnetic mirror bottle, and the antiproton loss ledger. The 2D Pillow
  canvas generator and its four 2D assets are retired.
- **Breaking**: replaced the pydantic `ParticleCloud` with `ParticleState`,
  a pytree-compatible dataclass with the same attribute/method surface plus
  a per-particle loss ledger channel `lost_at_element` (int32, -1 = alive).
  Pipeline stages now stamp the ledger with their stage index;
  `latent_dirac.diagnostics.loss_ledger` reconstructs weighted losses per
  stage name. The Boris pusher is now a pure-function kernel
  (`latent_dirac.solvers.kernels.boris_step`) in dimensionless momentum
  u = p/(m c); SI exists only at State boundaries.
- Added `FieldMapField`: table-based fields on a regular grid with
  trilinear interpolation, a COMSOL regular-grid CSV importer, and 3D
  field-magnitude volume rendering.
- Added the declarative scene schema (YAML/JSON, `schema_version: 1`) with
  fail-fast validation, label-anchored loss accounting, the new `drift`
  and `monitor` elements, and `run_scene` with optional trajectory
  recording. pyyaml is now a core dependency.
- Added scene-driven 3D rendering (`latent_dirac.viz.scene_3d`) with
  per-element fidelity labels in hover text.
- Added the composable field model library: `CompositeField`,
  hard-edge `DipoleField`, and hard-edge `QuadrupoleField`.
- Repositioned the project as an open interactive simulation platform for
  antimatter factories, with an enforced documentation honesty discipline.
- Added the 3D charge-sign splitter README animation rendered from a
  recorded Boris-solver `Trajectory`, plus an interactive Plotly HTML export
  (`tools/generate_hero_3d_webp.py`).
- Added CI (pytest matrix + ruff), a PyPI trusted-publishing workflow,
  CONTRIBUTING.md, and this changelog.

## 0.1.0

- Initial architecture skeleton: SI units and species, `ParticleCloud`
  state, positron pair / beta-plus / surrogate antiproton sources, uniform
  and idealized solenoid fields, relativistic Boris solver, aperture and
  momentum-window acceptance, pipeline loss accounting, accepted-yield
  diagnostics, optional Matplotlib/Plotly backends, placeholder adapters
  for Geant4, Xsuite, and ROOT.
