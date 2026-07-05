# Changelog

Latent Dirac uses 0.x semantics: minor releases may break APIs without
deprecation shims. Notable changes are recorded here starting from 0.2.0.

## Unreleased (0.2.0)

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
