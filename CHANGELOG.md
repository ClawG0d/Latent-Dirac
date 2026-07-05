# Changelog

Latent Dirac uses 0.x semantics: minor releases may break APIs without
deprecation shims. Notable changes are recorded here starting from 0.2.0.

## Unreleased (0.2.0)

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
