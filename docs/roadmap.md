# Roadmap

The positioning behind this roadmap is recorded in
[the platform positioning spec](superpowers/specs/2026-07-05-platform-positioning-and-roadmap-design.md).

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
- **2e openPMD output**: deferred until Phase 3 wrap-up.

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
- flagship batched-sweep 3D demo with trajectory downsampling/streaming
- honest benchmark suite: analytic cases in CI; external comparisons only
  after license and data-availability checks

## Phase 4 — digital twin and physics fill-in

- differentiable capture chain via autodiff with soft-aperture relaxation
  — **done early** (Spec 3d: `make_differentiable_objective`, gradients
  validated against finite differences; field-map objectives still pending)
- Penning-Malmberg trap element — **field model done** (Spec 4a: ideal
  quadrupole well + axial B with validated eigenfrequencies; electrode
  geometries stay on the field-map route); Surko buffer-gas Monte Carlo
  collisions with cross-section data curated as an open,
  provenance-tracked dataset — pending
- guiding-center/secular solver for long-timescale trap storage
- Geant4 adapter made real (implantation/moderation yield tables) plus
  semi-empirical moderation parameterizations
- offline digital twin: replay of measured data and historical calibration
  only; domain randomization for uncertainty quantification

## Continuous — ecosystem and community

- mkdocs-material documentation site, JOSS/PRAB paper
- adoption by flagship or under-construction facilities as the north star
- governance: neutral-home path reserved; CLA/IP design before accepting
  institutional contributions
