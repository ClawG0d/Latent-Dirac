# Architecture

## Current architecture

Latent Dirac is organized around a universal `ParticleState`. Source terms
create states, solvers transport them through electromagnetic fields,
beamline elements update the `alive` acceptance mask, the pipeline stamps
the per-particle loss ledger, and diagnostics summarize accepted yield,
losses by stage, and spectra.

The current package is intentionally lightweight:
- `core`: constants, unit conversion, species, and random helpers
- `state`: `ParticleState` (pytree-compatible dataclass: SoA arrays, alive
  mask, per-particle `lost_at_element` ledger channel) and trajectory
  containers
- `sources`: parameterized, simplified, and surrogate source models
- `fields`: uniform, solenoid, dipole, quadrupole, composite, and
  table-based field-map models
- `solvers`: relativistic Boris transport as a pure-function kernel in
  dimensionless momentum u = p/(m c), with SI only at State boundaries
- `beamline`: aperture and momentum-window acceptance
- `pipeline`: staged execution, loss accounting, and ledger stamping
  (stages stamp `lost_at_element` with their index)
- `scene`: declarative YAML/JSON scene schema, loader, and pipeline builder
- `backends`: optional JAX batched execution (`run_scene_batched`: one
  compiled program, `vmap` over configurations)
- `diagnostics`: accepted-yield, loss-ledger, and text-report utilities
- `adapters`: placeholders for future optional Geant4, Xsuite, and ROOT integrations

External scientific ecosystems are not integrated in this phase.

## Model / State / Control

The platform data model separates three concerns, following the pattern that
has independently converged in modern simulation engines:

- **Model** — the static scene: lattice, fields, geometry, apertures.
  Described declaratively (JSON/YAML with `schema_version`), validated
  fail-fast by pydantic, with a named label per element. Solver and source
  configuration classes are Model-layer objects and stay pydantic.
- **State** — the dynamic simulation data: `ParticleState`, a
  pytree-compatible dataclass (never pydantic) holding SoA particle arrays,
  the alive mask, and the `lost_at_element` ledger channel that keeps loss
  accounting under static array shapes. Particles are stamped, not deleted.
- **Control** — time-varying inputs (electrode voltages, magnet currents,
  RF parameters): reserved for a later phase alongside the Coupler slot in
  the solver layer.

Physics kernels are pure functions (full-array `where`-masking, no
data-dependent control flow), so the NumPy float64 reference backend and
the planned Phase 3 JAX backend share one implementation. Kernels use
dimensionless internal units; SI appears only at State boundaries.

Visualization stays an optional layer: scene descriptions drive both the
physics and the 3D viewers, and fidelity tier labels are rendered into the
scene views.
