# Architecture

## Current architecture

Latent Dirac is organized around a universal `ParticleCloud` state. Source
terms create clouds, solvers transport clouds through electromagnetic fields,
beamline elements update the `alive` acceptance mask, and diagnostics
summarize accepted yield and spectra.

The current skeleton is intentionally lightweight:
- `core`: constants, unit conversion, species, and random helpers
- `state`: particle cloud and trajectory containers
- `sources`: parameterized, simplified, and surrogate source models
- `fields`: uniform and idealized solenoid field models
- `solvers`: relativistic Boris transport
- `beamline`: aperture and momentum-window acceptance
- `pipeline`: staged execution and loss accounting
- `scene`: declarative YAML/JSON scene schema, loader, and pipeline builder
- `diagnostics`: accepted-yield and text-report utilities
- `adapters`: placeholders for future optional Geant4, Xsuite, and ROOT integrations

External scientific ecosystems are not integrated in this phase.

## Target architecture (Phase 2c and beyond)

The platform data model separates three concerns, following the pattern that
has independently converged in modern simulation engines:

- **Model** — the static scene: lattice, fields, geometry, apertures.
  Described declaratively (JSON/YAML with `schema_version`), validated
  fail-fast by pydantic, with a named label per element.
- **State** — the dynamic simulation data: SoA particle arrays, alive mask,
  and a per-particle `lost_at_element` loss channel that keeps the loss
  ledger under static array shapes. State containers are pytree-compatible
  (dataclass/NamedTuple), never pydantic.
- **Control** — time-varying inputs: electrode voltages, magnet currents,
  RF parameters.

Solvers implement a unified `SolverBase` interface and are orchestrated by a
Coupler pipeline. Physics kernels are pure functions, so the NumPy reference
backend and future JAX backends share one implementation. Kernels use
dimensionless internal units; SI appears only at State boundaries.

Visualization stays an optional layer: scene descriptions drive both the
physics and the 3D viewers, and fidelity tier labels are rendered into the
scene views.
