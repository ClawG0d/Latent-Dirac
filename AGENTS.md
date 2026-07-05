# AGENTS.md

## Project

Latent Dirac is an open interactive simulation platform for antimatter
factories: declarative scenes of positron and antiproton facilities
(source -> transport -> capture), batch-parallel simulation and sweeps, and
interactive 3D visualization. The platform architecture is a solver
composition ("solver zoo"): first-party solvers on the NumPy/JAX
substrate plus engine-backed solvers behind adapters, exchanging
`ParticleState` at stage boundaries (see
docs/superpowers/specs/2026-07-05-solver-zoo-composition-design.md).
The current codebase is the lightweight
source-to-acceptance core; the platform capabilities are phased in through
docs/roadmap.md and the positioning spec
docs/superpowers/specs/2026-07-05-platform-positioning-and-roadmap-design.md.

## Current phase

Phase 2: architecture foundation with visuals first. Spec status:

- 2-0 open-source hygiene: done (CI, ruff, CONTRIBUTING, CHANGELOG,
  publish workflow; PyPI release pending owner-side trusted publishing)
- 2a field model library: done
- 2b declarative scene schema plus minimal 3D rendering: done
  (the deferred CLI and hello-beamline landed as Spec 3c)
- 2c State/Model/Control refactor: done (`ParticleState` pytree dataclass,
  pure Boris kernel in dimensionless momentum, per-particle loss ledger)
- 2d FieldMap import (COMSOL regular-grid CSV): done
- 2e openPMD output: scheduled as the first closed-loop v1 item

Next: closed-loop v1 (openPMD output, ROOT I/O via uproot, the Xsuite
adapter, native mean-field space charge) with engine-track M1' build
recipes in parallel; then the GPU lane and the interactive viewer. See
docs/roadmap.md.

Geant4 engine track: the complete vanilla Geant4 v11.4.2 source tree is
vendored at `geant4-v11.4.2/` as the in-repo engine baseline. The first
build recipe (`engine/README.md`) and the first yield table
(`engine/yieldgen` feeding the `antiproton_yield_table` source) have
landed; exchange stays offline. The real adapter (GDML + subprocess)
and the companion acceleration library are phased in via
docs/roadmap.md; `latent_dirac/adapters/` stays placeholder-only until
that work lands.

Vendored Geant4 tree rules:

- `geant4-v11.4.2/` is a read-only vanilla baseline. Never edit files
  inside it, for any reason. Future modifications go exclusively through
  the patch protocol defined in the engine positioning spec (small,
  documented patches, each with regression validation against the vanilla
  baseline); until that protocol is established, the tree is frozen.
- Never run lint, formatters, pytest, or bulk searches over the vendored
  tree. The ruff `extend-exclude`, pytest `testpaths`, and packaging
  `MANIFEST.in prune` exclusions must be preserved.
- Commits touching the vendored tree use the `vendor:` prefix and must
  keep it byte-identical to an upstream release (`.gitattributes -text`;
  one documented exception: the upstream `CHANGELOG` symlink is stored
  as a regular file — see the engine positioning spec addendum, and do
  not restore the symlink).

Do not integrate FLUKA, MAD-X, or RF-Track. Xsuite and ROOT I/O (via
uproot) are scheduled closed-loop v1 items; until those changes land,
keep adapter interfaces and placeholders only for Geant4, Xsuite, and
ROOT.

## Core physics scope

Latent Dirac currently focuses on:
- positron source term models
- antiproton surrogate source term models
- relativistic motion of charged particles in electromagnetic fields
- beamline acceptance
- loss accounting
- accepted yield diagnostics
- particle-matter interaction (showers, stopping, energy-deposition
  diagnostics) via the vendored vanilla Geant4 engine, once the adapter
  becomes real per the roadmap
- future calibration against external scientific tools such as Xsuite

Trap physics: the ideal Penning trap field model landed early (Spec 4a,
validated eigenfrequencies). Buffer-gas collisions, rotating wall, space
charge, and guiding-center long-timescale transport remain Phase 4
roadmap directions.

## Out of scope

Do not implement:
- weaponization scenarios
- energetic-release applications (antimatter as an energy source or destructive payload in any form)
- real facility control systems
- detailed accelerator target engineering (thermal, mechanical, and materials design of production targets)
- high-yield operational recipes
- in-house shower physics (electromagnetic and hadronic showers are delegated to the vendored vanilla Geant4 engine; the Python core does not implement them)
- annihilation energetics as a figure of merit (energy deposition is in scope only as an engine-computed diagnostic; the Python core models annihilation only as a loss endpoint with kinematic two-photon emission for visualization)
- material activation
- radiation shielding design
- any real-time control loop or interface that writes back to a facility
  (the digital-twin direction is offline replay and calibration only)

## Physics rules

- Use SI units internally in the NumPy core.
- Future JAX kernels use dimensionless internal units (p/mc, t*omega_c,
  x*omega_c/c); SI exists only at State boundaries. float32 with SI units
  underflows for MeV-scale momenta, so this rule is load-bearing.
- Antiparticles have normal positive mass.
- Positron has the same mass as electron and opposite charge.
- Antiproton has the same mass as proton and opposite charge.
- Relativistic motion must use momentum, gamma, velocity, and Lorentz force.
- Validate motion against analytic cases in uniform fields.
- Source models must clearly state their fidelity tier: placeholder,
  parameterized, surrogate, table-based, or externally calibrated.

## Engineering rules

- Use Python, numpy, pydantic, and pytest.
- Container layering: pydantic owns Model and scene schema (static
  configuration, fail-fast validation). Simulation State containers must be
  pytree-compatible (dataclass or NamedTuple); do not put pydantic models on
  hot state paths.
- Write physics kernels as pure functions so the same kernel can run on the
  NumPy reference backend and future JAX backends.
- Keep the pip package lightweight; optional capabilities go behind
  extras. The vendored Geant4 tree lives in the repository but is not
  part of the Python distribution (wheel/sdist exclusions must be
  preserved).
- Prefer clear, testable modules over clever abstractions.
- Every physical assumption must be explicit.
- Every new physics feature must include tests and a fidelity tier.
- Honesty discipline: no comparative performance wording without an open,
  reproducible benchmark; performance numbers carry integrator, timestep,
  particle count, batch size, approximation tier, and hardware.
- Use Apache-2.0 license for first-party code; the vendored Geant4 tree
  keeps its own Geant4 Software License, and the NOTICE attribution to
  the Geant4 Collaboration must stay intact.
- Geant4 naming: describe the engine as vanilla Geant4 v11.4.2 (plus a
  published patch list, once the patch protocol exists); never use the
  Geant4 name for endorsement or promotion — that requires written
  permission from the Geant4 Collaboration.
- Keep optional external integrations behind adapters.
