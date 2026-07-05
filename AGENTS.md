# AGENTS.md

## Project

Latent Dirac is an open interactive simulation platform for antimatter
factories: declarative scenes of positron and antiproton facilities
(source -> transport -> capture), batch-parallel simulation and sweeps, and
interactive 3D visualization. The current codebase is the lightweight
source-to-acceptance core; the platform capabilities are phased in through
docs/roadmap.md and the positioning spec
docs/superpowers/specs/2026-07-05-platform-positioning-and-roadmap-design.md.

## Current phase

Phase 2: architecture foundation with visuals first. Work is split into
independently deliverable specs:

- 2-0 open-source hygiene: CI, ruff, CONTRIBUTING, PyPI release, CHANGELOG
- 2a field model library (spec exists: 2026-07-04-field-model-library-design.md)
- 2b declarative scene schema plus minimal 3D rendering (visual flagship)
- 2c State/Model/Control refactor (pure refactor, no new physics)
- 2d FieldMap import (COMSOL regular-grid CSV first)
- 2e openPMD output (deferred until the JOSS submission window)

Do not integrate Geant4, Xsuite, ROOT, FLUKA, MAD-X, or RF-Track yet.
Keep adapter interfaces and placeholders only for Geant4, Xsuite, and ROOT.

## Core physics scope

Latent Dirac currently focuses on:
- positron source term models
- antiproton surrogate source term models
- relativistic motion of charged particles in electromagnetic fields
- beamline acceptance
- loss accounting
- accepted yield diagnostics
- future calibration against external scientific tools such as Geant4 and Xsuite

Trap physics (Penning-Malmberg elements, buffer-gas collisions,
guiding-center long-timescale transport) is a Phase 4 roadmap direction, not
part of the current phase.

## Out of scope

Do not implement:
- weaponization scenarios
- energetic-release applications
- real facility control systems
- detailed accelerator target engineering
- high-yield operational recipes
- full electromagnetic shower physics
- full hadronic shower physics
- annihilation physics
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
- Keep the core package lightweight; optional capabilities go behind extras.
- Prefer clear, testable modules over clever abstractions.
- Every physical assumption must be explicit.
- Every new physics feature must include tests and a fidelity tier.
- Honesty discipline: no comparative performance wording without an open,
  reproducible benchmark; performance numbers carry integrator, timestep,
  particle count, batch size, approximation tier, and hardware.
- Use Apache-2.0 license.
- Keep optional external integrations behind adapters.
