# AGENTS.md

## Project

Latent Dirac is an open modular simulation platform for positron and antiproton source-to-acceptance modeling.

## Current phase

Implement only the first architecture skeleton and a minimal working simulation demo.

Do not integrate Geant4, Xsuite, ROOT, FLUKA, MAD-X, or RF-Track yet.
Create adapter interfaces and placeholders only for Geant4, Xsuite, and ROOT.

## Core physics scope

Latent Dirac currently focuses on:
- positron source term models
- antiproton surrogate source term models
- relativistic motion of charged particles in electromagnetic fields
- beamline acceptance
- loss accounting
- accepted yield diagnostics
- future calibration against external scientific tools such as Geant4 and Xsuite

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

## Physics rules

- Use SI units internally.
- Antiparticles have normal positive mass.
- Positron has the same mass as electron and opposite charge.
- Antiproton has the same mass as proton and opposite charge.
- Relativistic motion must use momentum, gamma, velocity, and Lorentz force.
- Validate motion against analytic cases in uniform fields.
- Source models must clearly state whether they are placeholder, parameterized, surrogate, table-based, or externally calibrated.

## Engineering rules

- Use Python, numpy, pydantic, and pytest.
- Keep the core package lightweight.
- Prefer clear, testable modules over clever abstractions.
- Every physical assumption must be explicit.
- Every new physics feature must include tests.
- Use Apache-2.0 license.
- Keep optional external integrations behind adapters.
