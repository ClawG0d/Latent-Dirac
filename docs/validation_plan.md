# Validation Plan

Initial validation focuses on analytic and accounting checks:
- species mass and charge relationships for particles and antiparticles
- eV/J and GeV/c/SI unit conversion round trips
- `ParticleState` shape validation, weighted counts, and ledger defaults
- source model output species, shape, weights, and documented model type
- uniform magnetic-field kinetic-energy preservation
- Larmor-radius agreement against the analytic gyroradius
- per-stage pipeline losses and transmission
- loss-ledger agreement with per-stage accounting
- accepted-yield calculation

Source and acceptance outputs are validated against external tools through
the adapter interfaces now that they are real: the Xsuite Lattice adapter
(`ParticleState` ↔ `xtrack.Particles` round-trip) and the Geant4 Matter
adapter (material-slab transform, physically bracketed by thin-foil dE/dx
and stopping-block annihilation checks — see the matter-adapter spec).
