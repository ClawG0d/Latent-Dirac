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

Future validation can compare source and acceptance outputs against external
tools through adapter interfaces once those integrations are intentionally
introduced.
