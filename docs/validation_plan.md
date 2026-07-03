# Validation Plan

Initial validation focuses on analytic and accounting checks:
- species mass and charge relationships for particles and antiparticles
- eV/J and GeV/c/SI unit conversion round trips
- `ParticleCloud` shape validation and weighted counts
- source model output species, shape, weights, and documented model type
- uniform magnetic-field kinetic-energy preservation
- Larmor-radius agreement for a quarter turn
- per-stage pipeline losses and transmission
- accepted-yield calculation

Future validation can compare source and acceptance outputs against external
tools through adapter interfaces once those integrations are intentionally
introduced.
