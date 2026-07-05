# Physics Scope

The current physics scope is fast source-to-acceptance modeling in SI units.
The package includes positron and antiproton source terms, relativistic
charged particle motion under Lorentz force, beamline acceptance, loss
accounting, and accepted-yield diagnostics.

## Fidelity tiers

Every source model declares one of five fidelity tiers: placeholder,
parameterized, surrogate, table-based, or externally calibrated.

Space-charge and collective effects follow the standard solver ladder used
across the industry, and each rung must be declared explicitly:

1. **single-particle tracking** (current tier): no self-consistent fields,
   valid for low-intensity clouds
2. **iteratively self-consistent** (future): gun-iteration style coupling
   for moderate densities
3. **fully self-consistent PIC** (future, via adapters): required for
   non-neutral plasmas in traps

This phase does not model full electromagnetic showers, full hadronic
showers, annihilation physics, material activation, shielding, target
engineering, real facility controls, or operational recipes.

Antiparticles use normal positive mass. Positrons use electron mass and
opposite electron charge. Antiprotons use proton mass and opposite proton
charge.
