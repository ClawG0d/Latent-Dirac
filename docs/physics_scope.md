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

1. **single-particle tracking** (default): no self-consistent fields,
   valid for low-intensity clouds
2. **mean-field self-field** (shipped, opt-in): a parameterized
   uniform-sphere model refit from the alive cloud every step
   (`space_charge: uniform_sphere`); electrostatic only (beta << 1),
   single-sphere fit, no iteration — trap-regime studies; NumPy
   pipeline only
3. **iteratively self-consistent** (future): gun-iteration style coupling
   for moderate densities
4. **fully self-consistent PIC** (future, via adapters): required for
   non-neutral plasmas in traps

## Field layer

The classical electromagnetic field layer is SI-unit based and callable by
position and time: every field model answers `E(x, t)` in V/m and `B(x, t)`
in tesla. Field models are composable through `CompositeField`, which sums
component contributions. Current models are idealized hard-edge beam optics
models (uniform, solenoid, dipole, quadrupole), plus a table-based
`FieldMapField` that trilinearly interpolates externally computed fields on
a regular grid (COMSOL regular-grid CSV import). The field layer is
intentionally separate from quantum wavefunction evolution and from full
Maxwell field solvers. RF fields and further field-map formats (CST,
SIMION) are later field-library extensions (see the roadmap).

The Python core does not implement shower physics in-house:
electromagnetic and hadronic showers are delegated to the vendored
vanilla Geant4 engine track and enter scope only as engine-computed
diagnostics (no build or runtime coupling ships yet). Annihilation is
modeled only as a ledgered loss endpoint with kinematic two-photon
emission for visualization. Annihilation energetics as a figure of
merit, material activation, shielding design, target engineering, real
facility controls, and operational recipes stay out of scope — see
[safety_scope.md](safety_scope.md) for the canonical exclusion list.

Antiparticles use normal positive mass. Positrons use electron mass and
opposite electron charge. Antiprotons use proton mass and opposite proton
charge.
