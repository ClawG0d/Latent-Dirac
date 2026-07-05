# Penning Trap Field Design (Spec 4a)

## Objective

The first half of the Phase 4 "trap physics step one": an ideal Penning
trap field model and its scene element, with the textbook eigenfrequency
physics as the validation suite. Buffer-gas collisions (the Surko half)
are a separate later spec.

## Physics

Ideal Penning trap: a uniform axial magnetic field `B = B0 ẑ` plus the
quadrupole electrostatic potential

```
V(x, y, z) = (V0 / (2 d²)) · (z² − (x² + y²)/2)
E = −∇V = (V0 / d²) · (x/2, y/2, −z)
```

with characteristic trap dimension `d` and well depth parameter `V0`
(positive `q·V0` confines axially). The potential satisfies Laplace's
equation exactly. Single-particle motion has three eigenfrequencies:

- axial: `ω_z² = q V0 / (m d²)`
- modified cyclotron / magnetron:
  `ω± = (ω_c ± sqrt(ω_c² − 2 ω_z²)) / 2` with `ω_c = q B0 / m`
- invariance: `ω+ + ω− = ω_c` and `ω+ · ω− = ω_z² / 2`
- stability requires `ω_c² > 2 ω_z²`

Fidelity tier: **parameterized** (ideal quadrupole potential; real
electrode stacks, space charge, and field imperfections are out of
scope — this is the trap-optics analog of the hard-edge magnets).

## Design

- `PenningTrapField` in `latent_dirac/fields/penning_trap.py`: parameters
  `v0_volt` (well parameter, sign chosen by the caller for the species),
  `d_m` (characteristic dimension), `b_tesla` (axial field),
  `center_z_m = 0`. `E(x, t)` from the analytic gradient, `B(x, t)`
  uniform axial — the standard `Field` contract, so the Boris solver and
  the scene pipeline work unchanged. The field is global (no hard edge):
  an ideal trap, documented as such. Constructor validates `d_m > 0` and
  warns of nothing else — stability is physics, checked by a helper
  `is_stable(species)` and validated in the scene layer? No: keep the
  field dumb; expose `eigenfrequencies(species) -> (omega_plus,
  omega_minus, omega_z)` raising if unstable, so tests and users get the
  analytic values from the same source.
- Scene element `penning_trap` (transport type): `v0_volt`, `d_m`,
  `b_tesla`, `center_z_m`, optional `steps`. Sweepable parameters
  registered for the JAX backend; the differentiable mirror gets the same
  branch (per the mirror note in `_make_simulator`).
- JAX field function `_penning_trap_field` mirroring the analytic E and B.

## Validation

- Field: E matches the analytic gradient; `div E = 0` numerically;
  potential shape (confining in z, deconfining radially for the right
  sign).
- Dynamics (NumPy pipeline): on-axis particle undergoes axial SHO at
  `ω_z` (period extracted from zero crossings, tight tolerance);
  radial-plane motion contains the two rotation frequencies satisfying
  `ω+ + ω− = ω_c` (fit via the analytic eigen-decomposition of the
  initial conditions, or measure `ω+` from a launch that suppresses the
  magnetron mode); trapped particle stays bounded over many axial
  periods.
- `eigenfrequencies` raises for an unstable configuration
  (`ω_c² ≤ 2 ω_z²`).
- JAX parity: scene with a `penning_trap` element matches the NumPy
  pipeline element-wise (extends the existing parity suite).
- Timescale honesty: tests simulate short windows (axial periods, not
  seconds of storage); the long-timescale guiding-center solver remains a
  separate roadmap item.

## Non-Goals

- buffer-gas collisions, rotating wall, space charge (later specs)
- real electrode geometries (field maps cover that route)
- second-order guiding-center / long-storage integration
