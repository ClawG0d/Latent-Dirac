# Trap storage lifecycle demo (capture, cool, store)

Date: 2026-07-06. Status: accepted.

## Motivation

The trap-physics elements all shipped without a demo: the ideal Penning
trap (validated eigenfrequencies), `buffer_gas_cooling` (Surko stand-in,
the only element that *changes energy*), and `residual_gas_loss`
(storage lifetime). This demo strings them into the antimatter-native
story the differentiable objective already optimizes: not most captured,
but most still alive — and cold — after the hold. It is also the first
demo whose ledger distinguishes *physics* losses (positronium formation
during cooling) from *storage* losses (annihilation on residual gas).

## Scene design

`examples/scenes/trap_storage_lifecycle.yaml`, NumPy pipeline only
(cooling and storage loss are stochastic; the JAX backend rejects them).

- Source: `beta_plus` with an eV-scale endpoint (isotropic burst) —
  used here purely as a parameterized isotropic eV bunch inside the
  trap volume, radius 0.5 mm; the caption says exactly that.
- Trap: ideal Penning trap, B = 0.05 T, V0 = +50 V, d = 5 mm
  (positron confinement needs V0 > 0). Eigenfrequencies:
  omega_c ≈ 8.8e9 rad/s (T_c ≈ 0.72 ns), omega_z ≈ 5.9e8 (T_z ≈ 11 ns),
  omega_- ≈ 2.0e7 (T_mag ≈ 314 ns) — stability margin
  omega_c^2 / (2 omega_z^2) ≈ 110. dt = 2e-11 s resolves the modified
  cyclotron ~36 steps/turn; each transport segment is ~one axial period.
- Interleaving: transport and cooling alternate
  (trap-1 → cool-1 → ... → cool-4 → trap-5 → storage-hold → trap-6 →
  monitor), so each cooling burst cuts kinetic energy at a random
  oscillation phase and the orbit amplitudes decay progressively across
  segments — real damping, made visible. A cooling element freezes
  positions during its (µs-scale) hold while transport windows are
  ns-scale: the README caption states that the animation samples
  transport windows between holds (the CLI report's scope note stays
  the standard transport/acceptance line).
- Cooling per burst: rate 7e6 Hz x hold 2e-6 s ≈ 14 collisions,
  0.3 eV per collision, Ps fraction 0.008, gas at 300 K (floor
  38.8 meV). A ~10 eV mean bunch reaches the floor by the last burst
  (measured: 10.3 eV → 0.06 eV end-station mean; the residue above the
  floor is the Poisson under-collided tail plus KE↔PE exchange in the
  well).
- Storage: `residual_gas_loss` mean lifetime 200 s, hold 60 s
  (~26% ledger deaths).

## Deliverables

- Scene YAML + entry in `tools/generate_scene_demo_webps.py`
  (ledger coloring: cool-N deaths vs storage-hold deaths get distinct
  palette colors) + WebP/HTML assets + README demo section (count
  goes 13 → 14) with the CLI command and the text report.
- Tests: the scene joins the demo-scene load/run suite; a dedicated
  test asserts the cooling actually cooled (final alive mean kinetic
  energy well below the initial mean), that survivors remain, and that
  the ledger carries deaths in at least two distinct stages
  (positronium formation and the storage hold).

## Fidelity notes (carried in the title and report)

ideal Penning trap (parameterized; no electrode geometry) +
Surko-cooling stand-in (constant-rate, parameterized) + storage
lifetime (parameterized, direct `mean_lifetime_s`); source is a
parameterized isotropic eV bunch. No rotating wall, no self-consistent
space charge, no cross-section data — those stay on the roadmap.
