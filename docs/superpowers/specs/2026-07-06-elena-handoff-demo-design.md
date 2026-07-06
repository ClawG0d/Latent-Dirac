# ELENA handoff demo (engine-backed degrader into a gated trap catch)

Date: 2026-07-06. Status: accepted.

## Motivation

The downstream half of the AD/ELENA story is now expressible with
shipped, engine-anchored pieces: an ELENA-like 100 keV antiproton bunch
hits a µm-scale aluminium degrader foil — energy loss, scattering, and
in-foil annihilation computed by the vendored vanilla Geant4 engine via
the `matter_slab` element — and the slow survivors are caught by an
ideal Penning trap whose field switches on around the bunch (the
time-gated idealization the decel-capture demo established). This is
the ALPHA-style catching sequence, honestly labeled. The in-ring
deceleration itself (RF ramp, electron cooling) stays out of scope and
out of the captions — README's not-implemented list records the gap.

## Physics baseline (real engine runs, this repo's WSL build)

100 keV antiprotons (13.7 MeV/c, the ELENA extraction momentum) on
G4_Al, FTFP_BERT, 50-particle scans:

- 0.5 µm: 50/50 survive, KE ≈ 55–69 keV
- 1.0 µm: 50/50 survive, KE ≈ 17–36 keV
- 2.0 µm: 30/50 survive, KE ≈ 1.1–15 keV (median 6.2) — the classic
  degrader trade-off: thicker foil, slower but fewer survivors; the
  stopped fraction annihilates in the foil (ledgered by the engine
  stage).

The demo uses the 2.0 µm foil: survivors in the keV band a multi-kV
well can hold.

## Scene design

`examples/scenes/elena_handoff.yaml`, NumPy pipeline only. The engine
transformer binary is injected at run time via
`LATENT_DIRAC_G4_TRANSFORMER` (never stored in the scene); the WebP
generator skips engine-gated demos when the variable is unset, so asset
regeneration on engine-less machines degrades gracefully.

- Source: `antiproton_surrogate` as the parameterized ELENA-extraction
  stand-in — central momentum 0.0137 GeV/c, relative spread 5e-4,
  small forward divergence, sigma 0.5 mm.
- Degrader: `matter_slab`, G4_Al, 2.0 µm (0.002 mm), entry at
  z = 3 mm. Survivors re-enter at the engine scoring plane ~1 mm
  downstream with engine-computed phase space; the engine does not
  advance the clock (the exchange carries no time column), so drift
  timing is measured from the bunch clock — stated in the caption.
- Catch: one gated `penning_trap` element (V0 = −10 kV, d = 12 mm,
  B = 3 T, center z = 10 mm; antiproton confinement needs V0 < 0).
  Eigenfrequencies: omega_c ≈ 2.87e8 rad/s (T_c ≈ 21.9 ns),
  omega_z ≈ 8.15e7 (T_z ≈ 77 ns), stability margin
  omega_c^2 / (2 omega_z^2) ≈ 6. dt = 6e-10 s (~36 steps per
  cyclotron turn). The element runs gate-off for the ns-scale keV
  drift from the foil (6 mm at ~1e6 m/s ≈ 6 ns — gate t_on = 6 ns),
  then the whole trap field (well + axial B) switches on around the
  bunch — the same idealization as the decel-capture demo, carried in
  the caption. Well depth at ±d is |V0|/2 = 5 keV; measured outcome
  (seed 2026): 53/96 annihilate in the foil, 28 caught at
  0.1–2.8 keV, the remainder oscillate at large amplitude in the
  ideal global well instead of escaping (the global parabola never
  lets go — finite electrodes are field-map territory; stated in the
  README caption).
- Ledger: in-foil annihilation (engine deaths at the degrader stage)
  vs escapees (fast survivors that overshoot; they remain alive and
  drift out of frame — the report counts alive, the caption explains
  caught vs escaped visually).

## Deliverables

- Scene YAML + generator entry flagged `requires_engine` (skip +
  warning without the env var) + WebP/HTML assets generated against
  the real engine + README demo section (count 14 → 15, placed in the
  factory-chain narrative after Chain 3) with the engine provenance
  four-tuple in the report block.
- Tests (engine-free, CI-safe): the scene is NOT in the generic
  demo-scene suite; a dedicated test drives it through the
  stub-transformer pattern (wiring: loads, runs, degrader deaths
  ledgered, gate parameters validated) and one test asserts the
  generator skips engine demos without the env var.

## Known wart (pre-existing, surfaced by this composition)

This is the first shipped scene composing a non-engine source with
`matter_slab`. The M2 adapter `setdefault()`s top-level provenance, so
the report's "Source provenance (engine four-tuple)" block shows the
surrogate source with the engine's version tuple filled in — truthful
content under a misleading heading. Revisiting the setdefault
semantics (or the report heading) is follow-up work on the adapter
contract, out of scope for a demo commit.

## Fidelity notes (title and report)

engine transformer (vanilla Geant4 v11.4.2, FTFP_BERT) for the foil;
parameterized surrogate for the ELENA bunch; ideal Penning trap with
idealized instantaneous gating; provenance four-tuple printed by the
report. No RF, no electron cooling, no ring — the handoff only.
