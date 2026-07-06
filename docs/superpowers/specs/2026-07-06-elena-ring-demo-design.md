# ELENA-like ring demo (Xsuite-tracked betatron portrait)

Date: 2026-07-06. Status: accepted.

## Motivation

The third rung of the AD/ELENA ladder (after the engine-backed
production demo and the degrader-catch handoff): the ring itself,
tracked through the `xsuite_lattice` scene element at fixed momentum.
The in-ring deceleration (RF ramp, electron cooling) remains
unmodeled and on the README's not-implemented list.

## Lattice artifact (license-clean, reproducible)

`examples/data/elena_like_ring.json` is built programmatically by
`tools/make_elena_like_line.py` from public machine-scale parameters
(circumference 30.4056 m, six-fold symmetry, 60° bends of bending
radius 0.927 m, antiproton reference at 13.7 MeV/c). It is explicitly
NOT the real ELENA optics: quadrupole strengths and pole-face rotation
come from the script's deterministic (edge, k1) stability scan — no
CERN optics files are copied, so there is no data-license question;
the generator script is the artifact's full provenance. Fidelity tier:
parameterized (ELENA-inspired).

Physics found by the scan, worth recording: a ring this tight
over-focuses one plane on bend geometry alone — pure sector bends
(edge 0) over-focus horizontally (per-bend horizontal focal length
0.89 m from rho = 0.927 m), full rectangular bends (edge = theta/2)
over-focus vertically; the stable window sits between. The committed
lattice uses edge angle 0.14 rad, k1 = 0.2 /m², tunes
qx = 2.6789, qy = 1.2377 (fractional parts away from low-order
resonances; asserted by the script and pinned by test).

## Scene and demo

- `examples/scenes/elena_ring.yaml`: `antiproton_surrogate` at fixed
  13.7 MeV/c (zero momentum spread, 2 mrad divergence, 3 mm sigma) →
  `xsuite_lattice` (60 turns) → monitor. Report: full transmission —
  the working point is stable — with mean kinetic energy 0.100 MeV.
- WebP: a stroboscopic (Poincaré-style) turn-by-turn portrait —
  plot axes (turn, x, y); once-per-turn samples are drawn as points,
  NOT connected lines, because the betatron phase advances a large
  fraction of a turn between samples and connecting them aliases into
  zig-zags. Color = initial transverse amplitude. Rendered by a
  dedicated `frames_fn` (the trajectory pipeline does not apply:
  the lattice element produces no per-step recording).
- Generation is `requires_xsuite`-gated (like the engine gate):
  the generator skips it with a note when `xtrack` is not importable,
  committed assets stay; the WSL box regenerates. The generator also
  gains `--only <file>` for single-demo regeneration.

## Validation (CI-safe: skips without xtrack)

- The committed lattice loads, has the design circumference, and its
  4d twiss reproduces the recorded tunes (abs 1e-3, robust across
  xtrack patch versions) — a drifted or hand-edited artifact fails.
- The scene runs: 60 turns, no losses, mean kinetic energy 100 keV.
- The generator marks the demo xsuite-gated; the demo-assets test
  accepts its absence exactly when xtrack is unavailable.
