# Rotating-wall trap element (Phase 4 trap physics)

Date: 2026-07-07
Status: implementation spec. Owner decisions (2026-07-07, via dialog):
model the rotating **field only** (parameterized tier; collective plasma
compression is out of scope — see Fidelity), and support **both** a
rotating dipole (m=1) and a rotating quadrupole (m=2).

## What it is

A rotating-wall drive applies a rotating multipole electric perturbation
via a segmented electrode in a Penning-Malmberg trap. In a real plasma it
delivers a torque that, coupled through collisions (e.g. a buffer gas),
compresses the plasma radially and counteracts outward transport — the
cornerstone technique for accumulating and holding positron / antiproton
plasmas.

## Scope / fidelity (honesty)

This models the rotating **single-particle electric field** only, at the
`parameterized` tier — the trap-optics analogue of the ideal Penning well
and the hard-edge magnets. It is a real, time-dependent E field; particles
pushed through it feel the correct rotating force and E×B drift.

It is **not** a self-consistent plasma model: the radial *compression* a
rotating wall produces is a collective, collisional effect (rotating field
+ dissipation), which requires a PIC / collective solver the NumPy+mean-field
core does not have (cf. the mean-field space-charge caveat). A demo pairs
`rotating_wall` with `buffer_gas_cooling` to show the qualitative
rotation/drift with dissipation, clearly labelled — but plasma compression
is **not** a figure of merit, and no energetics are claimed. This boundary
is stated in the field docstring, the scene report, and the demo.

## Field model

A transverse (x, y) rotating multipole E field, z-independent, global (no
hard edge — like `uniform_field` / `penning_trap`); B = 0. With
ω = 2π·`frequency_hz`, θ = ω·t + `phase_rad`:

- **dipole (m=1)** — a uniform transverse field rotating at ω:
  `E = amplitude_v_m · [cos θ, sin θ, 0]`. Independent of position.
- **quadrupole (m=2)** — a quadrupole pattern whose axes rotate, linear in
  transverse position, scaled so |E| = `amplitude_v_m` at radius
  `radius_m` (the electrode/wall radius):
  with `g = amplitude_v_m / radius_m`,
  `E = g · [−(x cos θ + y sin θ),  y cos θ − x sin θ, 0]`
  (= −∇Φ for Φ = (g/2)[(x²−y²)cos θ + 2xy sin θ], so it is curl- AND
  divergence-free — a true quadrupole, not a radial field — with |E| = g·r).

Both are exact, curl-free electrostatic-instant fields at each t (a
quasi-static rotating pattern; radiative/retardation effects are out of
scope, consistent with the rest of the field library).

## Schema — `RotatingWallElement`

- `type: "rotating_wall"`
- `multipole: Literal[1, 2]`
- `amplitude_v_m: float` (Field(gt=0)) — field magnitude (at `radius_m` for m=2)
- `radius_m: float` (Field(gt=0), default 0.02) — reference/wall radius; used only for m=2
- `frequency_hz: float` (Field(gt=0))
- `phase_rad: float = 0.0`
- optional `t_on_s` / `t_off_s` (reuse the existing time-gate mechanism)
- optional `steps` (per-element solver-step override, like other field elements)

Add to `ElementSpec`, `FIELD_ELEMENT_TYPES`, and the JAX
`_TRANSPORT_TYPES` / `_SWEEPABLE_PARAMS`.

## Backends (mirror pair)

- NumPy: `RotatingWallField(BaseModel, Field)` in
  `latent_dirac/fields/rotating_wall.py`; `E(x, t)` per the model, `B` zero.
  Wire in `build._base_field_for`.
- JAX: two pure field fns `_rotating_dipole_field` and
  `_rotating_quadrupole_field` `(jnp, positions, time_s, params)`;
  `_field_fn_for` dispatches on `element.multipole` (static, exactly like the
  solenoid `profile` dispatch). `_SWEEPABLE_PARAMS["rotating_wall"] =
  ("amplitude_v_m", "radius_m", "frequency_hz", "phase_rad")`. The shared
  `_field_fn_for` + `_TRANSPORT_TYPES` cover the differentiable mirror too.
- Time-dependence: the field fns use `time_s` directly for θ; the Boris
  solver already passes per-particle `time_s` to fields (as `TimeGatedField`
  relies on).

## Validation (TDD, analytic)

- m=1: at fixed t the field is uniform and its direction is θ; over t it
  rotates at ω (period 1/f); |E| = amplitude_v_m.
- m=2: |E| = amplitude_v_m·(r/radius_m); direction pattern; at r=0 field is 0.
- A test particle in a strong uniform B (added via a separate
  `uniform_field` or the trap) plus the rotating dipole shows the expected
  E×B drift sense; energy bookkeeping sane.
- NumPy↔JAX parity: same trajectory to fp tolerance (the standard mirror
  test), for both multipoles.
- Shape contract: single `(3,)` and batch `(N,3)` inputs.

## Viz / report / docs

- `FIDELITY_LABELS["rotating_wall"]` (honesty coverage test) +
  `_FIELD_DESCRIPTIONS` in `scene_report`, with the compression caveat in
  the status line.
- Field-line rendering: the rotating field is time-dependent; render its
  lines at t=0 (a snapshot), consistent with how field lines already draw
  the instantaneous model field.
- A demo scene (`rotating_wall` + `penning_trap` + `buffer_gas_cooling`),
  CHANGELOG, roadmap (flip the Phase-4 rotating-wall bullet), scene_schema.

## Non-goals

- No self-consistent plasma / PIC; no compression figure of merit; no
  energetics. No RF cavity structure, no ramp shapes (ideal switching via
  the existing gate), no electrode geometry. m ≥ 3 multipoles deferred.
