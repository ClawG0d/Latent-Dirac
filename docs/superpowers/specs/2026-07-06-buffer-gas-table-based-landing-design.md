# Buffer-gas table-based landing (T5 — implementation of the table tier)

Date: 2026-07-06
Status: implementation spec. Builds on the design spec
`2026-07-06-buffer-gas-collisions-design.md` (the physics, channels,
null-collision algorithm, and provenance discipline are decided there).
Owner decision (execution-plan spec, 2026-07-06): the Mac lane's next
task (T5) is the table-based upgrade of `buffer_gas_cooling`.

## Owner decision on data sourcing (2026-07-06)

Positron scattering cross sections are not in a single open database
(LXCat is electron-only); real tables must be curated per publication
with DOI provenance, and **no cross-section value may be invented**
(design spec). Chosen approach: **build the table-based infrastructure
first** — the loader, the null-collision operator, and the
`buffer_gas_cooling` table path — validated against a clearly-labeled
**synthetic** placeholder table. Real cited CSVs drop in later and
flip the fidelity tier to `table-based`; a synthetic table keeps the
element at `parameterized`. Nothing synthetic is ever presented as real
physics.

## Deliverables (sliced, each a reviewed commit)

### Slice 1 — cross-section table format + loader (this)

- New package `latent_dirac/collisions/`.
- `cross_sections.py`: `CrossSectionTable` (a frozen dataclass:
  `energies_ev` monotone-increasing ndarray; `channels` dict
  name→σ(m²) ndarray aligned to the grid; `thresholds_ev` dict
  name→float; `provenance` dict; `fidelity_tier` str) plus
  `load_cross_sections(path) -> CrossSectionTable` and
  `sigma(table, channel, energy_ev)` (linear interpolation; **zero
  outside the tabulated range**, matching the field-map out-of-range
  convention — a positron above/below the measured range simply has no
  modeled cross section there rather than an extrapolated guess).
- CSV format: a commented provenance header then a numeric block.
  Required header keys (loader fails fast if any missing):
  `gas`, `fidelity_tier` (∈ {`parameterized`, `table-based`}),
  `source`, `energy_unit` (must be `eV`), `sigma_unit` (must be `m^2`),
  `channels` (comma list), `thresholds_ev` (name=eV comma list). When
  `fidelity_tier: table-based`, a non-empty `doi` and `method` are also
  required (real data must be citable); for `parameterized` they are
  optional. Numeric block: header row `energy_eV,<channel>,…` then rows;
  the loader checks the energy grid is strictly increasing, all σ ≥ 0,
  and the channel columns match the `channels` header.
- Toy data: `examples/data/cross_sections/n2_positron_toy.csv` —
  `fidelity_tier: parameterized`, `source: synthetic placeholder — NOT
  physical data`, a hand-authored N₂-shaped grid (elastic flat-ish,
  electronic turning on at 8.5 eV, positronium at 8.8 eV). Purely for
  exercising the machinery; the header shouts SYNTHETIC.
- Tests (`tests/test_cross_sections.py`): loads the toy table; rejects a
  missing required header key; rejects a non-monotone energy grid;
  rejects a negative σ; rejects `table-based` with no DOI; `sigma()`
  interpolates mid-grid and returns 0 outside the range; a per-channel
  threshold is read.

### Slice 2 — null-collision (Skullerud) operator

- `operator.py`: pure `buffer_gas_collide(state, table, *, n_gas_m3,
  hold_time_s, substeps, rng, xp=np) -> ParticleState`. ν_max from
  max(n_gas·σ_total(E)·v(E)) on the grid; per particle per substep draw
  a candidate collision at the constant null rate; real vs null via
  σ_total(E)/σ_max; channel pick via σ_i/σ_total; kinematics — elastic
  = isotropic direction rotate + negligible ΔE; inelastic = subtract the
  channel threshold, rotate; loss channels (positronium, annihilation,
  ionization) = kill + stamp the ledger (reuse the residual_gas_loss
  kill-and-stamp). Pure, seeded, NumPy only (stochastic → the JAX
  backend rejects it, as it already rejects `buffer_gas_cooling`; no
  mirror-pair change).
- Tests: energy loss per inelastic collision equals the channel
  threshold; a hot cloud cools monotonically toward (3/2)kT over the
  hold; the trapped-vs-Ps branching at injection energy follows
  σ_electronic/σ_ps; loss particles are ledgered; determinism under a
  fixed seed.

### Slice 3 — wire the table tier into `buffer_gas_cooling`

- Schema: add optional `cross_section_path: str | None` and
  `gas_pressure_pa: float | None`. A `model_validator` enforces exactly
  one mode: **table** (cross_section_path + gas_pressure_pa set) or
  **constant-rate** (collision_rate_hz + energy_loss_ev + ps_fraction
  set, the current fields, now optional). Backward compatible: existing
  scenes with the constant-rate fields and no path keep working.
- `loader._resolve_scene_relative_paths`: resolve `cross_section_path`
  relative to the scene file (like `line_path`/`table_path`).
- `build._buffer_gas_cooling_action`: when a path is set, load the table
  and drive `buffer_gas_collide` (n_gas = P/(k_B T)); else the existing
  constant-rate path. The element's reported fidelity tier follows the
  table's (`parameterized` for the synthetic toy, `table-based` for a
  cited dataset).
- `viz/scene_3d.py`: `buffer_gas_cooling` already has a representation;
  no new geometry needed (region element). Confirm the fidelity label
  reflects the tier.
- Docs: `docs/scene_schema.md`, CHANGELOG; if a real table ever lands,
  the three-place status rule applies.

## Non-goals

- No real positron cross-section values in this work (owner decision).
- No rotating wall, no space charge, no guiding-center solver (Phase 4).
- No JAX-backend support (stochastic, energy-changing — rejected there).

## Honesty

The synthetic table's header and the CHANGELOG state plainly it is a
placeholder, not measured/theoretical data; the element stays
`parameterized` until a cited table is curated. No comparative
performance wording. Data provenance travels in every table header
(gas, tier, source, doi, method, energy range) — the cross-section
analogue of the engine four-tuple.
