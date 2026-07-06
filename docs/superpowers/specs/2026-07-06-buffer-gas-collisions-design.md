# Buffer-gas collisions design (T4 — research + design spec)

Date: 2026-07-06
Status: design spec (research deliverable; code landing deferred to spec
review, per TASK-SPLIT.md T4). No code in this change.

## Purpose

Design the Monte Carlo buffer-gas collision operator that models
Surko-type positron accumulation and cooling in a Penning-Malmberg
trap: the mechanism by which a hot positron cloud is captured and
cooled to near ambient temperature through inelastic collisions with a
buffer gas. This complements the already-shipped `residual_gas_loss`
element (a fixed-lifetime annihilation loss) by modeling the
energy-changing collisions that actually cool the cloud, with
annihilation/positronium formation as one competing channel among
several.

Scope of this document: the physics to model, the cross-section data
sourcing and its provenance discipline, the collision-operator
algorithm, the interface to the existing trap field model and Boris
kernel, and the fidelity tiering. The code landing point is decided
after this spec is reviewed.

## Physics to model

Surko buffer-gas trapping (three-stage differential pumping in the
canonical design) rests on one near-coincidence: in N2 the
near-threshold **electronic excitation** cross section (the a¹Πg and
a′¹Σ states, threshold ~8.5 eV) is *larger* than the competing
**positronium formation** cross section (threshold ~8.8 eV), so a
positron preferentially loses ~8.5 eV per collision and stays trapped
rather than annihilating. Successive collisions ladder the positron
down in energy; once below the electronic-excitation threshold, cooling
continues through vibrational and rotational excitation (often in a
second, lower-Z cooling gas such as CF4 or CO2) down to the ambient gas
temperature.

Collision channels the operator must represent, each with its own
cross section σ_i(E) and energy/angle kinematics:

- **elastic scattering** — momentum transfer, negligible energy loss
  (positron/molecule mass ratio), sets angular diffusion.
- **electronic excitation** — the trapping workhorse in N2; discrete
  energy loss at the state threshold.
- **vibrational / rotational excitation** — the low-energy cooling
  channels; small discrete energy losses.
- **positronium formation** — a loss channel (the positron is removed;
  Ps annihilates). Modeled as a ledgered loss endpoint, consistent with
  the safety scope (no energetics).
- **direct annihilation** — a second, much smaller loss channel
  (Z_eff-scaled); ledgered loss endpoint.
- **ionization** — a loss/heating channel above its threshold; usually
  subdominant in the trapping regime but included for completeness.

## Cross-section data sourcing and provenance

Load-bearing honesty point discovered in the research: unlike *electron*
cross sections (curated in the open LXCat database), **positron**
scattering cross sections are **not** in a single open database. LXCat
is presently electron-neutral only. Positron data lives in the
published experimental and theoretical literature (the Surko group at
UCSD; Marler & Surko; Chiari & Zecca reviews; the Surko RMP review
"Plasma and trap-based techniques for science with positrons").

Consequences for the AGENTS.md "open, provenance-tracked dataset" rule:

- The dataset is curated **per source**, not pulled from one database.
  Each σ_i(E) table carries its citation (author, year, DOI), the
  measurement/calculation method, the target gas, and the energy range
  of validity in its header — the same provenance discipline as the
  Geant4 four-tuple, adapted to cross-section data.
- Committed tables live under `examples/data/` (or a dedicated
  `data/cross_sections/`) as CSV with a provenance header; the loader
  validates the header and the monotonic energy grid.
- Fidelity tier: **table-based** when driven by a cited dataset;
  **parameterized** for any analytic stand-in (e.g. a single-channel
  constant-loss toy model) used before real tables are curated. A
  buffer-gas model calibrated against measured trapping/cooling times
  would graduate toward **externally calibrated**.
- No cross-section values are invented; a channel with no curated table
  is declared absent, not guessed.

## Collision-operator algorithm

The standard, provenance-clean choice is the **null-collision (Skullerud)
Monte Carlo** method, interleaved with the Boris transport by operator
splitting (transport for dt, then collide):

1. Precompute the maximum collision frequency ν_max = max over E of
   [n_gas · σ_total(E) · v(E)] on the modeled energy range (n_gas from
   the buffer-gas pressure and temperature).
2. Per particle per step, draw the time to the next candidate collision
   from ν_max (a constant rate — the null-collision trick avoids
   integrating a varying rate). Within the step, a particle either has a
   candidate collision or not.
3. On a candidate, draw a uniform r ∈ [0,1): if r < σ_total(E)/σ_max the
   collision is real, else it is a null (no-op). If real, pick the
   channel i with probability σ_i(E)/σ_total(E).
4. Apply the channel kinematics: elastic → rotate the velocity by a
   sampled scattering angle (isotropic or from a differential cross
   section if curated), tiny energy loss; inelastic → subtract the
   threshold energy and rotate; loss channels (Ps formation, direct
   annihilation, ionization above threshold) → kill the particle and
   stamp the ledger (reusing the `residual_gas_loss` / aperture
   kill-and-stamp pattern).

Purity discipline: the collision operator is a pure function of
(state, gas params, rng draws), matching the kernel/State conventions.
Seeded per stage like `residual_gas_loss`
(`default_rng(scene.seed + <offset> + stage_index)`) for reproducibility.
NumPy pipeline only (stochastic, energy-changing — no static-program
form; the JAX backend rejects it like the other stochastic elements).

## Interface to the existing trap

- The buffer-gas region is a scene element (working name
  `buffer_gas_cooling`) carrying: the gas species/mixture, pressure,
  temperature, the cross-section dataset reference (path, resolved
  relative to the scene file like the field map / yield table), and the
  residence/hold time or step count.
- It composes with the shipped `penning_trap` field: the trap confines
  while the collision operator cools. Operator splitting — one Boris
  substep in the trap field, then one collision substep — over the hold.
- It supersedes `residual_gas_loss` for the trap-cooling case: that
  element remains the cheap fixed-lifetime loss model; the buffer-gas
  operator is the fidelity upgrade that also *cools* (changes energy),
  not just removes. Both stay available at their declared tiers.
- Rotating wall and space charge remain separate Phase-4 directions;
  the collision operator does not model them.

## Validation plan

- Energy-loss-per-collision equals the channel threshold (electronic
  excitation removes ~8.5 eV in N2) — analytic per-channel checks.
- A hot cloud in a trap + buffer gas cools toward the gas temperature
  over the hold (mean energy decreases monotonically to ~kT), and the
  trapped fraction matches the electronic-excitation-vs-Ps-formation
  branching ratio at the injection energy — qualitative ordering checks,
  not absolute rates (absolute cooling times are an
  externally-calibrated goal, needing measured cross sections and a
  benchmark against published trapping efficiencies).
- Null-collision correctness: the realized collision rate reproduces
  n_gas · σ_total(E) · v(E) within sampling error.

## Rejected / deferred alternatives

- **Direct (non-null) collision integration** — requires integrating a
  time-varying collision rate per step; the null-collision method is the
  standard, cheaper, and less error-prone choice.
- **Pulling positron cross sections from LXCat** — not possible; LXCat
  is electron-only. Per-source curation is mandatory (see provenance).
- **Modeling Ps annihilation energetics** — out of safety scope; Ps
  formation is a ledgered loss endpoint only.

## Open questions for review

1. Data first or operator first: curate one real N2 dataset (with
   provenance) before coding, or land the operator against a
   parameterized single-channel stand-in and swap in tables later?
2. Element granularity: one `buffer_gas_cooling` element, or a general
   `gas_region` that also subsumes `residual_gas_loss`'s annihilation
   channel?
3. Data location: `examples/data/` (alongside the yield table) or a
   dedicated `data/cross_sections/` tree with its own README of sources?

## Sources

- Surko Plasma Research Group, UC San Diego — trap-based positron
  methods and measured near-threshold cross sections
  (https://positrons.ucsd.edu/traps.php).
- "Investigation of buffer gas trapping of positrons", J. Phys. B
  (https://iopscience.iop.org/article/10.1088/1361-6455/aba10c).
- "A CF4 based positron trap", J. Phys. B
  (https://iopscience.iop.org/article/10.1088/0953-4075/49/21/215001).
- Buffer-gas trap overview
  (https://en.wikipedia.org/wiki/Buffer-gas_trap).
- LXCat project (electron-only; establishes the open,
  provenance-tracked data model this spec adapts for positrons)
  (https://us.lxcat.net/).

## Implementation slice 1 (operator-first, parameterized stand-in)

Owner decisions (2026-07-06): operator-first with a parameterized
single-channel stand-in; a single `buffer_gas_cooling` element; data
under `examples/data/` when real tables land. This slice writes the
operator now and swaps in cited N2 cross-section tables later
(table-based tier), without touching the interface.

### Element `buffer_gas_cooling` (parameterized tier)

A standalone cooling region (like `residual_gas_loss`, but energy-
changing), applied over a hold time — not yet interleaved with Boris
transport (operator-splitting with the trap field is a later slice; a
standalone region is enough to model a fixed cooling stage and is
cleanly testable).

Parameters (all direct inputs at this tier):

- `hold_time_s >= 0` — residence in the cooling gas.
- `collision_rate_hz > 0` — mean collision frequency nu. Constant here;
  the energy-dependent `nu(E) = n_gas sigma(E) v(E)` form (which is what
  makes the null-collision method necessary) is the table-based upgrade.
- `energy_loss_ev > 0` — energy removed per inelastic (cooling)
  collision: the N2 electronic-excitation stand-in (~8.5 eV in reality),
  a single channel at this tier.
- `ps_fraction` in [0, 1] — probability a collision is a
  positronium-formation loss (particle killed) rather than a cooling
  collision.
- `gas_temperature_k >= 0` (default 300) — the cooling floor: kinetic
  energy is not driven below (3/2) k_B T (a positron cannot cool below
  the ambient gas).

### Operator (pure, seeded)

Per alive particle, with `default_rng(scene.seed + 3313 + stage_index)`:

1. Draw `n ~ Poisson(collision_rate_hz * hold_time_s)` collisions.
   Constant rate makes Poisson exact; no null-collision bookkeeping is
   needed until nu varies with energy.
2. For each collision: with probability `ps_fraction` it is a Ps-loss —
   kill the particle and stop (the `Stage` wrapper stamps the ledger,
   reusing the aperture/`residual_gas_loss` kill pattern). Otherwise it
   is a cooling collision: reduce kinetic energy by `energy_loss_ev`,
   floored at `(3/2) k_B T`, and rescale the momentum vector to the new
   magnitude (direction preserved at this tier; angular scattering is a
   later refinement — noted, not modeled). Relativistic
   |p| <-> KE via the existing unit helpers.
3. Survivors: advance `time_s` by `hold_time_s`.

NumPy pipeline only; the JAX backend rejects it (stochastic, energy-
changing — no static-program form), and the differentiable objective
does likewise, matching `residual_gas_loss` / space charge. Fidelity
tier stated in the docstring and CHANGELOG: parameterized (all inputs
direct); the table-based upgrade path is recorded above.

### Tests (TDD)

1. Cooling: a hot mono-energetic cloud's mean KE decreases monotonically
   over the hold and converges toward (3/2) k_B T (never below).
2. Energy floor: with a large hold, every survivor sits at the floor.
3. Ps loss: survival fraction tracks the Poisson/branching expectation
   for the chosen rate and `ps_fraction`; killed particles are ledgered
   at the element's stage index; survivors keep -1.
4. `hold_time_s = 0` or `collision_rate_hz -> 0` (few collisions): cloud
   essentially unchanged.
5. Momentum-direction preserved per cooling collision; only |p| shrinks
   (a stated stand-in simplification).
6. Determinism given seed; dead-on-entry particles untouched (no revive,
   no restamp).
7. Schema validation (ranges) and JAX-backend rejection.
