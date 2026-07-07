# Execution plan: GPU lane → M3/M4 → Phase 4 (owner-approved)

Date: 2026-07-06. Status: accepted (six owner decisions recorded
below). Near-term sessions are planned to the deliverable level;
far-term items stay direction-level on purpose — they get their own
specs when they start.

## Owner decisions (2026-07-06)

1. Plan horizon: the full Phase 4 blueprint (near-term detailed,
   far-term direction-level).
2. Priority: the GPU lane leads; M3 yield-table work interleaves in
   its gaps.
3. Interaction protocol: pop a dialog for genuine owner decisions
   (physics scope, outward-facing wording, destructive/irreversible
   actions, tasks expected to exceed ~1 hour); post a summary after
   every push once CI settles; otherwise proceed autonomously.
4. Official benchmark numbers live in `docs/benchmarks.md` with the
   full label set; the README links to it and never carries the table.
5. The Mac collaborator's next assignment: the buffer-gas
   cross-section table, first tier (per the 2026-07-06 buffer-gas
   collisions spec) — upgrading `buffer_gas_cooling` from
   constant-rate to table-based. T3 (CST/SIMION) stays open behind it.
6. M4 (companion acceleration library) starts only after the GPU
   benchmark publication, reusing its benchmark discipline.

## Environment facts (probed, not assumed)

WSL2 GPU passthrough works today: RTX 5070 Ti visible via nvidia-smi
(driver 591.86), libcuda present, Python 3.12. No owner-side install
is needed to start. Consumer Blackwell means the float32
dimensionless-kernel lane is the fit; float64 truth stays on the CPU
NumPy reference. Official numbers only from this box, fully labeled.

## Near-term sessions (this machine)

### G1 — WSL GPU environment + smoke (one session)

- Clone the repo on ext4 (`~/Latent-Dirac`; never `/mnt/c` — 17k-file
  vendored tree over 9P), venv, `pip install -e ".[dev,jax]"` plus
  `jax[cuda12]` per the official support matrix (Blackwell may need a
  recent jaxlib; if only a nightly supports it, that is an owner
  dialog before pinning).
- Smoke: `jax.devices()` sees the GPU; `boris_step` under
  `jax.jit` on GPU float32 vs CPU float64 for one analytic case
  (uniform-B gyration) at tolerance sanity level.
- Record the environment recipe in `docs/solver_backends.md`.
- No numbers published from this session.

### G2 — float32 GPU validation suite (one to two sessions)

- Spec first (tolerance policy is its own design record): tiered
  tolerances — per-element trajectory agreement (fp32 GPU vs fp64
  CPU) at statistical level, accepted-fraction agreement at absolute
  level, plus a strict x64-on-GPU vs CPU equality tier to separate
  hardware/compiler drift from precision loss.
- Implement the pytree registration wrapper `ParticleState` needs for
  real jit boundaries (the docstring's known Phase-3 gap:
  `tree_unflatten` must bypass `__post_init__`; `metadata` is not
  hashable for static aux) — TDD, mirror-pair discipline untouched.
- Tests live in the suite but skip without a GPU (CI stays green;
  the WSL box runs them for real).

### G3 — honest benchmark suite + publication (one session)

- `tools/run_benchmarks.py`: kernel micro-benchmark (particle-count
  sweep), whole-scene cases, batch sweep via `BatchedSceneProgram`;
  fp32 GPU vs fp64 CPU vs fp32 CPU.
- Output `docs/benchmarks.md` with the full label set (GPU model,
  WSL2, CUDA/driver, jax/jaxlib versions, integrator, dt, particle
  count, batch size, fidelity tier) and the exact reproduction
  commands. README links from Documentation. The honesty regex scans
  docs/*.md: comparative wording stays out; the numbers speak.

### M3-a — positron yield table (interleaved, one session + engine runtime)

- Generalize `engine/yieldgen` (or add a sibling app) for
  electron-on-tungsten positron production; same CSV/provenance
  contract. Long engine runs happen in G-session gaps.
- Python side: a positron yield-table source (generalizing or
  paralleling `antiproton_yield_table`), demo/README wiring.

### M3-b — surrogate graduation (half session)

- Calibrate the antiproton surrogate against the committed engine
  table; flip its tier to `externally calibrated` with the
  same-commit three-place status rule (README Solvers table,
  roadmap, CHANGELOG + the adapter/roadmap status test if touched).

### M4 — companion acceleration library (after G3 publication)

- Own positioning spec first; `G4VFastSimulationModel` hook, EM
  domain first; performance claims only against open vanilla-Geant4
  benchmarks using the G3 discipline. Session breakdown written when
  it starts (est. 3–5 sessions to a first honest prototype).

## Mac collaborator lane (recorded in TASK-SPLIT)

- T5 (new): buffer-gas cross-section table, first tier — curate
  positron–N2/CF4 energy-dependent cross sections per publication
  with per-source provenance (the buffer-gas spec's discipline),
  upgrade `buffer_gas_cooling` to table-based. CPU-only, research
  heavy.
- T3 (still open): CST/SIMION field-map importers.

## Phase 4 blueprint (direction level — spec before start)

- Trap physics: rotating-wall element (time-dependent rotating
  transverse E; needs a field-model spec), guiding-center/secular
  solver for long-timescale storage (new Solver class).
- JAX backend fill-ins (this machine, mirror-pair territory): field
  maps, batched monitor snapshots, streaming trajectory recording.
- Detector component (last `planned` row in the Solvers table):
  parameterized response model first, Garfield++ later — positioning
  spec required; unassigned.
- Digital twin: offline replay/calibration only (needs an openPMD
  read path — currently write-only).
- Continuous/owner-side: PyPI trusted publishing, docs-site
  publication, JOSS/PRAB paper.

## Interaction protocol (operational)

- Every session starts with `git fetch` + status reconciliation
  (three parallel sessions on one master is the norm).
- Dialogs for: physics-scope choices, outward-facing wording,
  destructive or irreversible actions, runs expected over ~1 hour,
  and anything off-plan.
- After every push: confirm final CI status, then post a Chinese
  summary (what landed, the numbers, what's next).
- Milestone landings sync ONBOARDING §七 and the roadmap in the same
  commit; this plan is updated when reality diverges from it.
