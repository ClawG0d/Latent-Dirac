@AGENTS.md

# Claude session notes

AGENTS.md (imported above) is the single source of truth for physics
rules, engineering rules, scope, and the current phase. This file adds
only Claude-Code-specific operational notes.

## Commands

- tests: `.venv/bin/python -m pytest -q`
- lint: `.venv/bin/python -m ruff check .` (run `ruff format` on new files
  only; several pre-existing files are not format-clean by choice)
- demo assets: `.venv/bin/python tools/generate_hero_3d_webp.py` and
  `.venv/bin/python tools/generate_scene_demo_webps.py` (44 frames;
  extract a late frame and inspect visually before committing assets)
- CLI: `.venv/bin/latent-dirac run|render <scene.yaml>`
- JAX tests enable x64 inside the test files; this machine is CPU-only —
  never produce performance numbers here (honesty discipline requires
  hardware-labeled numbers from Linux CUDA)

## Environments

- This Mac is the planning/dev box. The test/run box is the owner's
  Windows machine inside WSL2 (same `.venv/bin/...` paths); a
  Windows-native checkout is also kept (`.venv/Scripts/...`). The two
  checkouts are independent clones — never share one working tree via
  `/mnt/c`.
- In WSL the repo must live on ext4 (under `~`), never `/mnt/c`: the
  vendored 17k-file tree crawls over 9P.
- GPU: RTX 5070 Ti on the WSL2 box (consumer Blackwell — the float32
  dimensionless-kernel lane is the fit; float64 truth stays on the CPU
  NumPy reference). Official performance numbers come only from that
  box and carry full labels: GPU model + WSL2 + CUDA/driver versions +
  integrator, timestep, particle count, batch size, fidelity tier.
- Because the Windows-native checkout exists, the vendored `CHANGELOG`
  stays a regular file (zip semantics; upstream has it as a symlink) —
  do not "fix" it. See the engine positioning spec addendum.

## Vendored Geant4 tree

- `geant4-v11.4.2/` is read-only vendored upstream code (vanilla
  v11.4.2). Never edit, lint, format, or run tests over it.
- Exclude it from Grep/Glob searches by default — it holds 17k+ files;
  a plain `du`/`find` across it has hit the 2-minute command timeout.
- ruff (`extend-exclude`), pytest (`testpaths`), and packaging
  (`MANIFEST.in prune`) are already configured around it; keep all three
  exclusions in place.
- Commits that touch the vendored tree use the `vendor:` prefix and must
  keep it byte-identical to the upstream release (`.gitattributes
  -text` prevents EOL rewriting; known exception: the vendored
  `CHANGELOG` stays a regular file — see Environments above).

## Workflow (every feature)

1. Design spec to `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`
2. TDD: write the failing test first and confirm it fails for the right
   reason
3. Full `pytest -q` + `ruff check .` green
4. `superpowers:code-reviewer` review before committing — it has caught
   real physics bugs (e.g. a silent antiproton sign error) in 7/7 runs;
   do not skip
5. Conventional commit (`feat:` / `chore:` / `refactor!:` / `docs:`) with
   the Co-Authored-By trailer; then `git fetch` + rebase onto
   `origin/master`, re-run `pytest -q`, and push (see Collaboration —
   every push this era has started out behind)
6. After pushing, confirm the run's final CI status
   (`gh run list --repo ClawG0d/Latent-Dirac --limit 1`, or the Actions
   REST API if `gh` is not installed on the box) — the WSL venv is
   Python 3.14, but the matrix tests 3.10–3.14 (a `tomllib` import once
   broke only the 3.10 jobs and went unnoticed for five pushes)

## Collaboration (multiple agents on one master)

Up to three people work this repo simultaneously, each in their own
Claude Code session, all pushing to `master` (no feature-branch flow
yet). Assume the remote moved since you last looked.

- **Fetch before you push, always.** `git fetch origin` then rebase
  onto `origin/master` — never merge (keep history linear). If the push
  is rejected, fetch and rebase again; the remote may have moved twice
  in one work burst. This happened five times in one afternoon — it is
  the norm, not the exception.
- **Push small and often.** A large local pile of commits turns every
  rebase into a minefield. Land each reviewed feature immediately.
- **Rebase, then re-run the gates.** After a rebase, run
  `pytest -q` + `ruff check .` again before pushing — someone else's
  merged change (e.g. a new fidelity tier, a moved section) can break
  your green even with no textual conflict.
- **Append-heavy shared files collide constantly:** `CHANGELOG.md`,
  `docs/roadmap.md`, `AGENTS.md` "Next:" line, and `README.md`. On a
  conflict, KEEP BOTH sides' entries — a conflict here almost always
  means two real features landed, not that one overwrote the other.
- **README is owner-edited out-of-band** (direct GitHub web edits,
  including structural rewrites). When rebasing a README conflict,
  reconcile against the NEW structure on `origin/master`, do not blindly
  keep your HEAD version. Re-check `tests/test_project_positioning.py`
  after (the honesty gates parse the README).
- **Same-day spec filenames collide:** specs are
  `YYYY-MM-DD-<topic>-design.md`; two sessions on the same day can pick
  the same date — glob `docs/superpowers/specs/` first and keep topics
  distinct.
- **Avoid two sessions in the same module.** Coordinate ownership with
  the human before starting; the load-bearing mirror pairs
  (`differentiable.py` ↔ `jax_scene.py`) and the three-place safety
  pinning are especially painful to resolve blind.
- **When a component graduates** (placeholder → real, planned → shipped),
  flip its per-component assertion in `test_adapter_status_matches_roadmap`
  and update its status in the *same commit* across the README Solvers
  table, `docs/roadmap.md`, and `CHANGELOG.md`. `ALLOWED_ADAPTERS` stays
  fixed — only the real-vs-placeholder assertion moves. Out-of-sync
  status is how a shipped feature reads as "not implemented" 40 lines
  from where it reads as "done".
- **Keep `ONBOARDING.md` current when a milestone lands** — it is the
  relay handoff, and a stale "next up" bullet sends the next session to
  redo finished work (fold your milestone into §二 and flip its §七
  bullet from upcoming to done).
- CI is the shared backstop: after your push settles, confirm the final
  status (step 6 above) — a red master blocks everyone, so fix or
  revert promptly rather than leaving it for the next person.

## Traps (learned the hard way)

- The safety-scope exclusion bullets are pinned in **three** places:
  `EXPECTED_EXCLUSIONS` in `tests/test_project_positioning.py` (exact
  tuple equality) plus verbatim-identical copies in
  `docs/safety_scope.md` and `AGENTS.md`. Change one, change all three
  — and scope changes require a positioning spec first. The README
  deliberately does NOT mirror the list (owner decision, 2026-07-05);
  it only links to docs/safety_scope.md — do not re-add the section.
- Before editing README wording, read `tests/test_project_positioning.py`
  (comparative performance phrases without a benchmark reference fail CI).
- `latent_dirac/backends/differentiable.py` mirrors the element loop in
  `jax_scene._make_simulator` — new element types must be added to both.
- JAX field functions have the signature `(jnp, positions, time_s,
  params)`; the Boris kernel stays a pure, `xp`-generic function.
- State containers are pytree dataclasses, never pydantic (pydantic owns
  the Model/scene layer only); kernels work in dimensionless u = p/(m c),
  SI exists only at State boundaries (float32 + SI momenta underflow).
