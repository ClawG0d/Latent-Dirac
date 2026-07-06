# xsuite_lattice scene element design (T2)

Date: 2026-07-06
Status: adopted (Mac-side task T2 per TASK-SPLIT.md)

## Decision

Wire the already-real Xsuite adapter into the declarative scene layer: a
new `xsuite_lattice` element tracks the cloud through an `xtrack.Line`
declared in the scene. The adapter body
(`latent_dirac/adapters/xsuite/adapter.py`) is **not touched** — schema
+ loader + build + viz wiring only, the Lattice-component parallel to
T1's Matter `matter_slab`.

## Line reference and reference momentum (the design points)

Unlike the Matter transformer binary (machine-specific → env var), an
`xtrack.Line` is a **data artifact**, so it is referenced by path in the
scene like the COMSOL field map and the yield table:

- `line_path`: path to an xtrack Line JSON (`Line.to_json` output),
  resolved **relative to the scene file** (cwd-safe), extending the same
  `_resolve_scene_relative_paths` mechanism that already handles the
  source `table_path`.
- `p0c_ev`: the reference momentum × c in eV — a physics choice, always
  explicit in the scene (the adapter forbids an implicit reference).
- `num_turns`: optional, default 1.

## Constructability without xtrack

`xtrack` is an in-process dependency with no stub seam (unlike the
Matter engine's subprocess). So:

- `scene_from_mapping` / schema validation: always succeeds (no xtrack).
- `run_scene`: building the `xsuite_lattice` stage loads the Line, which
  needs xtrack; a clear `ImportError` (the adapter's `_require_xtrack`
  message, naming the `[xsuite]` extra) is raised when the stage is
  built if xtrack is missing.
- viz: the element carries an optional `length_m` / `center_z_m` viz
  hint so `scene_3d` draws a lattice box with **no** xtrack import; if
  `length_m` is omitted a small marker is drawn (never silently
  invisible — the roadmap-2b rule).

Local-test caveat (TASK-SPLIT §二): the tests require `xtrack` (and the
`xsuite` metapackage for its runtime kernels); they `importorskip` both,
so on a machine without them the T2 tests skip and CI is the backstop.

## Integration points

1. `scene/schema.py`: `XsuiteLatticeElement(type="xsuite_lattice",
   line_path, p0c_ev>0, num_turns>=1 default 1, center_z_m=0.0,
   length_m: float|None gt=0 default None)`, added to `ElementSpec`,
   `extra="forbid"`.
2. `scene/loader.py`: extend `_resolve_scene_relative_paths` to resolve a
   relative `line_path` on any `xsuite_lattice` element against the scene
   directory (same rule as the source `table_path`).
3. `scene/build.py`: `_xsuite_lattice_action(element)` loads
   `xtrack.Line.from_json(line_path)`, builds `ReferenceFrame(p0c_ev)`,
   and returns the action of `xsuite_tracking_stage(...)` (ledger
   stamping flows through `Stage.run` as everywhere).
4. `viz/scene_3d.py`: `xsuite_lattice` box (center_z_m, length_m) or a
   marker; `FIDELITY_LABELS` entry ("fidelity: externally tracked
   (Xsuite / xtrack)").
5. JAX side: zero code — not in `_SWEEPABLE_PARAMS`, so `_base_params`
   rejects it; a test asserts the rejection. The mirror pair is not
   touched.
6. Docs: `docs/scene_schema.md`, README/roadmap/CHANGELOG status sync in
   the same commit.

## Tests (TDD; require xtrack + xsuite, importorskip)

1. A scene with an `xsuite_lattice` (a one-drift Line written to a temp
   JSON) runs end-to-end: survivors' transverse position advances by the
   drift (paraxial handshake, documented tolerance).
2. A Line with a `LimitRect` aperture kills off-axis particles; they are
   ledgered at the lattice's stage index, survivors untouched.
3. `line_path` resolves relative to the scene file: write scene + line
   under a temp dir, `load_scene`, run from a different cwd.
4. Missing line file → a clear error at run; `scene_from_mapping`
   succeeds without the file.
5. Schema: `p0c_ev<=0` and `num_turns<1` rejected; unknown field
   rejected.
6. JAX backend rejects an `xsuite_lattice` scene.
7. `scene_3d` draws the lattice (not silently invisible), with and
   without a `length_m` hint.

## Honesty notes

Fidelity tier: externally tracked (Xsuite / xtrack vX.Y) — the tracking
result carries `metadata["xtrack_version"]`. Gradients stop at this
boundary (transformer form for autodiff, per the solver composition
spec). No performance wording; the drift handshake is a correctness
check with a documented tolerance.

## Coordination

Mac-owned files: `scene/schema.py`, `scene/loader.py`, `scene/build.py`,
`viz/scene_3d.py`, `docs/`. The adapter body and `adapters/xsuite/` stay
off-limits (Windows-owned per TASK-SPLIT §五) — flag the owner if this
turns out to need an adapter change.
