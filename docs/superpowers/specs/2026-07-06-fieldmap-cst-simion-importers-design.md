# Field-map importers: CST and SIMION (T3)

Date: 2026-07-06
Status: implementation spec. Builds on
`2026-07-05-fieldmap-import-design.md` (the `FieldMapField` regular-grid
container + the COMSOL CSV importer `load_comsol_grid_csv`, both in
`latent_dirac/fields/field_map.py`). Owner decision (execution-plan
spec, 2026-07-06): T3 is the Mac lane's task after T5 — CST/SIMION
field-map importers.

## What ships

Two new programmatic loaders in `latent_dirac/fields/field_map.py`,
each returning a `FieldMapField` exactly like `load_comsol_grid_csv`.
Field maps are **not scene elements yet** (the COMSOL importer isn't
either — it is used programmatically, e.g. the magnetic-mirror demo), so
this task adds no scene schema / `build.py` dispatch / JAX-backend /
mirror-pair changes. NumPy-only, table-based fidelity tier (the tier the
container already carries).

Sliced, each a reviewed commit:

### Slice 1 — `load_cst_ascii(path) -> FieldMapField`

Targets the **CST "Export Plot Data (ASCII)" 3D regular-grid field
export** (also the "ASCII field import" format CST itself accepts). The
document is a label line, a dashed separator line, then whitespace- (or
comma-) delimited numeric rows:

```
x [mm]  y [mm]  z [mm]  ExRe [V/m]  EyRe [V/m]  EzRe [V/m]
---------------------------------------------------------
-1.0  -1.0  -1.0  1.2e2  0.0  3.4e1
...
```

Parsing (robust, header-driven — do not assume column order):

- **Parse the label line.** Each label is `NAME [UNIT]`. Classify:
  coordinate columns by name `x|y|z`; field columns by name matching
  `^([EHB])([xyz])(Re|Im)?$` (case-insensitive). Everything else is
  ignored (CST sometimes appends magnitude/abs columns).
- **Coordinate units → meters.** Read the bracket unit per coordinate
  column; convert with a fixed factor map (`m`=1, `mm`=1e-3, `cm`=1e-2,
  `um`/`µm`=1e-6, `nm`=1e-9). Unlike COMSOL (which is SI and *rejects*
  non-SI), CST exports are natively mm, so the CST loader **converts**.
  An unrecognized length unit is a clear error.
- **Field quantity → SI on the container.** Group field columns by
  quantity letter. `E` (V/m) → `E_v_m`; `B` (T) → `B_t`; `H` (A/m) →
  `B_t = mu_0 * H`. Field unit is read from the bracket and converted
  (`V/m`, `kV/m`, `T`, `mT`, `A/m`). Exactly one of E-or-(B|H) families
  must be present with all three components; the missing family is left
  at its container default (E defaults to zero, B required — if only E
  is present, `B_t` is an explicit zero grid).
- **Complex → real part.** If `*Re`/`*Im` columns are present, import the
  `*Re` values (static tracking uses the real, in-phase field); a plain
  `Ex`/`Bx` (no Re/Im) is taken as-is. Log nothing invented — the
  imaginary part is simply dropped, documented in the docstring.
- **Grid reconstruction.** Reuse the COMSOL path's logic (unique sorted
  coordinates per axis → regular grid; incomplete/duplicate detection).
  Extract that logic into a shared helper `_field_map_from_rows(coords_m,
  field_by_quantity)` so both importers share one grid builder.
- **Errors** carry line numbers and say what was expected (missing
  coordinate/field columns, unknown unit, incomplete grid, non-numeric).

### Slice 2 — `load_simion_patxt(path) -> FieldMapField`

SIMION potential arrays store a **scalar potential** φ per grid point
(electrostatic in volts, magnetic in "mags") plus an electrode flag, not
a field vector (simion.com/info/potential_array_types.html). So the
importer reads φ on the grid and returns the **field** `E_v_m =
-∇φ` (central differences interior, one-sided at faces), `B_t = 0` for an
electrostatic array (a magnetic array maps to `B_t` analogously). Handles
the `.patxt` ASCII header (dimensions nx/ny/nz, grid spacing, 2D-vs-3D,
symmetry/mirroring) and the electrode-flagged body.

**Format-verification gate (honesty):** the exact `.patxt` header layout
was not fully pinned from public docs during design. Slice 2 begins by
confirming the format against the SIMION documentation's format section
(or a real exported `.patxt`); the parser is written to that confirmed
spec, not a guess. If the format cannot be confirmed from an
authoritative source, slice 2 pauses for an owner dialog rather than
shipping a guessed parser. Binary `.pa` is out of scope (ASCII only).

## Refactor (slice 1)

`load_comsol_grid_csv` and `load_cst_ascii` share the
rows→regular-grid→`FieldMapField` step. Extract
`_field_map_from_rows(coords_m, values_by_quantity)` and route COMSOL
through it too (its behavior and error messages preserved, pinned by the
existing tests).

## Tests

Inline hand-authored fixture strings (the COMSOL tests do the same — a
format-accurate sample, not a proprietary export). New
`tests/test_field_map_cst.py` (and `_simion.py` for slice 2):

- CST: builds the expected grid; mm→m conversion is exact; an `H [A/m]`
  export becomes `B_t = mu_0*H`; a `kV/m` E export scales; complex
  `*Re/*Im` imports the real part; shuffled rows + whitespace-vs-comma
  both parse; rejection paths (missing a field component, unknown unit,
  incomplete grid, non-numeric, no data rows). A round-trip test:
  sample a known uniform/linear field into a CST-format string, import,
  and assert the interpolated field matches (rtol 1e-12), mirroring
  `test_uniform_field_map_matches_uniform_field_transport`.
- The COMSOL tests must stay green through the refactor (regression).

## Non-goals

- No scene-element wiring (field maps stay programmatic until a
  field-map scene element is specified — same status as COMSOL).
- No JAX-backend support (grid interpolation is not a static field fn;
  deferred with the rest of the field-map JAX work).
- No RF / time-dependent fields, no tolerance-based grid snapping, no
  SIMION binary `.pa`.

## Honesty

No invented physics values: fixtures are hand-authored *format* samples,
and the loaders only reshape and unit-convert real input. The targeted
CST/SIMION format variants are named in the docstrings with their
references. `FieldMapField` stays table-based. Docs updated:
`docs/physics_scope.md` (CST/SIMION now importable),
`2026-07-05-fieldmap-import-design.md` (mark the follow-ups landing),
CHANGELOG.
