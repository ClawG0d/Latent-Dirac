# FieldMap Import Design (Spec 2d)

## Objective

Add a table-based field model that carries externally computed
electromagnetic fields on a regular grid, plus an importer for the COMSOL
regular-grid spreadsheet export. This connects Latent Dirac to the
industry's de facto exchange interface: design-stage FEM tools (COMSOL,
CST, Opera) export field maps; tracking codes read them.

This spec revises the non-goals of
`2026-07-04-field-model-library-design.md`: field-map interpolation is
implemented here, as its own phase.

## Scope

- `FieldMapField` in `latent_dirac/fields/field_map.py`: strictly
  increasing grid axes `x_m`, `y_m`, `z_m`, magnetic values
  `B_t` shaped `(nx, ny, nz, 3)`, optional electric values `E_v_m` with the
  same shape. Fidelity tier: **table-based**.
- Trilinear interpolation inside the grid; zero field outside the grid
  bounds (documented behavior, matching the hard-edge convention of the
  analytic models).
- Importer `load_comsol_grid_csv(path)`: COMSOL spreadsheet-style export
  with `%`-prefixed header lines and rows `x, y, z, Bx, By, Bz`
  (comma- or whitespace-separated, SI units). Rows may arrive in any
  order; the grid must be complete (`nx * ny * nz` rows) or the importer
  raises.
- Field-magnitude 3D rendering helper `render_field_magnitude_3d` in
  `latent_dirac/viz/field_3d.py` (viz extra; Plotly volume rendering of
  |B| with the table-based fidelity label in the hover/annotation).

## Validation

- Trilinear interpolation is exact for multilinear fields sampled on the
  grid (checked off-grid against the analytic expression).
- Queries outside the bounds return zero; single-position `(3,)` and batch
  `(N,3)` shapes both work (solver contract).
- Axes must be strictly increasing; shape mismatches are rejected.
- Importer round-trip: a synthetic COMSOL-style document parses into the
  expected grid; incomplete grids and malformed rows raise `ValueError`.
- Transport integration: a `FieldMapField` sampled from a uniform field
  reproduces `UniformField` Boris trajectories.
- Rendering smoke test behind `pytest.importorskip("plotly")`.

## Non-Goals

- CST ASCII importer: landed 2026-07-06 as T3 slice 1 (`load_cst_ascii`);
  SIMION `.PA`/`.patxt` importer is T3 slice 2. See
  `2026-07-06-fieldmap-cst-simion-importers-design.md`.
- RF/time-dependent field maps
- cylindrical `(r, z)` maps (later; the 3D regular grid comes first)
- scene-element integration of field maps (follow-up alongside 2c)
