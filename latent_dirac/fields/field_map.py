"""Table-based field model on a regular grid, with a COMSOL-style importer.

Fidelity tier: table-based. Values are trilinearly interpolated inside the
grid; queries outside the grid bounds return zero field, matching the
hard-edge convention of the analytic field models.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from latent_dirac.fields.base import Field


class FieldMapField(BaseModel, Field):
    """Externally computed electromagnetic field sampled on a regular grid.

    Axes are strictly increasing 1D coordinate arrays in meters; magnetic
    values `B_t` are shaped `(nx, ny, nz, 3)` in tesla, and the optional
    electric values `E_v_m` share that shape in V/m.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    B_t: np.ndarray
    E_v_m: np.ndarray | None = None

    @field_validator("x_m", "y_m", "z_m", mode="before")
    @classmethod
    def _as_axis(cls, value, info):
        axis = np.asarray(value, dtype=float)
        if axis.ndim != 1 or axis.size < 2:
            raise ValueError(f"{info.field_name} must be a 1D axis with at least 2 points")
        if not np.isfinite(axis).all():
            raise ValueError(f"{info.field_name} must contain only finite values")
        if not np.all(np.diff(axis) > 0.0):
            raise ValueError(f"{info.field_name} must be strictly increasing")
        return axis

    @field_validator("B_t", "E_v_m", mode="before")
    @classmethod
    def _as_values(cls, value, info):
        if value is None:
            return None
        values = np.asarray(value, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"{info.field_name} must contain only finite values; COMSOL exports "
                "write NaN for grid points outside the modeled geometry - crop the "
                "export region or replace those values explicitly before import"
            )
        return values

    @model_validator(mode="after")
    def _validate_shapes(self):
        expected = (self.x_m.size, self.y_m.size, self.z_m.size, 3)
        if self.B_t.shape != expected:
            raise ValueError(f"B_t must have shape {expected}, got {self.B_t.shape}")
        if self.E_v_m is not None and self.E_v_m.shape != expected:
            raise ValueError(f"E_v_m must have shape {expected}, got {self.E_v_m.shape}")
        return self

    def E(self, x, t) -> np.ndarray:
        if self.E_v_m is None:
            x_array = np.asarray(x, dtype=float)
            return np.zeros(3) if x_array.ndim == 1 else np.zeros_like(x_array)
        return self._interpolate(self.E_v_m, x)

    def B(self, x, t) -> np.ndarray:
        return self._interpolate(self.B_t, x)

    def _interpolate(self, values: np.ndarray, x) -> np.ndarray:
        points = np.asarray(x, dtype=float)
        single = points.ndim == 1
        query = np.atleast_2d(points)
        result = np.zeros((query.shape[0], 3))

        inside = (
            (query[:, 0] >= self.x_m[0])
            & (query[:, 0] <= self.x_m[-1])
            & (query[:, 1] >= self.y_m[0])
            & (query[:, 1] <= self.y_m[-1])
            & (query[:, 2] >= self.z_m[0])
            & (query[:, 2] <= self.z_m[-1])
        )
        if np.any(inside):
            p = query[inside]
            ix, fx = _locate(self.x_m, p[:, 0])
            iy, fy = _locate(self.y_m, p[:, 1])
            iz, fz = _locate(self.z_m, p[:, 2])

            blend = np.zeros((p.shape[0], 3))
            for dx, wx in ((0, 1.0 - fx), (1, fx)):
                for dy, wy in ((0, 1.0 - fy), (1, fy)):
                    for dz, wz in ((0, 1.0 - fz), (1, fz)):
                        weight = (wx * wy * wz)[:, np.newaxis]
                        blend += weight * values[ix + dx, iy + dy, iz + dz]
            result[inside] = blend

        return result[0] if single else result


def _locate(axis: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = np.clip(np.searchsorted(axis, coords, side="right") - 1, 0, axis.size - 2)
    fraction = (coords - axis[index]) / (axis[index + 1] - axis[index])
    return index, fraction


_NON_SI_UNIT_PATTERN = re.compile(r"\((mm|cm|um|µm|nm|in|inch|ft)\)", re.IGNORECASE)


def load_comsol_grid_csv(path: str | Path) -> FieldMapField:
    """Load a COMSOL spreadsheet-style regular-grid export.

    Expected document: `%`-prefixed header lines followed by data rows
    `x, y, z, Bx, By, Bz` (comma- or whitespace-separated, SI units: meters
    and tesla). Rows may arrive in any order, but the grid must be complete.

    Limitation: grid axes are detected by exact coordinate match. Exports
    with per-row coordinate jitter will be rejected as incomplete grids;
    tolerance-based grid snapping is deliberately out of scope.
    """

    rows: list[list[float]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            unit_match = _NON_SI_UNIT_PATTERN.search(stripped)
            if unit_match:
                raise ValueError(
                    f"line {line_number}: header declares non-SI length unit "
                    f"{unit_match.group(0)!r}; re-export in SI units (meters, tesla) "
                    "or rescale the file before import"
                )
            continue
        parts = stripped.split(",") if "," in stripped else stripped.split()
        if len(parts) != 6:
            raise ValueError(
                f"line {line_number}: expected 6 columns (x, y, z, Bx, By, Bz), got {len(parts)}"
            )
        try:
            rows.append([float(part) for part in parts])
        except ValueError as exc:
            raise ValueError(f"line {line_number}: non-numeric value in {stripped!r}") from exc

    if not rows:
        raise ValueError("no data rows found")

    data = np.asarray(rows)
    return _field_map_from_rows(data[:, :3], {"B": data[:, 3:6]})


def _field_map_from_rows(coords_m, values_by_quantity: dict) -> FieldMapField:
    """Assemble a FieldMapField from scattered rows on a regular grid.

    `coords_m` is `(N, 3)` in meters; `values_by_quantity` maps `"B"` (tesla)
    and/or `"E"` (V/m) to `(N, 3)` component arrays. Axes are detected by
    exact coordinate match (no tolerance snapping). Shared by every importer.
    """
    coords = np.asarray(coords_m, dtype=float)
    axes = [np.unique(coords[:, column]) for column in range(3)]
    grid_shape = tuple(axis.size for axis in axes)
    if int(np.prod(grid_shape)) != coords.shape[0]:
        raise ValueError(
            f"incomplete regular grid: {grid_shape} needs {int(np.prod(grid_shape))} rows, "
            f"got {coords.shape[0]}. Axes are detected by exact coordinate match; "
            "coordinate jitter in the export can inflate the axis count"
        )

    indices = [np.searchsorted(axes[column], coords[:, column]) for column in range(3)]
    filled = np.zeros(grid_shape, dtype=bool)
    filled[indices[0], indices[1], indices[2]] = True
    if not filled.all():
        raise ValueError("incomplete regular grid: duplicate rows leave grid cells unset")

    grids: dict[str, np.ndarray] = {}
    for quantity, values in values_by_quantity.items():
        grid = np.zeros((*grid_shape, 3))
        grid[indices[0], indices[1], indices[2]] = np.asarray(values, dtype=float)
        grids[quantity] = grid

    b_values = grids.get("B")
    if b_values is None:
        b_values = np.zeros((*grid_shape, 3))  # E-only export: explicit zero B grid
    return FieldMapField(
        x_m=axes[0], y_m=axes[1], z_m=axes[2], B_t=b_values, E_v_m=grids.get("E")
    )


# --- CST "Export Plot Data (ASCII)" 3D regular-grid field export ---
# Format: a `NAME [UNIT]` label line, a dashed separator, then numeric rows.
# Columns are classified by label (order-independent). See
# docs/superpowers/specs/2026-07-06-fieldmap-cst-simion-importers-design.md.
_CST_LABEL = re.compile(r"(\S+)\s*\[\s*([^\]]+?)\s*\]")
_CST_COMPONENT = re.compile(r"^([EHB])([xyz])(re|im)?$", re.IGNORECASE)
# Coordinate (length) units have no case collisions, so they are matched
# case-insensitively. Field units are matched CASE-SENSITIVELY: SI prefix case
# is load-bearing (mV/m milli vs MV/m mega, mT milli vs T), and folding case
# would silently apply a 10^9x-wrong factor.
_LENGTH_UNIT_FACTORS = {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "um": 1e-6, "µm": 1e-6, "nm": 1e-9}
_FIELD_UNIT_FACTORS = {
    "V/m": 1.0, "kV/m": 1e3, "MV/m": 1e6, "mV/m": 1e-3,
    "T": 1.0, "mT": 1e-3, "kT": 1e3, "Gauss": 1e-4, "gauss": 1e-4, "G": 1e-4,
    "A/m": 1.0, "kA/m": 1e3,
}


def _mu_0() -> float:
    from latent_dirac.core.constants import (
        SPEED_OF_LIGHT_M_PER_S,
        VACUUM_PERMITTIVITY_F_PER_M,
    )

    return 1.0 / (VACUUM_PERMITTIVITY_F_PER_M * SPEED_OF_LIGHT_M_PER_S**2)


def load_cst_ascii(path: str | Path) -> FieldMapField:
    """Load a CST "Export Plot Data (ASCII)" 3D regular-grid field export.

    The document is a `NAME [UNIT]` label line, a dashed separator, then
    whitespace- or comma-separated numeric rows. Columns are classified by
    label (order-independent): coordinates `x|y|z`, field components matching
    `[EHB][xyz](Re|Im)?`. Coordinate units are converted to meters and field
    units to SI; an `H` (A/m) family becomes `B = mu_0 * H` (tesla). Complex
    exports keep the real (`*Re`) part; the imaginary part is dropped. An `E`
    export fills `E_v_m` with a zero `B_t`; a `B`/`H` export fills `B_t`.

    Reference: CST ASCII field import/export format. Fidelity tier:
    table-based (values are the imported field, only reshaped and
    unit-converted).
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    # The header is the first line that labels all three x/y/z coordinates
    # (a bare ">= 3 bracket tokens" test is hijacked by CST metadata preambles
    # like "Field: E [V/m] at f=1 [GHz], port [1]").
    columns = None
    header_index = 0
    for index, line in enumerate(lines):
        matches = _CST_LABEL.findall(line)
        names = {name.strip().lower() for name, _ in matches}
        if {"x", "y", "z"} <= names:
            columns, header_index = matches, index
            break
    if columns is None:
        raise ValueError(
            "no CST header line with x/y/z coordinate columns found (expected a "
            "'x [unit]  y [unit]  z [unit]  ...' label line)"
        )

    coord_column: dict[str, int] = {}
    coord_factor: dict[str, float] = {}
    field_specs: list[tuple[str, str, str, float, int]] = []
    for column, (name, unit) in enumerate(columns):
        key = name.strip().lower()
        if key in ("x", "y", "z"):
            length_key = unit.strip().lower()
            if length_key not in _LENGTH_UNIT_FACTORS:
                raise ValueError(f"unknown length unit {unit!r} for coordinate {name!r}")
            coord_column[key] = column
            coord_factor[key] = _LENGTH_UNIT_FACTORS[length_key]
            continue
        match = _CST_COMPONENT.match(name.strip())
        if match:
            field_key = unit.strip()  # case-sensitive: mV/m (milli) != MV/m (mega)
            if field_key not in _FIELD_UNIT_FACTORS:
                raise ValueError(
                    f"unknown or case-ambiguous field unit {unit!r} for column {name!r}"
                )
            quantity, component, part = match.group(1).upper(), match.group(2).lower(), match.group(3)
            field_specs.append(
                (quantity, component, (part or "re").lower(), _FIELD_UNIT_FACTORS[field_key], column)
            )
        # any other labelled column (magnitude, abs, ...) is ignored

    for axis in ("x", "y", "z"):
        if axis not in coord_column:
            raise ValueError(f"CST header missing the {axis!r} coordinate column")

    n_columns = len(columns)
    row_values: list[list[float]] = []
    for line_number, line in enumerate(lines[header_index + 1 :], start=header_index + 2):
        stripped = line.strip()
        if not stripped or set(stripped) <= set("-—= \t"):
            continue  # blank or a dashed/decorative separator line
        parts = stripped.split(",") if "," in stripped else stripped.split()
        if len(parts) != n_columns:
            raise ValueError(
                f"line {line_number}: expected {n_columns} columns to match the header, "
                f"got {len(parts)}"
            )
        try:
            row_values.append([float(part) for part in parts])
        except ValueError as exc:
            raise ValueError(f"line {line_number}: non-numeric value in {stripped!r}") from exc

    if not row_values:
        raise ValueError("no data rows found")
    table = np.asarray(row_values)

    coords_m = np.column_stack(
        [table[:, coord_column[axis]] * coord_factor[axis] for axis in ("x", "y", "z")]
    )

    collected: dict[tuple[str, str], tuple[np.ndarray, str]] = {}
    saw_real = False
    for quantity, component, part, factor, column in field_specs:
        if part != "re":
            continue  # drop the imaginary part; static tracking uses the in-phase field
        saw_real = True
        target = "E" if quantity == "E" else "B"
        key = (target, component)
        if key in collected:
            previous = collected[key][1]
            if previous == quantity:
                raise ValueError(f"CST header has duplicate {quantity}{component} columns")
            raise ValueError(
                f"CST header declares both {previous} and {quantity} for the "
                f"{target}-field {component}-component; keep only one (B != mu_0*H in material)"
            )
        values = table[:, column] * factor
        if quantity == "H":
            values = values * _mu_0()
        collected[key] = (values, quantity)

    if field_specs and not saw_real:
        raise ValueError("CST field has only imaginary (*Im) components; no real part to import")

    families: dict[str, np.ndarray] = {}
    for target in ("E", "B"):
        present = [c for c in ("x", "y", "z") if (target, c) in collected]
        if not present:
            continue
        if len(present) != 3:
            missing = [c for c in ("x", "y", "z") if (target, c) not in collected]
            raise ValueError(f"CST {target}-field is missing component(s): {missing}")
        families[target] = np.column_stack([collected[(target, c)][0] for c in ("x", "y", "z")])

    if not families:
        raise ValueError("no recognized E/B/H field columns in the CST header")

    return _field_map_from_rows(coords_m, families)
