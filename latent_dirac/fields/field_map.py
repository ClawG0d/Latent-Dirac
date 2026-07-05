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
    axes = [np.unique(data[:, column]) for column in range(3)]
    grid_shape = tuple(axis.size for axis in axes)
    if int(np.prod(grid_shape)) != data.shape[0]:
        raise ValueError(
            f"incomplete regular grid: {grid_shape} needs {int(np.prod(grid_shape))} rows, "
            f"got {data.shape[0]}. Axes are detected by exact coordinate match; "
            "coordinate jitter in the export can inflate the axis count"
        )

    indices = [np.searchsorted(axes[column], data[:, column]) for column in range(3)]
    b_values = np.zeros((*grid_shape, 3))
    b_values[indices[0], indices[1], indices[2]] = data[:, 3:6]

    filled = np.zeros(grid_shape, dtype=bool)
    filled[indices[0], indices[1], indices[2]] = True
    if not filled.all():
        raise ValueError("incomplete regular grid: duplicate rows leave grid cells unset")

    return FieldMapField(x_m=axes[0], y_m=axes[1], z_m=axes[2], B_t=b_values)
