"""Cross-section table format + loader for buffer-gas collisions.

Positron scattering cross sections are not in a single open database
(LXCat is electron-only), so each table is curated per publication and
carries its provenance in a header — the cross-section analogue of the
Geant4 four-tuple. The loader enforces that discipline and never
extrapolates: a positron outside the tabulated energy range simply has
no modeled cross section there (returns 0), rather than a guessed value.

CSV format::

    # gas: N2
    # fidelity_tier: parameterized | table-based
    # source: <citation or "synthetic placeholder">
    # doi: <doi>            (required, non-empty, when table-based)
    # method: <experiment/theory/synthetic>   (required when table-based)
    # energy_unit: eV
    # sigma_unit: m^2
    # channels: elastic,electronic,positronium
    # thresholds_ev: elastic=0.0,electronic=8.5,positronium=8.8
    energy_eV,elastic,electronic,positronium
    0.1,5.0e-20,0.0,0.0
    ...
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_REQUIRED_HEADER = (
    "gas", "fidelity_tier", "source", "energy_unit", "sigma_unit", "channels", "thresholds_ev",
)
_TIERS = ("parameterized", "table-based")


@dataclass(frozen=True)
class CrossSectionTable:
    """A curated set of channel cross sections σ_i(E) on a shared energy grid.

    `energies_ev` is strictly increasing; `channels` maps a channel name to a
    non-negative σ array (m²) aligned to the grid; `thresholds_ev` maps a
    channel to its energy threshold; `provenance` is the parsed header;
    `fidelity_tier` is "parameterized" (synthetic/analytic) or "table-based"
    (a cited dataset).
    """

    energies_ev: np.ndarray
    channels: dict[str, np.ndarray]
    thresholds_ev: dict[str, float]
    provenance: dict[str, str]
    fidelity_tier: str


def _parse_header(lines: list[str]) -> dict[str, str]:
    header: dict[str, str] = {}
    for raw in lines:
        body = raw[1:].strip()  # drop the leading '#'
        if not body:
            continue
        key, _, value = body.partition(":")
        header[key.strip()] = value.strip()
    return header


def _parse_thresholds(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for pair in text.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, _, val = pair.partition("=")
        out[name.strip()] = float(val)
    return out


def load_cross_sections(path: str) -> CrossSectionTable:
    with open(path, encoding="utf-8") as fh:
        raw_lines = fh.read().splitlines()

    header_lines = [ln for ln in raw_lines if ln.lstrip().startswith("#")]
    data_lines = [ln for ln in raw_lines if ln.strip() and not ln.lstrip().startswith("#")]
    header = _parse_header(header_lines)

    missing = [k for k in _REQUIRED_HEADER if not header.get(k)]
    if missing:
        raise ValueError(f"cross-section header missing required key(s): {', '.join(missing)}")

    tier = header["fidelity_tier"]
    if tier not in _TIERS:
        raise ValueError(f"fidelity_tier must be one of {_TIERS}, got {tier!r}")
    if header["energy_unit"] != "eV":
        raise ValueError(f"energy_unit must be 'eV', got {header['energy_unit']!r}")
    if header["sigma_unit"] != "m^2":
        raise ValueError(f"sigma_unit must be 'm^2', got {header['sigma_unit']!r}")
    if tier == "table-based" and not (header.get("doi") and header.get("method")):
        raise ValueError("a table-based (real) dataset must be citable: doi and method are required")

    channel_names = [c.strip() for c in header["channels"].split(",") if c.strip()]
    if len(set(channel_names)) != len(channel_names):
        raise ValueError(f"duplicate channel name(s) in header: {channel_names}")
    thresholds = _parse_thresholds(header["thresholds_ev"])
    if set(thresholds) != set(channel_names):
        raise ValueError(
            "thresholds_ev must declare exactly the channels: "
            f"channels={sorted(channel_names)}, thresholds={sorted(thresholds)}"
        )

    if not data_lines:
        raise ValueError("cross-section table has no numeric rows")
    columns = [c.strip() for c in data_lines[0].split(",")]
    if columns[0] != "energy_eV" or columns[1:] != channel_names:
        raise ValueError(
            f"numeric columns {columns} do not match header channels "
            f"['energy_eV', {channel_names}]"
        )

    n_columns = len(columns)
    rows = [row.split(",") for row in data_lines[1:]]
    for line_no, row in enumerate(rows, start=1):
        if len(row) != n_columns:
            raise ValueError(
                f"cross-section row {line_no} has {len(row)} columns, expected {n_columns}"
            )
    grid = np.array([[float(x) for x in row] for row in rows], dtype=float)
    energies = grid[:, 0]
    if not np.all(np.diff(energies) > 0):
        raise ValueError("energy grid must be strictly increasing")
    if np.any(grid[:, 1:] < 0):
        raise ValueError("cross sections must be non-negative (>= 0)")

    channels = {name: grid[:, i + 1] for i, name in enumerate(channel_names)}

    # A channel with an energy threshold must have sigma == 0 below it: a
    # nonzero sub-threshold value would let the collision operator select a
    # channel a particle cannot physically access.
    for name, sigma_col in channels.items():
        threshold = thresholds[name]
        if threshold > 0.0:
            below = energies < threshold
            if np.any(sigma_col[below] != 0.0):
                raise ValueError(
                    f"channel {name!r} has nonzero cross section below its "
                    f"{threshold} eV threshold; sigma must be 0 for E < threshold"
                )

    return CrossSectionTable(
        energies_ev=energies,
        channels=channels,
        thresholds_ev=thresholds,
        provenance=header,
        fidelity_tier=tier,
    )


def sigma(table: CrossSectionTable, channel: str, energy_ev: float) -> float:
    """σ (m²) for a channel at an energy — linear interpolation, 0 outside range."""
    xs = table.energies_ev
    ys = table.channels[channel]
    return float(np.interp(energy_ev, xs, ys, left=0.0, right=0.0))
