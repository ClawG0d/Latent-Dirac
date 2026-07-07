"""Tests for the buffer-gas cross-section table loader (T5, slice 1).

Positron cross sections are curated per publication with provenance (no
open database, no invented values — see the buffer-gas design spec). The
loader enforces that discipline: a provenance header, a monotone energy
grid, non-negative cross sections, and a citable source for real
(table-based) data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from latent_dirac.collisions.cross_sections import (
    CrossSectionTable,
    load_cross_sections,
    sigma,
)

TOY = Path(__file__).resolve().parents[1] / "examples/data/cross_sections/n2_positron_toy.csv"

_VALID = """\
# gas: N2
# fidelity_tier: parameterized
# source: synthetic placeholder for a test
# doi:
# method: hand-authored
# energy_unit: eV
# sigma_unit: m^2
# channels: elastic,electronic,positronium
# thresholds_ev: elastic=0.0,electronic=8.5,positronium=8.8
energy_eV,elastic,electronic,positronium
0.1,5.0e-20,0.0,0.0
9.0,4.0e-20,3.0e-21,1.0e-21
20.0,3.0e-20,5.0e-21,2.0e-21
"""


def _write(tmp_path, text) -> str:
    p = tmp_path / "xs.csv"
    p.write_text(text)
    return str(p)


def test_loads_the_committed_toy_table():
    table = load_cross_sections(str(TOY))
    assert isinstance(table, CrossSectionTable)
    assert table.provenance["gas"] == "N2"
    assert table.fidelity_tier == "parameterized"  # synthetic toy, not real data
    assert "electronic" in table.channels and "positronium" in table.channels
    assert list(table.energies_ev) == sorted(table.energies_ev)  # monotone
    assert table.thresholds_ev["electronic"] == pytest.approx(8.5)


def test_sigma_interpolates_and_is_zero_outside_the_grid(tmp_path):
    table = load_cross_sections(_write(tmp_path, _VALID))
    # midpoint between 9.0 (3.0e-21) and 20.0 (5.0e-21) for electronic
    mid = sigma(table, "electronic", 14.5)
    assert 3.0e-21 < mid < 5.0e-21
    # below/above the tabulated range -> 0 (no extrapolated guess)
    assert sigma(table, "electronic", 0.01) == 0.0
    assert sigma(table, "electronic", 500.0) == 0.0


def test_rejects_a_missing_required_header_key(tmp_path):
    bad = _VALID.replace("# gas: N2\n", "")
    with pytest.raises(ValueError, match="gas"):
        load_cross_sections(_write(tmp_path, bad))


def test_rejects_a_non_monotone_energy_grid(tmp_path):
    bad = _VALID.replace("20.0,3.0e-20", "5.0,3.0e-20")  # 0.1, 9.0, 5.0 — not increasing
    with pytest.raises(ValueError, match="increasing|monoton"):
        load_cross_sections(_write(tmp_path, bad))


def test_rejects_a_negative_cross_section(tmp_path):
    bad = _VALID.replace("9.0,4.0e-20,3.0e-21,1.0e-21", "9.0,4.0e-20,-3.0e-21,1.0e-21")
    with pytest.raises(ValueError, match="negative|>= 0|non-negative"):
        load_cross_sections(_write(tmp_path, bad))


def test_table_based_tier_requires_a_doi(tmp_path):
    # a dataset claiming to be real must be citable
    bad = _VALID.replace("# fidelity_tier: parameterized", "# fidelity_tier: table-based")
    with pytest.raises(ValueError, match="doi|citable|table-based"):
        load_cross_sections(_write(tmp_path, bad))


def test_rejects_thresholds_that_do_not_match_channels(tmp_path):
    # every declared channel must carry a threshold (else a cooling channel
    # silently removes zero energy in the operator)
    bad = _VALID.replace(
        "# thresholds_ev: elastic=0.0,electronic=8.5,positronium=8.8",
        "# thresholds_ev: elastic=0.0,electronic=8.5",  # positronium missing
    )
    with pytest.raises(ValueError, match="threshold"):
        load_cross_sections(_write(tmp_path, bad))


def test_rejects_duplicate_channel_names(tmp_path):
    bad = _VALID.replace(
        "# channels: elastic,electronic,positronium",
        "# channels: elastic,elastic,positronium",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_cross_sections(_write(tmp_path, bad))


def test_rejects_a_ragged_numeric_row(tmp_path):
    bad = _VALID.replace("9.0,4.0e-20,3.0e-21,1.0e-21", "9.0,4.0e-20,3.0e-21")  # short a column
    with pytest.raises(ValueError, match="columns"):
        load_cross_sections(_write(tmp_path, bad))


def test_rejects_nonzero_cross_section_below_threshold(tmp_path):
    # electronic threshold is 8.5 eV; a nonzero electronic sigma at 0.1 eV is
    # unphysical (the channel is not accessible below its threshold)
    bad = _VALID.replace("0.1,5.0e-20,0.0,0.0", "0.1,5.0e-20,7.0e-21,0.0")
    with pytest.raises(ValueError, match="threshold|below"):
        load_cross_sections(_write(tmp_path, bad))
