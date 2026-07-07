"""Tests for the CST ASCII field-map importer (T3, slice 1).

Targets the CST "Export Plot Data (ASCII)" 3D regular-grid field export:
a `NAME [UNIT]` label line, a dashed separator, then whitespace/comma
rows. Columns are classified by label (order-independent); coordinate
units convert to meters and field units to SI (H -> B via mu_0). Fixtures
are hand-authored *format* samples, not proprietary exports.
"""

from __future__ import annotations

import numpy as np
import pytest

from latent_dirac.core.constants import SPEED_OF_LIGHT_M_PER_S, VACUUM_PERMITTIVITY_F_PER_M
from latent_dirac.fields.field_map import load_cst_ascii

MU_0 = 1.0 / (VACUUM_PERMITTIVITY_F_PER_M * SPEED_OF_LIGHT_M_PER_S**2)

# 2x2x2 grid, coords in mm; a real E field in V/m.
_CST_E_MM = """\
x [mm]  y [mm]  z [mm]  ExRe [V/m]  EyRe [V/m]  EzRe [V/m]
---------------------------------------------------------
0  0  0   10  20  30
0  0  3   11  21  31
0  2  0   12  22  32
0  2  3   13  23  33
1  0  0   14  24  34
1  0  3   15  25  35
1  2  0   16  26  36
1  2  3   17  27  37
"""


def _write(tmp_path, text):
    p = tmp_path / "field.txt"
    p.write_text(text)
    return str(p)


def test_cst_builds_grid_and_converts_mm_to_m(tmp_path):
    fm = load_cst_ascii(_write(tmp_path, _CST_E_MM))
    np.testing.assert_allclose(fm.x_m, [0.0, 1e-3])
    np.testing.assert_allclose(fm.y_m, [0.0, 2e-3])
    np.testing.assert_allclose(fm.z_m, [0.0, 3e-3])
    np.testing.assert_allclose(fm.E_v_m[0, 0, 0], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(fm.E_v_m[1, 1, 1], [17.0, 27.0, 37.0])
    np.testing.assert_allclose(fm.B_t, 0.0)  # E-only export -> zero B grid


def test_cst_h_field_becomes_b_via_mu0(tmp_path):
    doc = _CST_E_MM.replace("ExRe [V/m]", "HxRe [A/m]").replace(
        "EyRe [V/m]", "HyRe [A/m]"
    ).replace("EzRe [V/m]", "HzRe [A/m]")
    fm = load_cst_ascii(_write(tmp_path, doc))
    np.testing.assert_allclose(fm.B_t[0, 0, 0], np.array([10.0, 20.0, 30.0]) * MU_0)
    assert fm.E_v_m is None  # no E family present


def test_cst_kv_per_m_scales(tmp_path):
    doc = _CST_E_MM.replace("[V/m]", "[kV/m]")
    fm = load_cst_ascii(_write(tmp_path, doc))
    np.testing.assert_allclose(fm.E_v_m[0, 0, 0], [10e3, 20e3, 30e3])


def test_cst_complex_imports_real_part(tmp_path):
    # add *Im columns with different values; the loader keeps *Re
    doc = """\
x [mm]  y [mm]  z [mm]  ExRe [V/m]  EyRe [V/m]  EzRe [V/m]  ExIm [V/m]  EyIm [V/m]  EzIm [V/m]
--------------------------------------------------------------------------------------------
0  0  0   10  20  30   -1  -2  -3
0  0  3   11  21  31   -1  -2  -3
0  2  0   12  22  32   -1  -2  -3
0  2  3   13  23  33   -1  -2  -3
1  0  0   14  24  34   -1  -2  -3
1  0  3   15  25  35   -1  -2  -3
1  2  0   16  26  36   -1  -2  -3
1  2  3   17  27  37   -1  -2  -3
"""
    fm = load_cst_ascii(_write(tmp_path, doc))
    np.testing.assert_allclose(fm.E_v_m[0, 0, 0], [10.0, 20.0, 30.0])  # Re, not Im


def test_cst_accepts_shuffled_rows_and_commas(tmp_path):
    lines = _CST_E_MM.splitlines()
    header, sep, rows = lines[0], lines[1], lines[2:]
    shuffled = [header, sep] + list(reversed(rows))
    doc = "\n".join(shuffled).replace("  ", ",").replace(" ", "")
    # rebuild with commas but keep the header readable: simpler — comma-join tokens
    doc = "\n".join([header, sep] + [",".join(r.split()) for r in reversed(rows)])
    fm = load_cst_ascii(_write(tmp_path, doc))
    np.testing.assert_allclose(fm.E_v_m[1, 1, 1], [17.0, 27.0, 37.0])


def test_cst_b_field_in_tesla_round_trips_uniform(tmp_path):
    # a uniform Bz = 0.45 T sampled onto a grid imports and interpolates back
    lines = ["Bx [T]  By [T]  Bz [T]"]
    header = "x [mm]  y [mm]  z [mm]  " + lines[0]
    body = ["-" * 40]
    for xi in (0.0, 5.0):
        for yi in (0.0, 5.0):
            for zi in (0.0, 10.0):
                body.append(f"{xi} {yi} {zi} 0.0 0.0 0.45")
    fm = load_cst_ascii(_write(tmp_path, "\n".join([header, *body])))
    got = fm.B(np.array([1e-3, 1e-3, 1e-3]), 0.0)  # inside the grid, meters
    np.testing.assert_allclose(got, [0.0, 0.0, 0.45], atol=1e-12)


def test_cst_rejects_missing_field_component(tmp_path):
    # only Ex, Ey declared (no Ez) — an incomplete field family
    doc = """\
x [mm]  y [mm]  z [mm]  ExRe [V/m]  EyRe [V/m]
---------------------------------------------
0  0  0   10  20
0  0  3   11  21
0  2  0   12  22
0  2  3   13  23
1  0  0   14  24
1  0  3   15  25
1  2  0   16  26
1  2  3   17  27
"""
    with pytest.raises(ValueError, match="component|Ez|z"):
        load_cst_ascii(_write(tmp_path, doc))


def test_cst_rejects_unknown_length_unit(tmp_path):
    doc = _CST_E_MM.replace("x [mm]", "x [furlong]")
    with pytest.raises(ValueError, match="unit"):
        load_cst_ascii(_write(tmp_path, doc))


def test_cst_rejects_incomplete_grid(tmp_path):
    doc = "\n".join(_CST_E_MM.splitlines()[:-1])  # drop the last data row
    with pytest.raises(ValueError, match="incomplete regular grid"):
        load_cst_ascii(_write(tmp_path, doc))


def test_cst_rejects_non_numeric(tmp_path):
    doc = _CST_E_MM.replace("1  0  0   14  24  34", "1  0  0   xx  24  34")
    with pytest.raises(ValueError, match="numeric"):
        load_cst_ascii(_write(tmp_path, doc))


def test_cst_millivolt_and_megavolt_do_not_collide(tmp_path):
    # SI prefix case is load-bearing: mV/m (milli) vs MV/m (mega)
    milli = load_cst_ascii(_write(tmp_path, _CST_E_MM.replace("[V/m]", "[mV/m]")))
    mega = load_cst_ascii(_write(tmp_path, _CST_E_MM.replace("[V/m]", "[MV/m]")))
    np.testing.assert_allclose(milli.E_v_m[0, 0, 0], [10e-3, 20e-3, 30e-3])
    np.testing.assert_allclose(mega.E_v_m[0, 0, 0], [10e6, 20e6, 30e6])


def test_cst_rejects_both_b_and_h_families(tmp_path):
    # B and H for the same component disagree in material; refuse to pick one
    header = "x [mm]  y [mm]  z [mm]  BxRe [T]  ByRe [T]  BzRe [T]  HxRe [A/m]  HyRe [A/m]  HzRe [A/m]"
    body = ["-" * 40]
    for xi in (0.0, 1.0):
        for yi in (0.0, 1.0):
            for zi in (0.0, 1.0):
                body.append(f"{xi} {yi} {zi} 0.1 0.2 0.3 1 2 3")
    with pytest.raises(ValueError, match="both B and H|keep only one"):
        load_cst_ascii(_write(tmp_path, "\n".join([header, *body])))


def test_cst_rejects_duplicate_component_column(tmp_path):
    header = "x [mm]  y [mm]  z [mm]  ExRe [V/m]  EyRe [V/m]  EzRe [V/m]  ExRe [V/m]"
    body = ["-" * 40]
    for xi in (0.0, 1.0):
        for yi in (0.0, 1.0):
            for zi in (0.0, 1.0):
                body.append(f"{xi} {yi} {zi} 10 20 30 999")
    with pytest.raises(ValueError, match="duplicate"):
        load_cst_ascii(_write(tmp_path, "\n".join([header, *body])))


def test_cst_skips_metadata_preamble_before_the_header(tmp_path):
    # a CST-style metadata line with bracket tokens must not hijack detection
    doc = "Field: E [V/m] at f=1 [GHz], port [1]\n" + _CST_E_MM
    fm = load_cst_ascii(_write(tmp_path, doc))
    np.testing.assert_allclose(fm.E_v_m[0, 0, 0], [10.0, 20.0, 30.0])


def test_cst_rejects_only_imaginary_field(tmp_path):
    doc = _CST_E_MM.replace("ExRe", "ExIm").replace("EyRe", "EyIm").replace("EzRe", "EzIm")
    with pytest.raises(ValueError, match="imaginary"):
        load_cst_ascii(_write(tmp_path, doc))
