"""Apparatus glyph smoke tests: every glyph draws artists without error.

Design record: docs/superpowers/specs/2026-07-07-apparatus-visuals-design.md.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from tools import mpl3d  # noqa: E402


@pytest.fixture()
def axes3d():
    plt, _ = mpl3d.load_matplotlib()
    figure = plt.figure()
    axes = figure.add_subplot(projection="3d")
    yield axes
    plt.close(figure)


def _artist_count(axes) -> int:
    return len(axes.lines) + len(axes.collections)


def test_coil_glyph(axes3d):
    before = _artist_count(axes3d)
    mpl3d.draw_coil(axes3d, radius=0.02, z0=0.0, z1=0.15)
    assert _artist_count(axes3d) > before


def test_dipole_pole_glyph(axes3d):
    before = _artist_count(axes3d)
    mpl3d.draw_dipole_poles(axes3d, [0.0, 0.3, 0.0], z0=0.0, z1=0.1, gap_half=0.01, face_half=0.012)
    assert _artist_count(axes3d) > before
    # B along z has no transverse pole picture: draws nothing, no crash
    neutral = _artist_count(axes3d)
    mpl3d.draw_dipole_poles(axes3d, [0.0, 0.0, 1.0], z0=0.0, z1=0.1, gap_half=0.01, face_half=0.012)
    assert _artist_count(axes3d) == neutral


def test_quadrupole_tip_glyph(axes3d):
    before = _artist_count(axes3d)
    mpl3d.draw_quadrupole_tips(axes3d, z0=0.0, z1=0.1, r0=0.01)
    assert _artist_count(axes3d) > before
    # polarity colors flip with the gradient sign (B = g*(y, x, 0):
    # flux enters the 45 deg tip for g > 0 -> south/cool there)
    plus = axes3d.lines[before].get_color()
    flipped = _artist_count(axes3d)
    mpl3d.draw_quadrupole_tips(axes3d, z0=0.0, z1=0.1, r0=0.01, gradient_sign=-1.0)
    minus = axes3d.lines[flipped].get_color()
    assert plus == mpl3d.POLE_S and minus == mpl3d.POLE_N


def test_diagonal_dipole_glyph_stays_inside_frame(axes3d):
    # a tilted-B dipole must clamp its pole faces per-axis into the framed
    # rectangle (a projection bound is not rectangle membership)
    from types import SimpleNamespace

    from latent_dirac.scene.schema import DipoleElement

    element = DipoleElement(
        type="dipole", label="tilted", B_vector_t=(0.3, 0.3, 0.0), length_m=0.4, center_z_m=0.5
    )
    scene = SimpleNamespace(elements=[element])
    run_result = SimpleNamespace(monitors={})
    mpl3d.draw_scene_elements(
        axes3d, scene, run_result,
        display_scale={"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.0, 1.0)},
    )
    # yoke-edge lines connect the pole-face corners: plotted as (z, x, y)
    for line in axes3d.lines:
        _, xs, ys = line.get_data_3d()
        assert np.all(np.abs(xs) <= 1.0 + 1e-9)
        assert np.all(np.abs(ys) <= 1.0 + 1e-9)


def test_washer_screen_foil_glyphs(axes3d):
    for draw in (
        lambda: mpl3d.draw_washer(axes3d, 0.005, 0.008, 0.1),
        lambda: mpl3d.draw_screen(axes3d, 0.01, 0.006, 0.2),
        lambda: mpl3d.draw_foil(axes3d, 0.01, 0.006, 0.05),
    ):
        before = _artist_count(axes3d)
        draw()
        assert _artist_count(axes3d) > before


def test_plotly_glyph_segments():
    from latent_dirac.viz.scene_3d import _coil_segments, _quadrupole_tip_segments

    coil = _coil_segments(0.02, 0.075, 0.15)
    assert coil[0].shape[0] > 100  # the helix is a dense polyline
    assert np.allclose(np.hypot(coil[0][:, 0], coil[0][:, 1]), 0.02)

    tips = _quadrupole_tip_segments(0.4, 0.1, 0.01)
    assert len(tips) == 12  # 4 poles x (two end profiles + one spine)
    apex_distance = np.hypot(tips[0][10, 0], tips[0][10, 1])
    assert apex_distance == pytest.approx(0.01, rel=1e-6)
