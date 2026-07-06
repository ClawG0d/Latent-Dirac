"""ELENA-like ring demo: lattice integrity and scene wiring.

Skips without xtrack (CI installs the [xsuite] extra best-effort).
Design record: docs/superpowers/specs/2026-07-06-elena-ring-demo-design.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

xt = pytest.importorskip("xtrack")

from latent_dirac.scene.build import run_scene  # noqa: E402
from latent_dirac.scene.loader import load_scene  # noqa: E402

LINE_PATH = Path("examples/data/elena_like_ring.json")
SCENE_PATH = Path("examples/scenes/elena_ring.yaml")

# recorded by tools/make_elena_like_line.py when the committed lattice
# was generated; a drifted or hand-edited artifact fails here
EXPECTED_CIRCUMFERENCE_M = 30.4056
EXPECTED_QX = 2.6789
EXPECTED_QY = 1.2377


def test_committed_lattice_matches_its_provenance():
    line = xt.Line.from_json(str(LINE_PATH))
    assert abs(line.get_length() - EXPECTED_CIRCUMFERENCE_M) < 1e-6
    assert float(line.particle_ref.q0) == -1.0  # antiproton reference

    line.build_tracker()
    twiss = line.twiss(method="4d")
    assert float(twiss.qx) == pytest.approx(EXPECTED_QX, abs=1e-3)
    assert float(twiss.qy) == pytest.approx(EXPECTED_QY, abs=1e-3)


def test_ring_scene_tracks_sixty_turns_without_losses():
    scene = load_scene(SCENE_PATH)
    assert scene.elements[0].num_turns == 60

    result = run_scene(scene)
    final = result.pipeline_result.final_cloud
    assert final.alive.all()  # the working point is stable
    mean_ke_mev = final.mean_kinetic_energy_joule() / 1.602176634e-13
    assert mean_ke_mev == pytest.approx(0.1, rel=5e-3)  # ELENA extraction energy
