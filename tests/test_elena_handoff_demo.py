"""ELENA handoff demo: engine-free wiring via the stub transformer.

The real physics numbers (degrader survival band, caught fraction) come
from engine runs recorded in the design spec; CI exercises the wiring
only. Design record:
docs/superpowers/specs/2026-07-06-elena-handoff-demo-design.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from test_matter_slab_scene import set_transformer, write_stub

from latent_dirac.scene.build import run_scene
from latent_dirac.scene.loader import load_scene

SCENE_PATH = Path("examples/scenes/elena_handoff.yaml")


def test_scene_wiring_through_the_stub(tmp_path, monkeypatch):
    set_transformer(monkeypatch, write_stub(tmp_path))
    scene = load_scene(SCENE_PATH)

    # gate parameters: the catch trap switches on after the keV drift
    trap = scene.elements[1]
    assert trap.type == "penning_trap"
    assert trap.v0_volt < 0.0  # antiproton confinement needs V0 < 0
    assert trap.t_on_s is not None and trap.t_off_s > trap.t_on_s

    result = run_scene(scene)
    final = result.pipeline_result.final_cloud

    # the stub absorbs odd ids in the foil: those must be ledgered at the
    # degrader stage; even ids survive and keep transporting
    absorbed = ~final.alive
    assert absorbed.any()
    assert np.all(final.lost_at_element[absorbed] == 0)
    assert final.weighted_count() > 0.0


def test_generator_marks_the_demo_engine_gated():
    from tools.generate_scene_demo_webps import SCENE_DEMOS

    config = SCENE_DEMOS["elena_handoff_3d.webp"]
    assert config.get("requires_engine") is True
    assert "Geant4" in config["title"]  # engine fidelity carried in the title
