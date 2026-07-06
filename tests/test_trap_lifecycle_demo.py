"""Trap storage lifecycle demo: the cooling chain actually cools.

Design record: docs/superpowers/specs/2026-07-06-trap-storage-lifecycle-demo-design.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from latent_dirac.scene.build import build_source, run_scene
from latent_dirac.scene.loader import load_scene

SCENE_PATH = Path("examples/scenes/trap_storage_lifecycle.yaml")


def test_lifecycle_cools_and_ledgers_two_loss_channels():
    scene = load_scene(SCENE_PATH)
    initial = build_source(scene).sample(np.random.default_rng(scene.seed))
    initial_mean_kev = initial.mean_kinetic_energy_joule()

    result = run_scene(scene)
    final = result.pipeline_result.final_cloud

    # survivors remain, and they are cold: the interleaved cooling bursts
    # must have removed most of the initial kinetic energy
    assert final.weighted_count() > 0.0
    assert final.mean_kinetic_energy_joule() < 0.25 * initial_mean_kev

    # the ledger separates physics losses (positronium formation in the
    # cooling stages) from storage losses (the residual-gas hold)
    labels = [element.label for element in scene.elements]
    dead_stages = {int(index) for index in final.lost_at_element[~final.alive]}
    dead_labels = {labels[index] for index in dead_stages}
    assert any(label.startswith("cool-") for label in dead_labels)
    assert "storage-hold" in dead_labels


def test_lifecycle_stays_confined():
    scene = load_scene(SCENE_PATH)
    result = run_scene(scene)
    final = result.pipeline_result.final_cloud
    positions = final.position_m[final.alive]
    # everything alive stays well inside the trap scale d = 5 mm
    assert float(np.max(np.abs(positions))) < 0.005
