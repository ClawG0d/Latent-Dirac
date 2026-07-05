"""Physics and loading checks for the README demo scenes."""

from pathlib import Path

import numpy as np
import pytest

from latent_dirac.scene.build import run_scene
from latent_dirac.scene.loader import load_scene

SCENES_DIR = Path("examples/scenes")
SCENE_FILES = (
    "annihilation_endpoint.yaml",
    "decel_capture.yaml",
    "target_production.yaml",
    "decay_emission.yaml",
    "scene_tour.yaml",
    "positron_capture.yaml",
    "dipole_quad_line.yaml",
    "wien_filter.yaml",
    "antiproton_ledger.yaml",
)


@pytest.mark.parametrize("name", SCENE_FILES)
def test_demo_scene_loads_and_runs(name):
    scene = load_scene(SCENES_DIR / name)
    result = run_scene(scene)

    assert result.pipeline_result.final_cloud.weighted_count() > 0.0


def test_energy_coloring_maps_initial_kinetic_energy():
    from tools.generate_scene_demo_webps import _particle_colors_energy

    scene = load_scene(SCENES_DIR / "decay_emission.yaml")
    colors = _particle_colors_energy(scene)

    assert len(colors) == 96
    assert all(len(color) == 3 for color in colors)
    # a continuous beta spectrum must span the colormap, not collapse to one hue
    unique = {tuple(np.round(color, 3)) for color in colors}
    assert len(unique) > 20


def test_wien_filter_selects_matched_velocity():
    from examples.wien_filter_demo import run_demo
    from latent_dirac.scene.build import build_source

    summary = run_demo()

    # the matched-velocity core passes, mismatched tails are cut
    assert 0.05 < summary["accepted_fraction"] < 0.95
    assert summary["matched_velocity_m_s"] > 0.0
    # crossed fields are declared in the scene
    assert summary["e_field_v_m"] > 0.0
    assert summary["b_field_t"] > 0.0

    # selection is by velocity, not a random subset: accepted particles start
    # closer to the design velocity v = E/B than the ones that were cut
    scene = load_scene(SCENES_DIR / "wien_filter.yaml")
    initial = build_source(scene).sample(np.random.default_rng(scene.seed))
    from latent_dirac.scene.build import run_scene as run

    final = run(scene).pipeline_result.final_cloud
    speed_error = np.abs(np.linalg.norm(initial.velocity(), axis=1) - summary["matched_velocity_m_s"])
    assert speed_error[final.alive].mean() < speed_error[~final.alive].mean()


def test_antiproton_ledger_records_multiple_killing_stages():
    from examples.antiproton_ledger_demo import run_demo

    summary = run_demo()
    ledger = summary["ledger"]

    killing_stages = [name for name, loss in ledger.items() if name != "surviving" and loss > 0.0]
    assert len(killing_stages) >= 2
    assert ledger["surviving"] > 0.0


def test_magnetic_mirror_traps_and_reflects():
    from examples.magnetic_mirror_demo import run_demo

    summary = run_demo()

    # trapped particles stay inside the bottle and reverse axial direction
    assert summary["trapped_fraction"] > 0.3
    assert summary["max_abs_z_m"] < summary["mirror_half_length_m"] * 1.05
    assert summary["field_model"] == "table-based field map (synthetic analytic mirror)"


def test_magnetic_mirror_uses_comsol_import_pipeline(tmp_path):
    from examples.magnetic_mirror_demo import write_mirror_field_csv
    from latent_dirac.fields.field_map import load_comsol_grid_csv

    path = write_mirror_field_csv(tmp_path / "mirror.csv")
    field = load_comsol_grid_csv(path)

    # on-axis field strengthens toward the throats (mirror ratio > 1)
    b_center = field.B(np.array([0.0, 0.0, 0.0]), 0.0)
    b_throat = field.B(np.array([0.0, 0.0, field.z_m[-1]]), 0.0)
    assert np.linalg.norm(b_throat) > 1.5 * np.linalg.norm(b_center)
