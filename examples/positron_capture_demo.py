"""Positron capture demo, defined by a declarative scene.

A parameterized positron pair source spirals through a thin-sheet
solenoid (smooth first-order fringe), an aperture and a momentum window
select the accepted cloud, and the collector plate ends the story with
ledgered two-photon annihilation events. The scene file is
`examples/scenes/positron_capture.yaml`.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from latent_dirac.diagnostics.scene_report import scene_report
from latent_dirac.scene.build import run_scene
from latent_dirac.scene.loader import load_scene

SCENE_PATH = PROJECT_ROOT / "examples" / "scenes" / "positron_capture.yaml"


def run_demo() -> dict:
    scene = load_scene(SCENE_PATH)
    result = run_scene(scene)
    final = result.pipeline_result.final_cloud
    return {
        "accepted_weighted": final.weighted_count(),
        "mean_kinetic_energy_mev": final.mean_kinetic_energy_joule() / 1.602176634e-13,
    }


def run_report() -> str:
    scene = load_scene(SCENE_PATH)
    result = run_scene(scene)
    return scene_report(
        scene,
        result,
        "positron transport and acceptance diagnostic only",
    )


if __name__ == "__main__":
    print(run_report())
