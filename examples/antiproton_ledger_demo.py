"""Antiproton loss-ledger demo.

A surrogate antiproton source is transported through a uniform solenoidal
field, then cut by a beam-pipe aperture and a momentum window. The
per-particle ledger records which element killed each antiparticle.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from latent_dirac.diagnostics.loss_ledger import loss_ledger
from latent_dirac.diagnostics.scene_report import scene_report
from latent_dirac.scene.build import run_scene
from latent_dirac.scene.loader import load_scene

SCENE_PATH = PROJECT_ROOT / "examples" / "scenes" / "antiproton_ledger.yaml"


def run_demo() -> dict:
    scene = load_scene(SCENE_PATH)
    result = run_scene(scene)
    ledger = loss_ledger(result.pipeline_result.final_cloud, result.pipeline_result.stage_results)
    return {
        "ledger": ledger,
        "accepted_weighted": result.pipeline_result.final_cloud.weighted_count(),
    }


def run_report() -> str:
    scene = load_scene(SCENE_PATH)
    result = run_scene(scene)
    return scene_report(
        scene,
        result,
        "antiproton transport and acceptance ledger diagnostic only",
    )


if __name__ == "__main__":
    print(run_report())
