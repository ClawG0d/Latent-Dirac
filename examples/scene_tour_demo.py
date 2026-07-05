"""Scene tour demo: one YAML file drives simulation, report, and 3D view.

Loads `examples/scenes/scene_tour.yaml` (a beta-plus source, a guide
solenoid, a drift, a collimator, and a monitor), runs the pipeline, prints
the report, and — when Plotly is installed — writes an interactive 3D HTML
rendering of the same scene.
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

SCENE_PATH = PROJECT_ROOT / "examples" / "scenes" / "scene_tour.yaml"


def run_demo() -> dict:
    scene = load_scene(SCENE_PATH)
    result = run_scene(scene)
    final = result.pipeline_result.final_cloud
    return {
        "accepted_weighted": final.weighted_count(),
        "stage_names": [stage.stage_name for stage in result.pipeline_result.stage_results],
    }


def run_report() -> str:
    scene = load_scene(SCENE_PATH)
    result = run_scene(scene)
    return scene_report(
        scene,
        result,
        "beta-plus transport and acceptance diagnostic only",
    )


def write_interactive_html(output_path: str | Path) -> Path | None:
    """Write the interactive Plotly rendering; requires an explicit target path."""

    scene = load_scene(SCENE_PATH)
    result = run_scene(scene, record_trajectories=True)
    try:
        from latent_dirac.viz.scene_3d import render_scene_3d

        figure = render_scene_3d(scene, result)
    except ImportError:
        return None
    target = Path(output_path)
    figure.write_html(target, include_plotlyjs="cdn")
    return target


if __name__ == "__main__":
    print(run_report())
