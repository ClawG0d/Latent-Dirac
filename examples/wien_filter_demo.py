"""Wien velocity filter demo (single-species positrons).

Crossed uniform E and B fields pass particles whose velocity satisfies
v = E/B undeflected; mismatched velocities are deflected and removed by a
downstream aperture. Classical transport and acceptance diagnostics only.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.scene_report import scene_report
from latent_dirac.scene.build import run_scene
from latent_dirac.scene.loader import load_scene

SCENE_PATH = PROJECT_ROOT / "examples" / "scenes" / "wien_filter.yaml"


def run_demo() -> dict[str, float]:
    scene = load_scene(SCENE_PATH)
    wien_cell = next(element for element in scene.elements if element.type == "uniform_field")
    e_field = float(wien_cell.E_vector_v_m[0])
    b_field = float(wien_cell.B_vector_t[1])

    result = run_scene(scene)
    final = result.pipeline_result.final_cloud
    total = float(final.weight.sum())

    return {
        "matched_velocity_m_s": e_field / b_field,
        "e_field_v_m": e_field,
        "b_field_t": b_field,
        "accepted_fraction": final.weighted_count() / total if total > 0.0 else 0.0,
    }


def run_report() -> str:
    scene = load_scene(SCENE_PATH)
    result = run_scene(scene)
    summary = run_demo()
    header = [
        "Wien velocity filter demo",
        "",
        f"- matched velocity E/B: {summary['matched_velocity_m_s']:.4g} m/s",
        f"- accepted fraction: {summary['accepted_fraction']:.3f}",
        "",
    ]
    return "\n".join(header) + scene_report(
        scene,
        result,
        "velocity-selection transport and acceptance diagnostic only",
    )


if __name__ == "__main__":
    print(run_report())
