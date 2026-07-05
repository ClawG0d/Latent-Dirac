"""Human-readable text reporting for declarative scenes."""

from __future__ import annotations

from latent_dirac.diagnostics.loss_ledger import loss_ledger
from latent_dirac.scene.build import SceneRunResult
from latent_dirac.scene.schema import Scene

_FIELD_DESCRIPTIONS = {
    "uniform_field": "uniform field",
    "solenoid": "idealized solenoid (hard-edge)",
    "dipole": "idealized dipole (hard-edge)",
    "quadrupole": "idealized quadrupole (hard-edge)",
    "penning_trap": "ideal Penning trap (quadrupole well + axial B)",
}


def field_status_lines(scene: Scene) -> list[str]:
    lines = ["Magnetic field status:"]
    for element in scene.elements:
        if element.type == "solenoid":
            lines.append(f"- field model: {_FIELD_DESCRIPTIONS[element.type]}")
            lines.append(f"- B vector [T]: [0, 0, {element.b_tesla:g}] inside solenoid envelope")
            lines.append(
                f"- status: active inside radius {element.radius_m:g} m and length {element.length_m:g} m"
            )
        elif element.type == "uniform_field":
            lines.append(f"- field model: {_FIELD_DESCRIPTIONS[element.type]}")
            b = element.B_vector_t
            lines.append(f"- B vector [T]: [{b[0]:g}, {b[1]:g}, {b[2]:g}]")
            e = element.E_vector_v_m
            if any(component != 0.0 for component in e):
                lines.append(f"- E vector [V/m]: [{e[0]:g}, {e[1]:g}, {e[2]:g}]")
            lines.append("- status: active over all sampled positions")
        elif element.type == "penning_trap":
            lines.append(f"- field model: {_FIELD_DESCRIPTIONS[element.type]}")
            lines.append(
                f"- well parameter V0: {element.v0_volt:g} V, d: {element.d_m:g} m, "
                f"axial B: {element.b_tesla:g} T"
            )
            lines.append("- status: ideal global field (no hard edge)")
        elif element.type in ("dipole", "quadrupole"):
            lines.append(f"- field model: {_FIELD_DESCRIPTIONS[element.type]}")
            lines.append(f"- status: active over length {element.length_m:g} m")
    return lines


def scene_report(scene: Scene, result: SceneRunResult, scope_note: str) -> str:
    lines = [f"Latent Dirac scene report: {scene.name}", "", "Stage accounting:"]
    for stage in result.pipeline_result.stage_results:
        lines.append(
            f"- {stage.stage_name}: input={stage.input_weighted_count:g}, "
            f"output={stage.output_weighted_count:g}, "
            f"transmission={stage.transmission:.3g}, losses={stage.losses:g}"
        )

    ledger = loss_ledger(result.pipeline_result.final_cloud, result.pipeline_result.stage_results)
    lines.append("")
    lines.append("Loss ledger (weighted, by killing element):")
    for name, loss in ledger.items():
        lines.append(f"- {name}: {loss:g}")

    final = result.pipeline_result.final_cloud
    lines.append("")
    lines.append("Accepted state:")
    lines.append(f"- weighted count: {final.weighted_count():g}")
    lines.append(f"- mean kinetic energy: {final.mean_kinetic_energy_joule() / 1.602176634e-13:.6g} MeV")

    lines.append("")
    lines.extend(field_status_lines(scene))
    lines.append("")
    lines.append("Scope note:")
    lines.append(f"- {scope_note}")
    return "\n".join(lines)
