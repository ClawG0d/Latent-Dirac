"""Human-readable text reporting for declarative scenes."""

from __future__ import annotations

from latent_dirac.diagnostics.loss_ledger import loss_ledger
from latent_dirac.fields.space_charge import (
    VALIDITY_NOTE as SPACE_CHARGE_VALIDITY_NOTE,
)
from latent_dirac.scene.build import SceneRunResult
from latent_dirac.scene.schema import Scene

_FIELD_DESCRIPTIONS = {
    "uniform_field": "uniform field",
    "solenoid": "idealized solenoid (hard-edge)",
    "dipole": "idealized dipole (hard-edge)",
    "quadrupole": "idealized quadrupole (hard-edge)",
    "penning_trap": "ideal Penning trap (quadrupole well + axial B)",
}


def _append_gate_line(lines: list[str], element) -> None:
    if getattr(element, "t_on_s", None) is not None:
        lines.append(
            f"- gate: active for t in [{element.t_on_s:g} s, {element.t_off_s:g} s) "
            "(ideal instantaneous switching)"
        )


def field_status_lines(scene: Scene) -> list[str]:
    lines = ["Magnetic field status:"]
    for element in scene.elements:
        if getattr(element, "space_charge", None) is not None:
            lines.append(f"- space charge ({element.label}): {SPACE_CHARGE_VALIDITY_NOTE}")
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
            _append_gate_line(lines, element)
        elif element.type == "penning_trap":
            lines.append(f"- field model: {_FIELD_DESCRIPTIONS[element.type]}")
            lines.append(
                f"- well parameter V0: {element.v0_volt:g} V, d: {element.d_m:g} m, "
                f"axial B: {element.b_tesla:g} T"
            )
            lines.append("- status: ideal global field (no hard edge)")
            _append_gate_line(lines, element)
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

    # Engine-derived results must carry the provenance four-tuple
    # (docs/safety_scope.md); sources put it in the cloud metadata.
    provenance = final.metadata.get("provenance") if isinstance(final.metadata, dict) else None
    if provenance:
        lines.append("")
        lines.append("Source provenance (engine four-tuple):")
        lines.append(f"- model type: {final.metadata.get('model_type', 'unknown')}")
        lines.append(f"- geant4 version: {provenance.get('geant4_version', 'unknown')}")
        lines.append(f"- physics list: {provenance.get('physics_list', 'unknown')}")
        lines.append(f"- datasets: {provenance.get('datasets', 'unknown')}")
        lines.append(f"- patches: {provenance.get('patches', 'unknown')}")
        lines.append(f"- table primaries: {provenance.get('n_primaries', 'unknown')}")

    # A Matter transform namespaces its four-tuple under `matter` so it never
    # clobbers an upstream source's provenance; surface it independently, or a
    # yield-table-source + matter-slab chain would silently drop the engine's.
    matter = final.metadata.get("matter") if isinstance(final.metadata, dict) else None
    if isinstance(matter, dict) and isinstance(matter.get("provenance"), dict):
        mp = matter["provenance"]
        lines.append("")
        lines.append("Matter engine provenance (four-tuple):")
        lines.append(
            f"- material: {matter.get('material', 'unknown')}, "
            f"thickness: {matter.get('thickness_mm', 'unknown')} mm"
        )
        lines.append(f"- geant4 version: {mp.get('geant4_version', 'unknown')}")
        lines.append(f"- physics list: {mp.get('physics_list', 'unknown')}")
        lines.append(f"- datasets: {mp.get('datasets', 'unknown')}")
        lines.append(f"- patches: {mp.get('patches', 'unknown')}")

    for label, events in result.annihilations.items():
        lines.append("")
        lines.append(f"Annihilation endpoint {label!r}:")
        lines.append(f"- events: {events['positions'].shape[0]}")
        lines.append("- at-rest two-photon kinematics (511 keV label only; no energetics)")

    lines.append("")
    lines.extend(field_status_lines(scene))
    lines.append("")
    lines.append("Scope note:")
    lines.append(f"- {scope_note}")
    return "\n".join(lines)
