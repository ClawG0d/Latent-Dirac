"""Human-readable text reporting for declarative scenes."""

from __future__ import annotations

from latent_dirac.diagnostics.loss_ledger import loss_ledger
from latent_dirac.fields.space_charge import (
    VALIDITY_NOTE as SPACE_CHARGE_VALIDITY_NOTE,
)
from latent_dirac.scene.build import SceneRunResult
from latent_dirac.scene.schema import FIELD_ELEMENT_TYPES, Scene

_FIELD_DESCRIPTIONS = {
    "uniform_field": "uniform field",
    "solenoid": "idealized solenoid (hard-edge)",
    "dipole": "idealized dipole (hard-edge)",
    "quadrupole": "idealized quadrupole (hard-edge)",
    "penning_trap": "ideal Penning trap (quadrupole well + axial B)",
    "rotating_wall": "rotating multipole E field (single-particle)",
}

_MULTIPOLE_NAMES = {1: "dipole", 2: "quadrupole"}


def _deduplicated_field_elements(scene: Scene):
    """Field elements with physics-identical consecutive repeats dropped.

    Interleaved pipelines (e.g. trap -> cool -> trap -> cool) re-declare
    the same field element many times; the field-status block should
    describe the field once per distinct configuration, not once per
    pipeline stage. Only the field subsequence is compared, and `label`/
    `steps` are ignored (stage bookkeeping, not field physics).
    """

    previous = None
    for element in scene.elements:
        if element.type not in FIELD_ELEMENT_TYPES:
            continue
        fingerprint = element.model_dump(exclude={"label", "steps"})
        if fingerprint != previous:
            yield element
        previous = fingerprint


def _append_gate_line(lines: list[str], element) -> None:
    if getattr(element, "t_on_s", None) is not None:
        lines.append(
            f"- gate: active for t in [{element.t_on_s:g} s, {element.t_off_s:g} s) "
            "(ideal instantaneous switching)"
        )


def field_status_lines(scene: Scene) -> list[str]:
    if not any(element.type in FIELD_ELEMENT_TYPES for element in scene.elements):
        return []  # no first-party field elements: no empty header
    lines = ["Magnetic field status:"]
    for element in _deduplicated_field_elements(scene):
        if getattr(element, "space_charge", None) is not None:
            lines.append(f"- space charge ({element.label}): {SPACE_CHARGE_VALIDITY_NOTE}")
        if element.type == "solenoid":
            if element.profile == "thin_sheet":
                lines.append(
                    "- field model: thin-sheet solenoid (smooth finite-length profile, "
                    "first-order fringe)"
                )
                lines.append(
                    f"- sheet strength B0 [T]: {element.b_tesla:g} (center Bz below B0 for short coils)"
                )
                lines.append(
                    f"- status: sheet radius {element.radius_m:g} m, length {element.length_m:g} m, "
                    "smooth fringe outside"
                )
            else:
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
        elif element.type == "rotating_wall":
            pole = _MULTIPOLE_NAMES[element.multipole]
            lines.append(
                f"- field model: {_FIELD_DESCRIPTIONS[element.type]} ({pole}, m={element.multipole})"
            )
            lines.append(
                f"- drive: amplitude {element.amplitude_v_m:g} V/m at r={element.radius_m:g} m, "
                f"frequency {element.frequency_hz:g} Hz"
            )
            lines.append(
                "- status: single-particle rotating field; plasma compression "
                "is collective (out of scope, needs PIC)"
            )
            _append_gate_line(lines, element)
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
    matter_block = final.metadata.get("matter") if isinstance(final.metadata, dict) else None
    matter_provenance = matter_block.get("provenance") if isinstance(matter_block, dict) else None
    if provenance and provenance == matter_provenance:
        # the top-level tuple was setdefault-ed by the Matter stage onto a
        # source that carried none: printing it under a "Source provenance"
        # heading would attribute engine provenance to a non-engine source.
        # The matter block below reports it under its honest heading.
        provenance = None
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

    # Table-based buffer-gas cooling carries a cross-section provenance block
    # (the cross-section analogue of the engine four-tuple).
    buffer_gas = final.metadata.get("buffer_gas") if isinstance(final.metadata, dict) else None
    if isinstance(buffer_gas, dict) and buffer_gas:
        lines.append("")
        lines.append("Buffer-gas cross-section provenance:")
        for label, prov in buffer_gas.items():
            energy_range = prov.get("energy_range_ev", ["?", "?"])
            lines.append(
                f"- {label}: gas {prov.get('gas', 'unknown')}, "
                f"tier {prov.get('fidelity_tier', 'unknown')}, "
                f"source {prov.get('source', 'unknown')}, "
                f"doi {prov.get('doi') or 'none'}"
            )
            lines.append(
                f"  channels {prov.get('channels', [])}, "
                f"E range [{energy_range[0]:g}, {energy_range[1]:g}] eV, "
                f"n_gas {prov.get('n_gas_m3', float('nan')):.3g} m^-3"
            )

    for label, events in result.annihilations.items():
        lines.append("")
        lines.append(f"Annihilation endpoint {label!r}:")
        lines.append(f"- events: {events['positions'].shape[0]}")
        lines.append("- at-rest two-photon kinematics (511 keV label only; no energetics)")

    field_lines = field_status_lines(scene)
    if field_lines:
        lines.append("")
        lines.extend(field_lines)
    lines.append("")
    lines.append("Scope note:")
    lines.append(f"- {scope_note}")
    return "\n".join(lines)
