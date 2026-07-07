"""Build and run pipelines from declarative scenes."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField

from latent_dirac.beamline.aperture import Aperture
from latent_dirac.beamline.momentum_window import MomentumWindow
from latent_dirac.core.units import momentum_gev_c_to_si
from latent_dirac.fields.base import Field
from latent_dirac.fields.composite import CompositeField
from latent_dirac.fields.dipole import DipoleField
from latent_dirac.fields.penning_trap import PenningTrapField
from latent_dirac.fields.quadrupole import QuadrupoleField
from latent_dirac.fields.rotating_wall import RotatingWallField
from latent_dirac.fields.solenoid import SolenoidField, ThinSheetSolenoidField
from latent_dirac.fields.space_charge import fit_uniform_sphere
from latent_dirac.fields.time_gated import TimeGatedField
from latent_dirac.fields.uniform import UniformField
from latent_dirac.pipeline.runner import PipelineResult, PipelineRunner
from latent_dirac.pipeline.stage import Stage
from latent_dirac.scene.schema import FIELD_ELEMENT_TYPES, Scene
from latent_dirac.solvers.relativistic_boris import RelativisticBorisSolver
from latent_dirac.sources.antiproton_surrogate import AntiprotonSurrogateSource
from latent_dirac.sources.antiproton_table import AntiprotonYieldTableSource
from latent_dirac.sources.base import SourceTerm
from latent_dirac.sources.cold_sphere import ColdUniformSphereSource
from latent_dirac.sources.positron_beta import BetaPlusPositronSource
from latent_dirac.sources.positron_pair import PositronPairSource
from latent_dirac.sources.positron_table import PositronYieldTableSource
from latent_dirac.state.particle_state import ParticleState

_SOURCE_CLASSES = {
    "positron_pair": PositronPairSource,
    "beta_plus": BetaPlusPositronSource,
    "antiproton_surrogate": AntiprotonSurrogateSource,
    "antiproton_yield_table": AntiprotonYieldTableSource,
    "positron_yield_table": PositronYieldTableSource,
    "cold_uniform_sphere": ColdUniformSphereSource,
}


class SceneRunResult(BaseModel):
    """Pipeline outcome plus scene-level diagnostics keyed by element label."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pipeline_result: PipelineResult
    monitors: dict[str, ParticleState] = PydanticField(default_factory=dict)
    trajectories: dict[str, np.ndarray] = PydanticField(default_factory=dict)
    annihilations: dict[str, dict[str, np.ndarray]] = PydanticField(default_factory=dict)


def build_source(scene: Scene) -> SourceTerm:
    return _SOURCE_CLASSES[scene.source.type](**scene.source.params)


def run_scene(
    scene: Scene,
    rng: np.random.Generator | None = None,
    record_trajectories: bool = False,
) -> SceneRunResult:
    """Sample the scene source and run its elements as a staged pipeline.

    A caller-supplied `rng` overrides `scene.seed`; determinism is then the
    caller's responsibility.
    """

    rng = np.random.default_rng(scene.seed) if rng is None else rng
    cloud = build_source(scene).sample(rng)

    monitors: dict[str, ParticleState] = {}
    annihilations: dict[str, dict[str, np.ndarray]] = {}
    trajectories: dict[str, np.ndarray] | None = {} if record_trajectories else None
    runner = PipelineRunner(stages=_build_stages(scene, monitors, trajectories, annihilations))
    pipeline_result = runner.run(cloud)

    return SceneRunResult(
        pipeline_result=pipeline_result,
        monitors=monitors,
        trajectories=trajectories if trajectories is not None else {},
        annihilations=annihilations,
    )


def _build_stages(
    scene: Scene,
    monitors: dict[str, ParticleState],
    trajectories: dict[str, np.ndarray] | None,
    annihilations: dict[str, dict[str, np.ndarray]] | None = None,
) -> list[Stage]:
    stages: list[Stage] = []
    annihilations = annihilations if annihilations is not None else {}
    for stage_index, element in enumerate(scene.elements):
        if element.type in FIELD_ELEMENT_TYPES or element.type == "drift":
            steps = element.steps if element.steps is not None else scene.solver.steps
            action = _transport_action(
                _field_for(element),
                scene.solver.dt_s,
                steps,
                element.label,
                trajectories,
                space_charge=getattr(element, "space_charge", None),
            )
        elif element.type == "aperture":
            action = Aperture(radius_m=element.radius_m, z_m=element.z_m).apply
        elif element.type == "momentum_window":
            action = MomentumWindow(
                momentum_gev_c_to_si(element.p_min_gev_c),
                momentum_gev_c_to_si(element.p_max_gev_c),
            ).apply
        elif element.type == "annihilation_plate":
            action = _annihilation_action(
                element, annihilations, np.random.default_rng(scene.seed + 7919 + stage_index)
            )
        elif element.type == "residual_gas_loss":
            action = _residual_gas_loss_action(
                element, np.random.default_rng(scene.seed + 6229 + stage_index)
            )
        elif element.type == "buffer_gas_cooling":
            action = _buffer_gas_cooling_action(
                element, np.random.default_rng(scene.seed + 3313 + stage_index)
            )
        elif element.type == "matter_slab":
            action = _matter_slab_action(element)
        elif element.type == "xsuite_lattice":
            action = _xsuite_lattice_action(element)
        elif element.type == "monitor":
            action = _monitor_action(element.label, monitors)
        else:  # pragma: no cover - the schema union prevents this
            raise ValueError(f"unsupported element type {element.type!r}")
        stages.append(Stage(element.label, action))
    return stages


def _field_for(element) -> Field:
    field = _base_field_for(element)
    t_on = getattr(element, "t_on_s", None)
    if t_on is not None:
        field = TimeGatedField(inner=field, t_on_s=t_on, t_off_s=element.t_off_s)
    return field


def _base_field_for(element) -> Field:
    if element.type == "uniform_field":
        return UniformField(
            B_vector_t=np.asarray(element.B_vector_t, dtype=float),
            E_vector_v_m=np.asarray(element.E_vector_v_m, dtype=float),
        )
    if element.type == "solenoid":
        solenoid_class = ThinSheetSolenoidField if element.profile == "thin_sheet" else SolenoidField
        return solenoid_class(
            b_tesla=element.b_tesla,
            radius_m=element.radius_m,
            length_m=element.length_m,
            center_z_m=element.center_z_m,
        )
    if element.type == "dipole":
        return DipoleField(
            B_vector_t=element.B_vector_t,
            length_m=element.length_m,
            center_z_m=element.center_z_m,
        )
    if element.type == "quadrupole":
        return QuadrupoleField(
            gradient_t_m=element.gradient_t_m,
            length_m=element.length_m,
            center_z_m=element.center_z_m,
        )
    if element.type == "penning_trap":
        return PenningTrapField(
            v0_volt=element.v0_volt,
            d_m=element.d_m,
            b_tesla=element.b_tesla,
            center_z_m=element.center_z_m,
        )
    if element.type == "rotating_wall":
        return RotatingWallField(
            multipole=element.multipole,
            amplitude_v_m=element.amplitude_v_m,
            radius_m=element.radius_m,
            frequency_hz=element.frequency_hz,
            phase_rad=element.phase_rad,
        )
    if element.type == "drift":
        return UniformField()
    raise ValueError(f"element type {element.type!r} has no field model")


def _transport_action(field, dt_s, steps, label, trajectories, space_charge=None):
    def transport(cloud: ParticleState) -> ParticleState:
        if trajectories is None and space_charge is None:
            return RelativisticBorisSolver(dt_s=dt_s, steps=steps).propagate(cloud, field)

        stepper = RelativisticBorisSolver(dt_s=dt_s, steps=1)
        current = cloud
        history = [current.position_m.copy()] if trajectories is not None else None
        for _ in range(steps):
            step_field = field
            if space_charge is not None:
                # mean field frozen within the step, refitted every step
                # from the alive cloud (parameterized uniform-sphere tier)
                self_field = fit_uniform_sphere(current)
                if self_field is not None:
                    step_field = CompositeField(fields=[field, self_field])
            current = stepper.propagate(current, step_field)
            if history is not None:
                history.append(current.position_m.copy())
        if history is not None:
            trajectories[label] = np.stack(history)
        return current

    return transport


def _annihilation_action(element, annihilations, rng):
    """Kill crossing positrons and record at-rest 2-photon kinematics.

    Isotropic back-to-back unit-vector pairs (511 keV appears as a label
    only); no energy release or deposition is computed - see the safety
    scope.
    """

    def annihilate(cloud: ParticleState) -> ParticleState:
        if cloud.species.name != "positron":
            raise ValueError(
                f"annihilation_plate {element.label!r} models positron two-photon "
                f"annihilation; species {cloud.species.name!r} is not supported "
                "(antiproton annihilation is a multi-pion final state — engine "
                "physics, see the safety scope)"
            )
        result = cloud.copy()
        radial = np.linalg.norm(result.position_m[:, :2], axis=1)
        hits = result.alive & (result.position_m[:, 2] >= element.z_m) & (radial <= element.radius_m)

        count = int(np.sum(hits))
        directions = rng.normal(size=(count, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
        # the stage-boundary kill overshoots the plane by up to a full
        # transport stage; project each event straight back along its
        # velocity onto the plate plane (exact in field-free drift,
        # first-order in a field) so recorded vertices and clocks sit on
        # the plate the caption points at
        positions = result.position_m[hits].copy()
        times = result.time_s[hits].copy()
        velocities = result.velocity()[hits]
        v_z = velocities[:, 2]
        forward = v_z > 0.0
        dt_back = np.where(forward, (positions[:, 2] - element.z_m) / np.where(forward, v_z, 1.0), 0.0)
        positions -= velocities * dt_back[:, np.newaxis]
        times -= dt_back
        annihilations[element.label] = {
            "positions": positions,
            "photon_directions": np.stack([directions, -directions], axis=1),
            # plane-projected event clock and identity so renderers can
            # animate each burst at the moment its trail reaches the plate
            "time_s": times,
            "particle_id": result.particle_id[hits].copy(),
        }

        result.apply_alive_mask(~hits)
        return result

    return annihilate


def _residual_gas_loss_action(element, rng):
    """Stochastic annihilation on residual gas over a hold time.

    Per-particle exponential survival exp(-hold/tau); survivors age by the
    hold time, killed particles are booked by `Stage.run` into the ledger.
    Fidelity tier: parameterized (tau is a direct input) - see the
    residual-gas storage-lifetime spec.
    """
    p_survive = float(np.exp(-element.hold_time_s / element.mean_lifetime_s))

    def hold(cloud: ParticleState) -> ParticleState:
        result = cloud.copy()
        draws = rng.random(result.alive.shape[0])
        # only alive particles are at risk; dead ones keep their state
        survive = ~result.alive | (draws < p_survive)
        aged = result.alive & survive
        result.time_s = result.time_s + aged * element.hold_time_s
        result.apply_alive_mask(survive)
        return result

    return hold


def _matter_slab_action(element):
    """Track the cloud through a NIST material slab via the Geant4 engine.

    The transformer binary is machine-specific and injected at run time
    (LATENT_DIRAC_G4_TRANSFORMER, shlex-split; optional
    LATENT_DIRAC_G4_PATH_STYLE = native | wsl), never stored in the scene.
    The stage is built unconditionally so scenes construct and render with
    no engine present; a missing binary only fails when the stage runs.
    """
    import os
    import shlex

    from latent_dirac.adapters.geant4.adapter import Geant4MatterAdapter

    # shlex.split: a transformer path containing spaces must be quoted
    command_str = os.environ.get("LATENT_DIRAC_G4_TRANSFORMER")
    if not command_str:

        def missing(cloud: ParticleState) -> ParticleState:
            raise RuntimeError(
                f"matter_slab {element.label!r} needs the engine transformer: set "
                "LATENT_DIRAC_G4_TRANSFORMER to the transformer command (e.g. the "
                "engine/transformer binary). The scene builds and renders without it, "
                "but running the slab stage requires it."
            )

        return missing

    adapter = Geant4MatterAdapter(
        command=tuple(shlex.split(command_str)),
        material=element.material,
        thickness_mm=element.thickness_mm,
        entry_z_m=element.entry_z_m,
        path_style=os.environ.get("LATENT_DIRAC_G4_PATH_STYLE", "native"),
        transverse_half_width_m=element.transverse_half_width_m,
        world_half_length_m=element.world_half_length_m,
    )
    return adapter.apply


def _xsuite_lattice_action(element):
    """Track the cloud through an xtrack.Line declared in the scene (T2).

    The Line is loaded from `line_path` (already resolved relative to the
    scene file by the loader); the reference momentum `p0c_ev` is always
    explicit. Loading needs xtrack, so a missing `[xsuite]` extra raises a
    clear ImportError when the stage is built (scenes still construct and
    render without it). Ledger stamping flows through `Stage.run`.
    """
    from latent_dirac.adapters.xsuite.adapter import (
        ReferenceFrame,
        _require_xtrack,
        xsuite_tracking_stage,
    )

    xtrack = _require_xtrack()
    line = xtrack.Line.from_json(element.line_path)
    frame = ReferenceFrame(p0c_ev=element.p0c_ev)
    stage = xsuite_tracking_stage(
        element.label, line, frame, num_turns=element.num_turns
    )
    return stage.action


def _buffer_gas_cooling_action(element, rng):
    """Surko-type buffer-gas cooling over a hold time.

    Table-based mode (cross_section_path set): the null-collision operator
    driven by a curated cross-section table and the gas number density
    n = gas_pressure_pa / (k_B * gas_temperature_k). Otherwise the
    constant-rate parameterized tier (Poisson collisions; each is a
    cooling collision — energy drops by energy_loss_ev, floored at
    (3/2)kT, direction preserved — or a Ps-formation loss). Pure and
    seeded; NumPy pipeline only.
    """
    if element.cross_section_path is not None:
        return _buffer_gas_table_action(element, rng)

    from latent_dirac.core.constants import ELEMENTARY_CHARGE_C, k_B
    from latent_dirac.core.units import kinetic_energy_to_momentum_magnitude

    delta_e = element.energy_loss_ev * ELEMENTARY_CHARGE_C
    floor = 1.5 * k_B * element.gas_temperature_k
    mean_collisions = element.collision_rate_hz * element.hold_time_s

    def cool(cloud: ParticleState) -> ParticleState:
        result = cloud.copy()
        mass = result.species.mass_kg
        alive_in = result.alive.copy()
        counts = rng.poisson(mean_collisions, size=alive_in.shape[0])
        ke = result.kinetic_energy_joule()
        survive = result.alive.copy()

        for i in np.flatnonzero(alive_in):
            energy = ke[i]
            for _ in range(int(counts[i])):
                if rng.random() < element.ps_fraction:
                    survive[i] = False  # positronium formation: particle lost
                    break
                energy = max(energy - delta_e, floor)  # cooling collision
            if survive[i] and int(counts[i]) > 0:
                # rescale momentum to the cooled energy, direction preserved
                p_now = result.momentum_kg_m_s[i]
                norm = float(np.linalg.norm(p_now))
                if norm > 0.0:
                    new_mag = float(kinetic_energy_to_momentum_magnitude(energy, mass))
                    result.momentum_kg_m_s[i] = p_now * (new_mag / norm)

        aged = alive_in & survive
        result.time_s = result.time_s + aged * element.hold_time_s
        result.apply_alive_mask(survive)
        return result

    return cool


def _buffer_gas_table_action(element, rng):
    """Table-based buffer-gas cooling: the null-collision operator on a
    curated cross-section table. The table's provenance and fidelity tier
    are stamped into the cloud metadata (the cross-section analogue of the
    engine four-tuple) so the scene report can surface them.
    """
    from latent_dirac.collisions import buffer_gas_collide, load_cross_sections
    from latent_dirac.core.constants import k_B

    table = load_cross_sections(element.cross_section_path)
    n_gas_m3 = element.gas_pressure_pa / (k_B * element.gas_temperature_k)
    provenance = {
        "gas": table.provenance.get("gas"),
        "fidelity_tier": table.fidelity_tier,
        "source": table.provenance.get("source"),
        "doi": table.provenance.get("doi", ""),
        "channels": list(table.channels.keys()),
        "energy_range_ev": [float(table.energies_ev[0]), float(table.energies_ev[-1])],
        "gas_pressure_pa": element.gas_pressure_pa,
        "n_gas_m3": n_gas_m3,
    }

    def cool_table(cloud: ParticleState) -> ParticleState:
        result = buffer_gas_collide(
            cloud,
            table,
            n_gas_m3=n_gas_m3,
            hold_time_s=element.hold_time_s,
            gas_temperature_k=element.gas_temperature_k,
            rng=rng,
        )
        result.metadata.setdefault("buffer_gas", {})[element.label] = provenance
        return result

    return cool_table


def _monitor_action(label, monitors):
    def monitor(cloud: ParticleState) -> ParticleState:
        monitors[label] = cloud.copy()
        return cloud

    return monitor
