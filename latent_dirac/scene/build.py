"""Build and run pipelines from declarative scenes."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField

from latent_dirac.beamline.aperture import Aperture
from latent_dirac.beamline.momentum_window import MomentumWindow
from latent_dirac.core.units import momentum_gev_c_to_si
from latent_dirac.fields.base import Field
from latent_dirac.fields.dipole import DipoleField
from latent_dirac.fields.penning_trap import PenningTrapField
from latent_dirac.fields.quadrupole import QuadrupoleField
from latent_dirac.fields.solenoid import SolenoidField
from latent_dirac.fields.time_gated import TimeGatedField
from latent_dirac.fields.uniform import UniformField
from latent_dirac.pipeline.runner import PipelineResult, PipelineRunner
from latent_dirac.pipeline.stage import Stage
from latent_dirac.scene.schema import FIELD_ELEMENT_TYPES, Scene
from latent_dirac.solvers.relativistic_boris import RelativisticBorisSolver
from latent_dirac.sources.antiproton_surrogate import AntiprotonSurrogateSource
from latent_dirac.sources.antiproton_table import AntiprotonYieldTableSource
from latent_dirac.sources.base import SourceTerm
from latent_dirac.sources.positron_beta import BetaPlusPositronSource
from latent_dirac.sources.positron_pair import PositronPairSource
from latent_dirac.state.particle_state import ParticleState

_SOURCE_CLASSES = {
    "positron_pair": PositronPairSource,
    "beta_plus": BetaPlusPositronSource,
    "antiproton_surrogate": AntiprotonSurrogateSource,
    "antiproton_yield_table": AntiprotonYieldTableSource,
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
                _field_for(element), scene.solver.dt_s, steps, element.label, trajectories
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
        return SolenoidField(
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
    if element.type == "drift":
        return UniformField()
    raise ValueError(f"element type {element.type!r} has no field model")


def _transport_action(field, dt_s, steps, label, trajectories):
    def transport(cloud: ParticleState) -> ParticleState:
        if trajectories is None:
            return RelativisticBorisSolver(dt_s=dt_s, steps=steps).propagate(cloud, field)

        stepper = RelativisticBorisSolver(dt_s=dt_s, steps=1)
        current = cloud
        history = [current.position_m.copy()]
        for _ in range(steps):
            current = stepper.propagate(current, field)
            history.append(current.position_m.copy())
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
        result = cloud.copy()
        radial = np.linalg.norm(result.position_m[:, :2], axis=1)
        hits = result.alive & (result.position_m[:, 2] >= element.z_m) & (radial <= element.radius_m)

        count = int(np.sum(hits))
        directions = rng.normal(size=(count, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
        annihilations[element.label] = {
            "positions": result.position_m[hits].copy(),
            "photon_directions": np.stack([directions, -directions], axis=1),
        }

        result.apply_alive_mask(~hits)
        return result

    return annihilate


def _monitor_action(label, monitors):
    def monitor(cloud: ParticleState) -> ParticleState:
        monitors[label] = cloud.copy()
        return cloud

    return monitor
