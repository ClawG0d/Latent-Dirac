"""Pydantic models for the declarative scene schema (schema_version 1).

Validation is fail-fast: unknown element types, unknown keys, missing
labels, and duplicate labels are rejected at load time. Labels become
pipeline stage names, anchoring loss accounting to the scene description.

Batch convention: every numeric parameter must remain liftable into a
batch-dimension array (the Phase 3 vmap prerequisite); the schema must not
grow structures that prevent vmap over configurations.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SceneModel(BaseModel):
    """Base model with fail-fast validation for all scene nodes."""

    model_config = ConfigDict(extra="forbid")


class SourceSpec(SceneModel):
    type: Literal[
        "positron_pair",
        "beta_plus",
        "antiproton_surrogate",
        "antiproton_yield_table",
        "cold_uniform_sphere",
    ]
    label: str
    params: dict[str, Any] = Field(default_factory=dict)


class SolverSpec(SceneModel):
    type: Literal["relativistic_boris"] = "relativistic_boris"
    dt_s: float
    steps: int

    @field_validator("dt_s")
    @classmethod
    def _positive_dt(cls, value):
        if value <= 0.0:
            raise ValueError("dt_s must be positive")
        return value

    @field_validator("steps")
    @classmethod
    def _positive_steps(cls, value):
        if value <= 0:
            raise ValueError("steps must be positive")
        return value


class ElementBase(SceneModel):
    label: str


def _validate_gate_window(element):
    has_on = element.t_on_s is not None
    has_off = element.t_off_s is not None
    if has_on != has_off:
        raise ValueError("t_on_s and t_off_s must be set together")
    if has_on and element.t_off_s <= element.t_on_s:
        raise ValueError("t_off_s must be greater than t_on_s")
    return element


class UniformFieldElement(ElementBase):
    type: Literal["uniform_field"]
    B_vector_t: tuple[float, float, float] = (0.0, 0.0, 0.0)
    E_vector_v_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    t_on_s: float | None = None
    t_off_s: float | None = None
    steps: int | None = Field(default=None, ge=1)
    space_charge: Literal["uniform_sphere"] | None = None

    _gate_window = model_validator(mode="after")(_validate_gate_window)


class SolenoidElement(ElementBase):
    type: Literal["solenoid"]
    b_tesla: float
    radius_m: float
    length_m: float
    center_z_m: float = 0.0
    steps: int | None = Field(default=None, ge=1)


class DipoleElement(ElementBase):
    type: Literal["dipole"]
    B_vector_t: tuple[float, float, float]
    length_m: float
    center_z_m: float = 0.0
    steps: int | None = Field(default=None, ge=1)


class QuadrupoleElement(ElementBase):
    type: Literal["quadrupole"]
    gradient_t_m: float
    length_m: float
    center_z_m: float = 0.0
    steps: int | None = Field(default=None, ge=1)


class PenningTrapElement(ElementBase):
    """Ideal Penning trap (parameterized; no electrode geometry)."""

    type: Literal["penning_trap"]
    v0_volt: float
    d_m: float = Field(gt=0)
    b_tesla: float
    center_z_m: float = 0.0
    t_on_s: float | None = None
    t_off_s: float | None = None
    steps: int | None = Field(default=None, ge=1)
    space_charge: Literal["uniform_sphere"] | None = None

    _gate_window = model_validator(mode="after")(_validate_gate_window)


class DriftElement(ElementBase):
    """Zero-field transport segment (exact within the solver contract)."""

    type: Literal["drift"]
    steps: int

    @field_validator("steps")
    @classmethod
    def _positive_steps(cls, value):
        if value <= 0:
            raise ValueError("steps must be positive")
        return value


class ApertureElement(ElementBase):
    type: Literal["aperture"]
    radius_m: float
    z_m: float


class MomentumWindowElement(ElementBase):
    type: Literal["momentum_window"]
    p_min_gev_c: float
    p_max_gev_c: float


class AnnihilationPlateElement(ElementBase):
    """Annihilation endpoint: kills crossing e+ and records 2-photon kinematics.

    At-rest approximation, no energetics (see the safety scope); fidelity
    tier: parameterized.
    """

    type: Literal["annihilation_plate"]
    z_m: float
    radius_m: float = Field(gt=0)


class MonitorElement(ElementBase):
    """Diagnostic snapshot of the cloud at this pipeline position (no physics)."""

    type: Literal["monitor"]


ElementSpec = Annotated[
    UniformFieldElement
    | SolenoidElement
    | DipoleElement
    | QuadrupoleElement
    | PenningTrapElement
    | DriftElement
    | ApertureElement
    | MomentumWindowElement
    | AnnihilationPlateElement
    | MonitorElement,
    Field(discriminator="type"),
]

FIELD_ELEMENT_TYPES = ("uniform_field", "solenoid", "dipole", "quadrupole", "penning_trap")


class Scene(SceneModel):
    schema_version: Literal[1]
    name: str
    seed: int = 0
    source: SourceSpec
    solver: SolverSpec
    elements: list[ElementSpec]

    @model_validator(mode="after")
    def _unique_labels(self):
        labels = [self.source.label] + [element.label for element in self.elements]
        duplicates = {label for label in labels if labels.count(label) > 1}
        if duplicates:
            raise ValueError(f"scene labels must be unique, duplicated: {sorted(duplicates)}")
        return self
