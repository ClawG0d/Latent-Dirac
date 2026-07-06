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
    """Solenoid with a selectable axial profile.

    `hard_edge` (default): uniform Bz inside the cylinder, zero outside.
    `thin_sheet`: smooth finite-length thin-current-sheet profile with a
    first-order radial fringe (exactly divergence-free); `b_tesla` is
    then the sheet strength B0, preserving the integrated on-axis
    strength B0 * length_m of the hard-edge element.
    """

    type: Literal["solenoid"]
    b_tesla: float
    radius_m: float
    length_m: float
    center_z_m: float = 0.0
    profile: Literal["hard_edge", "thin_sheet"] = "hard_edge"
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


class XsuiteLatticeElement(ElementBase):
    """Track the cloud through an xtrack.Line declared in the scene (T2).

    The Line is a data artifact referenced by `line_path` (resolved
    relative to the scene file); `p0c_ev` is the always-explicit reference
    momentum x c in eV. Physics/portable config only. `center_z_m` /
    `length_m` are viz hints so the lattice renders without importing
    xtrack. Fidelity tier: externally tracked (Xsuite / xtrack).

    Requires forward-going particles (p_z > 0 for every alive particle):
    the adapter keeps only |p| across the accelerator-coordinate boundary,
    so a wide-angle or isotropic upstream source (e.g. a bare `beta_plus`
    emitter) is rejected at run time. See
    docs/superpowers/specs/2026-07-06-xsuite-lattice-scene-element-design.md.
    """

    type: Literal["xsuite_lattice"]
    line_path: str
    p0c_ev: float = Field(gt=0)
    num_turns: int = Field(default=1, ge=1)
    center_z_m: float = 0.0
    length_m: float | None = Field(default=None, gt=0)


class MatterSlabElement(ElementBase):
    """A slab of NIST material tracked by the vanilla Geant4 engine (M2b).

    Physics/portable config only; the machine-specific transformer binary
    is injected at run time via the LATENT_DIRAC_G4_TRANSFORMER environment
    variable (never stored in a scene). The geometry envelope must match
    the compiled transformer build. Fidelity tier: engine transformer
    (vanilla Geant4 v11.4.2, FTFP_BERT). See
    docs/superpowers/specs/2026-07-06-matter-slab-scene-element-design.md.
    """

    type: Literal["matter_slab"]
    material: str
    thickness_mm: float = Field(gt=0)
    entry_z_m: float = 0.0
    transverse_half_width_m: float = Field(default=0.20, gt=0)
    world_half_length_m: float = Field(default=0.60, gt=0)


class ResidualGasLossElement(ElementBase):
    """Storage lifetime: stochastic annihilation on residual gas over a hold.

    Per-particle exponential survival exp(-hold_time_s / mean_lifetime_s);
    killed particles enter the loss ledger, survivors age by the hold time.
    `mean_lifetime_s` is a direct input (fidelity tier: parameterized); the
    cross-section-derived form tau = 1/(n sigma v) is a future upgrade — see
    docs/superpowers/specs/2026-07-06-residual-gas-storage-lifetime-design.md.
    """

    type: Literal["residual_gas_loss"]
    mean_lifetime_s: float = Field(gt=0)
    hold_time_s: float = Field(ge=0)


class BufferGasCoolingElement(ElementBase):
    """Surko-type buffer-gas cooling region (parameterized stand-in).

    Over `hold_time_s`, each particle undergoes Poisson(collision_rate_hz *
    hold_time_s) collisions; each is either a cooling collision (kinetic
    energy drops by `energy_loss_ev`, floored at (3/2) k_B * gas_temperature_k)
    or a positronium-formation loss (probability `ps_fraction`; particle
    killed and ledgered). Single-channel, constant-rate parameterized tier;
    the energy-dependent cross-section table is a later upgrade — see
    docs/superpowers/specs/2026-07-06-buffer-gas-collisions-design.md.
    """

    type: Literal["buffer_gas_cooling"]
    hold_time_s: float = Field(ge=0)
    collision_rate_hz: float = Field(gt=0)
    energy_loss_ev: float = Field(gt=0)
    ps_fraction: float = Field(ge=0, le=1)
    gas_temperature_k: float = Field(default=300.0, ge=0)


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
    | ResidualGasLossElement
    | MatterSlabElement
    | XsuiteLatticeElement
    | BufferGasCoolingElement
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
