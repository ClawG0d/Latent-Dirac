"""Surrogate antiproton source term.

This is not a detailed accelerator target model. It samples accepted-source
macro-particles from simple yield, momentum, angular, and source-size inputs.

With ``calibration="ad_ftfp_bert_26gevc_ir"`` the four physics inputs are
loaded from moments measured on the committed engine yield table instead
of being hand-supplied, and the sampled state carries the
``externally calibrated`` fidelity tier with the table's provenance.
Calibration standard (owner decision, 2026-07-07): moment matching — the
surrogate reproduces the table's in-band first two momentum moments,
angular rms, and in-band yield; it does NOT claim distribution-level
agreement (the Gaussian shape is still a surrogate form). Design record:
docs/superpowers/specs/2026-07-07-surrogate-calibration-design.md.
"""

from __future__ import annotations

import numpy as np
from pydantic import field_validator, model_validator

from latent_dirac.core.species import antiproton
from latent_dirac.core.units import momentum_gev_c_to_si
from latent_dirac.sources.base import (
    SourceTerm,
    forward_directions,
    get_rng,
    particle_arrays,
    validate_nonnegative,
    validate_positive,
)
from latent_dirac.state.particle_state import ParticleState

# Effective Gaussian parameters fitted so that the SAMPLED cloud's
# in-band (3.0-4.2 GeV/c) moments reproduce those measured on
# examples/data/pbar_yield_ftfp_bert_26gevc_ir.csv (546 of 2547 rows,
# 2e6 primaries: in-band mean 3.5834 GeV/c, in-band std/mean 0.0961,
# angular rms 0.1457 rad, in-band yield 2.73e-4 per primary). The
# table's in-band spectrum is nearly flat, so a truncated narrow
# Gaussian cannot match its in-band std — the fitted sigma is an
# EFFECTIVE width (a wide Gaussian whose band-selected middle is
# flat), and the yield constant is the total such that the in-band
# weighted content equals the table's (in-band fraction 0.2466).
# Out-of-band tails are surrogate form, not calibrated. Pinned by
# tests against the committed table.
CALIBRATIONS: dict[str, dict[str, float]] = {
    "ad_ftfp_bert_26gevc_ir": {
        "yield_pbar_per_primary_in_acceptance": 1.107e-3,
        "central_momentum_GeV_c": 3.405,
        "momentum_spread_fraction": 0.5580,
        "angular_rms_rad": 0.1457,
    }
}
_CALIBRATED_FIELDS = (
    "yield_pbar_per_primary_in_acceptance",
    "central_momentum_GeV_c",
    "momentum_spread_fraction",
    "angular_rms_rad",
)


class AntiprotonSurrogateSource(SourceTerm):
    primary_proton_count: float
    source_sigma_m: float
    bunch_length_s: float
    macro_particles: int
    calibration: str | None = None
    yield_pbar_per_primary_in_acceptance: float | None = None
    central_momentum_GeV_c: float | None = None
    momentum_spread_fraction: float | None = None
    angular_rms_rad: float | None = None

    @model_validator(mode="after")
    def _resolve_calibration(self):
        explicit = [name for name in _CALIBRATED_FIELDS if getattr(self, name) is not None]
        if self.calibration is not None:
            if self.calibration not in CALIBRATIONS:
                raise ValueError(f"unknown calibration {self.calibration!r}; known: {sorted(CALIBRATIONS)}")
            if explicit:
                raise ValueError(
                    f"calibration {self.calibration!r} conflicts with explicit values for {explicit}; "
                    "supply one or the other"
                )
            for name, value in CALIBRATIONS[self.calibration].items():
                setattr(self, name, value)
        elif len(explicit) != len(_CALIBRATED_FIELDS):
            missing = sorted(set(_CALIBRATED_FIELDS) - set(explicit))
            raise ValueError(f"missing physics inputs {missing} (or set calibration=...)")
        return self

    @field_validator("primary_proton_count", "central_momentum_GeV_c", "source_sigma_m", "macro_particles")
    @classmethod
    def _positive(cls, value, info):
        if value is None:
            return value  # resolved by the calibration validator
        return validate_positive(info.field_name, value)

    @field_validator(
        "yield_pbar_per_primary_in_acceptance",
        "momentum_spread_fraction",
        "angular_rms_rad",
        "bunch_length_s",
    )
    @classmethod
    def _nonnegative(cls, value, info):
        if value is None:
            return value  # resolved by the calibration validator
        return validate_nonnegative(info.field_name, value)

    def sample(self, rng: np.random.Generator | None = None) -> ParticleState:
        rng = get_rng(rng)
        count = int(self.macro_particles)
        total_yield = self.primary_proton_count * self.yield_pbar_per_primary_in_acceptance
        momentum_gev_c = rng.normal(
            self.central_momentum_GeV_c,
            self.central_momentum_GeV_c * self.momentum_spread_fraction,
            size=count,
        )
        momentum_gev_c = np.clip(momentum_gev_c, 1.0e-12, None)
        momentum = momentum_gev_c_to_si(momentum_gev_c)
        directions = forward_directions(rng, count, self.angular_rms_rad)

        if self.calibration is not None:
            model_type = "externally calibrated"
            physics_note = (
                "Surrogate form with moments calibrated against the committed "
                f"engine yield table ({self.calibration}); moment-level "
                "calibration only — the Gaussian shape is not distribution-level."
            )
        else:
            model_type = "surrogate"
            physics_note = "Surrogate source, not a detailed target model."

        metadata = {
            "source": "AntiprotonSurrogateSource",
            "model_type": model_type,
            "physics_note": physics_note,
            "assumptions": {
                "momentum_distribution": "normal GeV/c around central momentum",
                "angular_distribution": "small-angle Gaussian around +z",
                "position_distribution": "3D Gaussian source size",
            },
        }
        if self.calibration is not None:
            metadata["calibration"] = {
                "name": self.calibration,
                "table": "examples/data/pbar_yield_ftfp_bert_26gevc_ir.csv",
                "band_gev_c": [3.0, 4.2],
                "standard": "moment matching (first two momentum moments, angular rms, in-band yield)",
            }

        return ParticleState(
            species=antiproton,
            position_m=rng.normal(0.0, self.source_sigma_m, size=(count, 3)),
            momentum_kg_m_s=momentum[:, np.newaxis] * directions,
            time_s=rng.normal(0.0, self.bunch_length_s, size=count),
            weight=np.full(count, total_yield / count),
            metadata=metadata,
            **particle_arrays(count),
        )
