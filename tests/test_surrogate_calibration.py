"""Surrogate graduation (M3-b): moment-matched against the engine table.

Calibration standard (owner decision): moment matching — NOT a
distribution-level claim. Design record:
docs/superpowers/specs/2026-07-07-surrogate-calibration-design.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latent_dirac.sources.antiproton_surrogate import CALIBRATIONS, AntiprotonSurrogateSource
from latent_dirac.sources.antiproton_table import _parse_table

TABLE = Path("examples/data/pbar_yield_ftfp_bert_26gevc_ir.csv")
BAND = (3.0, 4.2)


def table_moments():
    header, rows = _parse_table(TABLE)
    momenta = rows[:, 3:6]
    magnitude = np.linalg.norm(momenta, axis=1)
    in_band = (magnitude >= BAND[0]) & (magnitude <= BAND[1])
    selected = momenta[in_band]
    mag = magnitude[in_band]
    theta = np.arctan2(np.hypot(selected[:, 0], selected[:, 1]), selected[:, 2])
    return {
        "yield": float(in_band.sum()) / int(header["n_primaries"]),
        "central": float(mag.mean()),
        "spread": float(mag.std() / mag.mean()),
        "angular_rms": float(np.sqrt(np.mean(theta**2))),
    }


def calibrated_source(macro=20_000):
    return AntiprotonSurrogateSource(
        primary_proton_count=1.0e6,
        source_sigma_m=1e-3,
        bunch_length_s=1e-9,
        macro_particles=macro,
        calibration="ad_ftfp_bert_26gevc_ir",
    )


def test_pinned_angular_rms_matches_the_committed_table():
    # angular rms is sampled directly (no band truncation): the pinned
    # constant must stay the measured table moment; a regenerated table
    # forces a recalibration commit
    moments = table_moments()
    pinned = CALIBRATIONS["ad_ftfp_bert_26gevc_ir"]
    assert pinned["angular_rms_rad"] == pytest.approx(moments["angular_rms"], rel=1e-2)


def test_sampled_moments_match_within_five_percent():
    # the owner-approved criterion, applied to OUTPUTS: the calibrated
    # surrogate's sampled in-band first two momentum moments, and its
    # in-band weighted yield, agree with the engine table within 5%
    # (the pinned constants are EFFECTIVE parameters — see the module
    # comment: the table's in-band spectrum is nearly flat, so the
    # Gaussian width is fitted through the band truncation)
    moments = table_moments()
    state = calibrated_source().sample(np.random.default_rng(11))
    gev = np.linalg.norm(state.momentum_kg_m_s, axis=1) * 299792458.0 / 1.602176634e-19 / 1e9
    in_band = (gev >= BAND[0]) & (gev <= BAND[1])
    assert gev[in_band].mean() == pytest.approx(moments["central"], rel=5e-2)
    assert gev[in_band].std() / gev[in_band].mean() == pytest.approx(moments["spread"], rel=5e-2)
    in_band_yield = float(state.weight[in_band].sum()) / 1.0e6  # per primary
    assert in_band_yield == pytest.approx(moments["yield"], rel=5e-2)


def test_tier_flips_and_provenance_is_carried():
    state = calibrated_source(macro=64).sample(np.random.default_rng(1))
    assert state.metadata["model_type"] == "externally calibrated"
    assert state.metadata["calibration"]["name"] == "ad_ftfp_bert_26gevc_ir"
    assert "moment matching" in state.metadata["calibration"]["standard"]


def test_explicit_params_remain_surrogate_tier():
    source = AntiprotonSurrogateSource(
        primary_proton_count=1.0e6,
        yield_pbar_per_primary_in_acceptance=1e-4,
        central_momentum_GeV_c=3.5,
        momentum_spread_fraction=0.1,
        angular_rms_rad=0.1,
        source_sigma_m=1e-3,
        bunch_length_s=1e-9,
        macro_particles=32,
    )
    assert source.sample(np.random.default_rng(0)).metadata["model_type"] == "surrogate"


def test_calibration_conflicts_and_unknown_names_rejected():
    with pytest.raises(ValueError, match="conflicts"):
        AntiprotonSurrogateSource(
            primary_proton_count=1.0e6,
            central_momentum_GeV_c=3.5,
            source_sigma_m=1e-3,
            bunch_length_s=1e-9,
            macro_particles=32,
            calibration="ad_ftfp_bert_26gevc_ir",
        )
    with pytest.raises(ValueError, match="unknown calibration"):
        AntiprotonSurrogateSource(
            primary_proton_count=1.0e6,
            source_sigma_m=1e-3,
            bunch_length_s=1e-9,
            macro_particles=32,
            calibration="nope",
        )
    with pytest.raises(ValueError, match="missing physics inputs"):
        AntiprotonSurrogateSource(
            primary_proton_count=1.0e6,
            source_sigma_m=1e-3,
            bunch_length_s=1e-9,
            macro_particles=32,
        )
