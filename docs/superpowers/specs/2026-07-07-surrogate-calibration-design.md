# Surrogate graduation to externally calibrated (M3-b)

Date: 2026-07-07. Status: accepted (owner decision: moment matching —
NOT a distribution-level claim).

## Standard

`AntiprotonSurrogateSource` gains `calibration="ad_ftfp_bert_26gevc_ir"`:
the four physics inputs load from constants derived from the committed
engine yield table (`examples/data/pbar_yield_ftfp_bert_26gevc_ir.csv`),
and the sampled state carries the `externally calibrated` tier with the
calibration name, table reference, band, and standard in metadata.

The criterion applies to OUTPUTS: the calibrated source's sampled
in-band (3.0–4.2 GeV/c) first two momentum moments, and its in-band
weighted yield per primary, agree with the table's measured values
within 5%; angular rms is pinned directly. Explicit-parameter use keeps
the `surrogate` tier; `calibration` and explicit physics values are
mutually exclusive (validated).

## The effective-width subtlety (recorded honestly)

The table's in-band spectrum is nearly flat: its in-band std
(0.3444 GeV/c) sits just below the uniform-distribution limit for the
band ((4.2−3.0)/sqrt(12) = 0.3464), so no narrow Gaussian truncated to
the band can reproduce it — the truncation always narrows a Gaussian
parameterized by the raw in-band moments (first attempt measured 18%
low). The pinned constants are therefore EFFECTIVE parameters, fitted
numerically through the band truncation: mu = 3.405 GeV/c,
sigma/mu = 0.5580 (a wide Gaussian whose band-selected middle is
flat), with the yield constant scaled by the in-band fraction (0.2466)
so the in-band weighted content equals the table's 2.73e-4 per
primary. Out-of-band tails are surrogate form and are not calibrated —
stated in the class docstring, the metadata physics note, and here.

## Validation

`tests/test_surrogate_calibration.py`: sampled in-band moments and
in-band yield vs the table within 5% (the criterion); angular-rms
constant pinned against the table (1%); tier flip and calibration
provenance in metadata; explicit-params path stays `surrogate`;
conflict/unknown-name/missing-input validation. Regenerating the
table forces a recalibration commit (the pins fail loudly).
