"""Minimal antiproton transport demo."""

from __future__ import annotations

import numpy as np

from latent_dirac.beamline.momentum_window import MomentumWindow
from latent_dirac.core.units import momentum_gev_c_to_si
from latent_dirac.diagnostics.reports import text_report
from latent_dirac.fields.uniform import UniformField
from latent_dirac.pipeline.runner import PipelineRunner
from latent_dirac.pipeline.stage import Stage
from latent_dirac.solvers.relativistic_boris import RelativisticBorisSolver
from latent_dirac.sources.antiproton_surrogate import AntiprotonSurrogateSource


def run_demo() -> str:
    primary_count = 50_000
    source = AntiprotonSurrogateSource(
        primary_proton_count=primary_count,
        yield_pbar_per_primary_in_acceptance=2.0e-5,
        central_momentum_GeV_c=3.0,
        momentum_spread_fraction=0.04,
        angular_rms_rad=0.01,
        source_sigma_m=1.0e-3,
        bunch_length_s=2.0e-12,
        macro_particles=512,
    )
    field = UniformField(B_vector_t=np.array([0.0, 0.0, 0.15]))
    solver = RelativisticBorisSolver(dt_s=2.0e-12, steps=100)

    initial_cloud = source.sample(np.random.default_rng(2027))
    result = PipelineRunner(
        stages=[
            Stage("uniform-field transport", lambda cloud: solver.propagate(cloud, field)),
            Stage(
                "momentum window",
                MomentumWindow(
                    momentum_gev_c_to_si(2.5),
                    momentum_gev_c_to_si(3.5),
                ).apply,
            ),
        ]
    ).run(initial_cloud)
    return text_report(result.stage_results, result.final_cloud, primary_count=primary_count)


if __name__ == "__main__":
    print(run_demo())
