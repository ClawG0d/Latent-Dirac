"""Minimal positron capture demo."""

from __future__ import annotations

import numpy as np

from latent_dirac.beamline.aperture import Aperture
from latent_dirac.beamline.momentum_window import MomentumWindow
from latent_dirac.core.units import momentum_gev_c_to_si
from latent_dirac.diagnostics.reports import text_report
from latent_dirac.fields.solenoid import SolenoidField
from latent_dirac.pipeline.runner import PipelineRunner
from latent_dirac.pipeline.stage import Stage
from latent_dirac.solvers.relativistic_boris import RelativisticBorisSolver
from latent_dirac.sources.positron_pair import PositronPairSource


def run_demo() -> str:
    primary_count = 10_000
    source = PositronPairSource(
        primary_count=primary_count,
        yield_eplus_per_primary=0.02,
        mean_energy_MeV=3.0,
        energy_spread_MeV=0.4,
        angular_rms_rad=0.03,
        source_sigma_m=1.0e-3,
        bunch_length_s=1.0e-12,
        macro_particles=512,
    )
    field = SolenoidField(b_tesla=0.8, radius_m=0.05, length_m=0.5)
    solver = RelativisticBorisSolver(dt_s=2.0e-12, steps=100)

    initial_cloud = source.sample(np.random.default_rng(2026))
    result = PipelineRunner(
        stages=[
            Stage("solenoid transport", lambda cloud: solver.propagate(cloud, field)),
            Stage("aperture", Aperture(radius_m=0.04, z_m=0.06).apply),
            Stage(
                "momentum window",
                MomentumWindow(
                    momentum_gev_c_to_si(0.001),
                    momentum_gev_c_to_si(0.020),
                ).apply,
            ),
        ]
    ).run(initial_cloud)
    return text_report(result.stage_results, result.final_cloud, primary_count=primary_count)


if __name__ == "__main__":
    print(run_demo())
