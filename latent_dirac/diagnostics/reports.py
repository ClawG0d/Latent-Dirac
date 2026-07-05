"""Text reports for demos and quick diagnostics."""

from __future__ import annotations

from latent_dirac.diagnostics.accepted_yield import accepted_yield_from_cloud
from latent_dirac.diagnostics.spectra import energy_spectrum_summary
from latent_dirac.pipeline.stage import StageResult
from latent_dirac.state.particle_state import ParticleState


def text_report(
    stage_results: list[StageResult],
    final_cloud: ParticleState,
    primary_count: float | None = None,
) -> str:
    lines = ["Latent Dirac simulation report", ""]
    lines.append("Stage accounting:")
    for result in stage_results:
        lines.append(
            f"- {result.stage_name}: input={result.input_weighted_count:.6g}, "
            f"output={result.output_weighted_count:.6g}, "
            f"transmission={result.transmission:.6g}, losses={result.losses:.6g}"
        )

    spectrum = energy_spectrum_summary(final_cloud)
    lines.extend(
        [
            "",
            "Accepted cloud:",
            f"- weighted count: {final_cloud.weighted_count():.6g}",
            f"- mean kinetic energy: {spectrum['mean_energy_MeV']:.6g} MeV",
        ]
    )
    if primary_count is not None:
        lines.append(f"- accepted yield: {accepted_yield_from_cloud(final_cloud, primary_count):.6g}")
    return "\n".join(lines)
