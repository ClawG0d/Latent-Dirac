"""Matplotlib-based static visualization backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from latent_dirac.core.units import joule_to_ev
from latent_dirac.viz.base import import_optional, particle_cloud_from_result_or_cloud

POSITION_AXES = {"x": 0, "y": 1, "z": 2}
MOMENTUM_AXES = {"px": 0, "py": 1, "pz": 2}


class MatplotlibBackend:
    """Static renderer for quick reports and notebook workflows."""

    name = "matplotlib"

    def _pyplot(self):
        return import_optional("matplotlib.pyplot", "matplotlib")

    def plot_energy_spectrum(self, result_or_cloud):
        cloud = particle_cloud_from_result_or_cloud(result_or_cloud)
        plt = self._pyplot()

        fig, ax = plt.subplots()
        live = cloud.alive
        energies_mev = joule_to_ev(cloud.kinetic_energy_joule()[live]) / 1.0e6
        weights = cloud.weight[live]
        if energies_mev.size:
            ax.hist(energies_mev, bins=min(40, max(5, energies_mev.size)), weights=weights)
        ax.set_xlabel("Kinetic energy [MeV]")
        ax.set_ylabel("Weighted count")
        ax.set_title("Energy spectrum")
        fig.tight_layout()
        return fig

    def plot_phase_space(self, cloud, x_axis: str = "x", p_axis: str = "px"):
        if x_axis not in POSITION_AXES:
            raise ValueError(f"x_axis must be one of {sorted(POSITION_AXES)}")
        if p_axis not in MOMENTUM_AXES:
            raise ValueError(f"p_axis must be one of {sorted(MOMENTUM_AXES)}")

        plt = self._pyplot()
        live = cloud.alive
        x_values = cloud.position_m[live, POSITION_AXES[x_axis]]
        p_values = cloud.momentum_kg_m_s[live, MOMENTUM_AXES[p_axis]]

        fig, ax = plt.subplots()
        ax.scatter(x_values, p_values, s=16, alpha=0.75)
        ax.set_xlabel(f"{x_axis} [m]")
        ax.set_ylabel(f"{p_axis} [kg m/s]")
        ax.set_title("Phase space")
        fig.tight_layout()
        return fig

    def plot_losses_by_stage(self, pipeline_result):
        plt = self._pyplot()
        stage_results = pipeline_result.stage_results
        names = [stage.stage_name for stage in stage_results]
        losses = np.array([stage.losses for stage in stage_results], dtype=float)

        fig, ax = plt.subplots()
        ax.bar(names, losses)
        ax.set_xlabel("Stage")
        ax.set_ylabel("Weighted losses")
        ax.set_title("Losses by stage")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        return fig

    def save_all_basic_report_figures(self, result, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        figures = {
            "energy_spectrum": self.plot_energy_spectrum(result),
            "phase_space": self.plot_phase_space(result.final_cloud),
            "losses_by_stage": self.plot_losses_by_stage(result),
        }
        paths = {}
        for name, figure in figures.items():
            path = output_path / f"{name}.png"
            figure.savefig(path, dpi=150)
            paths[name] = path
        return paths
