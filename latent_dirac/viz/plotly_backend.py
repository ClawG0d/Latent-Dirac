"""Plotly-based interactive visualization backend."""

from __future__ import annotations

import numpy as np

from latent_dirac.viz.base import import_optional


class PlotlyBackend:
    """Interactive renderer for trajectories, phase space, and loss summaries."""

    name = "plotly"

    def _graph_objects(self):
        return import_optional("plotly.graph_objects", "plotly")

    def plot_trajectory_3d(self, trajectory, max_particles: int = 200):
        if max_particles <= 0:
            raise ValueError("max_particles must be positive")

        go = self._graph_objects()
        positions = trajectory.position_m
        particle_count = min(positions.shape[1], max_particles)

        fig = go.Figure()
        for index in range(particle_count):
            particle_positions = positions[:, index, :]
            fig.add_trace(
                go.Scatter3d(
                    x=particle_positions[:, 0],
                    y=particle_positions[:, 1],
                    z=particle_positions[:, 2],
                    mode="lines",
                    name=f"particle {index}",
                    showlegend=index < 10,
                )
            )
        fig.update_layout(
            scene={
                "xaxis_title": "x [m]",
                "yaxis_title": "y [m]",
                "zaxis_title": "z [m]",
            },
            title="Particle trajectories",
        )
        return fig

    def plot_phase_space_interactive(self, cloud):
        go = self._graph_objects()
        live = cloud.alive
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=cloud.position_m[live, 0],
                    y=cloud.momentum_kg_m_s[live, 0],
                    mode="markers",
                    marker={"size": 7, "opacity": 0.75},
                )
            ]
        )
        fig.update_layout(
            title="Phase space",
            xaxis_title="x [m]",
            yaxis_title="px [kg m/s]",
        )
        return fig

    def plot_losses_interactive(self, pipeline_result):
        go = self._graph_objects()
        names = [stage.stage_name for stage in pipeline_result.stage_results]
        losses = np.array([stage.losses for stage in pipeline_result.stage_results], dtype=float)
        fig = go.Figure(data=[go.Bar(x=names, y=losses)])
        fig.update_layout(
            title="Losses by stage",
            xaxis_title="Stage",
            yaxis_title="Weighted losses",
        )
        return fig
