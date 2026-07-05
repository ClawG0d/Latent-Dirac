"""3D field-magnitude rendering for table-based field maps (Plotly backend)."""

from __future__ import annotations

import numpy as np

from latent_dirac.fields.field_map import FieldMapField
from latent_dirac.viz.base import import_optional

_FIDELITY_TEXT = "fidelity: table-based field map"


def render_field_magnitude_3d(
    field_map: FieldMapField,
    opacity: float = 0.15,
    surface_count: int = 12,
):
    """Render |B| of a field map as a translucent Plotly volume."""

    go = import_optional("plotly.graph_objects", "plotly")

    grid_x, grid_y, grid_z = np.meshgrid(field_map.x_m, field_map.y_m, field_map.z_m, indexing="ij")
    magnitude = np.linalg.norm(field_map.B_t, axis=-1)

    figure = go.Figure(
        go.Volume(
            x=grid_x.ravel(),
            y=grid_y.ravel(),
            z=grid_z.ravel(),
            value=magnitude.ravel(),
            opacity=opacity,
            surface_count=surface_count,
            name=f"|B| [T] ({_FIDELITY_TEXT})",
            hovertext=_FIDELITY_TEXT,
            colorbar={"title": "|B| [T]"},
        )
    )
    figure.update_layout(
        scene={
            "xaxis_title": "x [m]",
            "yaxis_title": "y [m]",
            "zaxis_title": "z [m]",
        }
    )
    return figure
