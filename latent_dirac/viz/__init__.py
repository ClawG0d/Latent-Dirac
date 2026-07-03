"""Optional visualization backends.

Importing this package does not import matplotlib or plotly. Backend methods
load optional plotting dependencies only when they are used.
"""

from latent_dirac.viz.base import RendererBackend
from latent_dirac.viz.matplotlib_backend import MatplotlibBackend
from latent_dirac.viz.plotly_backend import PlotlyBackend

__all__ = ["RendererBackend", "MatplotlibBackend", "PlotlyBackend"]
