import importlib
import sys

import numpy as np
import pytest

from latent_dirac.beamline.aperture import Aperture
from latent_dirac.core.species import positron
from latent_dirac.core.units import kinetic_energy_to_momentum_magnitude, mev_to_joule
from latent_dirac.pipeline.runner import PipelineResult
from latent_dirac.pipeline.stage import StageResult
from latent_dirac.state.particle_cloud import ParticleCloud
from latent_dirac.state.trajectory import Trajectory


CORE_MODULES = (
    "latent_dirac.core.constants",
    "latent_dirac.core.units",
    "latent_dirac.core.species",
    "latent_dirac.state.particle_cloud",
    "latent_dirac.sources.positron_pair",
    "latent_dirac.fields.uniform",
    "latent_dirac.solvers.relativistic_boris",
    "latent_dirac.beamline.aperture",
    "latent_dirac.pipeline.runner",
    "latent_dirac.diagnostics.accepted_yield",
)


def clear_optional_viz_modules():
    for module_name in list(sys.modules):
        if module_name == "matplotlib" or module_name.startswith("matplotlib."):
            del sys.modules[module_name]
        if module_name == "plotly" or module_name.startswith("plotly."):
            del sys.modules[module_name]


def make_cloud() -> ParticleCloud:
    momentum = kinetic_energy_to_momentum_magnitude(mev_to_joule(np.array([1.0, 2.0, 3.0])), positron.mass_kg)
    return ParticleCloud(
        species=positron,
        position_m=np.array([[0.0, 0.0, 0.0], [0.01, 0.02, 0.03], [0.02, 0.01, 0.04]]),
        momentum_kg_m_s=np.column_stack([momentum, momentum * 0.1, momentum * 0.0]),
        time_s=np.zeros(3),
        weight=np.ones(3),
        alive=np.array([True, True, False]),
        particle_id=np.arange(3),
        parent_id=np.full(3, -1),
        metadata={},
    )


def make_pipeline_result() -> PipelineResult:
    return PipelineResult(
        final_cloud=make_cloud(),
        stage_results=[
            StageResult(
                stage_name="aperture",
                input_weighted_count=3.0,
                output_weighted_count=2.0,
                transmission=2.0 / 3.0,
                losses=1.0,
            ),
            StageResult(
                stage_name="momentum",
                input_weighted_count=2.0,
                output_weighted_count=2.0,
                transmission=1.0,
                losses=0.0,
            ),
        ],
    )


def make_trajectory() -> Trajectory:
    return Trajectory(
        time_s=np.array([0.0, 1.0, 2.0]),
        position_m=np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.01, 0.0]],
                [[0.01, 0.0, 0.0], [0.0, 0.02, 0.01]],
                [[0.02, 0.01, 0.0], [0.0, 0.03, 0.02]],
            ]
        ),
        momentum_kg_m_s=np.zeros((3, 2, 3)),
    )


def test_core_imports_do_not_import_visualization_packages():
    clear_optional_viz_modules()

    for module_name in CORE_MODULES:
        importlib.import_module(module_name)

    imported = set(sys.modules)
    assert "matplotlib" not in imported
    assert "plotly" not in imported
    assert not any(name.startswith("matplotlib.") for name in imported)
    assert not any(name.startswith("plotly.") for name in imported)


def test_existing_core_objects_do_not_depend_on_visualization_packages():
    clear_optional_viz_modules()

    aperture = Aperture(radius_m=0.1, z_m=0.0)
    assert aperture.apply(make_cloud()).weighted_count() == 2.0

    imported = set(sys.modules)
    assert "matplotlib" not in imported
    assert "plotly" not in imported


def test_matplotlib_backend_raises_clear_import_error_when_missing(monkeypatch):
    from latent_dirac.viz.matplotlib_backend import MatplotlibBackend

    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "matplotlib.pyplot":
            raise ModuleNotFoundError("No module named 'matplotlib'")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="matplotlib.*latent-dirac\\[viz\\]"):
        MatplotlibBackend().plot_energy_spectrum(make_cloud())


def test_plotly_backend_raises_clear_import_error_when_missing(monkeypatch):
    from latent_dirac.viz.plotly_backend import PlotlyBackend

    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "plotly.graph_objects":
            raise ModuleNotFoundError("No module named 'plotly'")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="plotly.*latent-dirac\\[viz\\]"):
        PlotlyBackend().plot_losses_interactive(make_pipeline_result())


def test_matplotlib_backend_returns_figures_when_installed():
    pytest.importorskip("matplotlib.pyplot")
    from latent_dirac.viz.matplotlib_backend import MatplotlibBackend

    backend = MatplotlibBackend()
    cloud = make_cloud()
    result = make_pipeline_result()

    energy_fig = backend.plot_energy_spectrum(cloud)
    phase_fig = backend.plot_phase_space(cloud)
    losses_fig = backend.plot_losses_by_stage(result)

    assert energy_fig.__class__.__name__ == "Figure"
    assert phase_fig.__class__.__name__ == "Figure"
    assert losses_fig.__class__.__name__ == "Figure"


def test_plotly_backend_returns_figures_when_installed():
    pytest.importorskip("plotly.graph_objects")
    from latent_dirac.viz.plotly_backend import PlotlyBackend

    backend = PlotlyBackend()
    result = make_pipeline_result()

    trajectory_fig = backend.plot_trajectory_3d(make_trajectory())
    phase_fig = backend.plot_phase_space_interactive(make_cloud())
    losses_fig = backend.plot_losses_interactive(result)

    assert trajectory_fig.__class__.__name__ == "Figure"
    assert phase_fig.__class__.__name__ == "Figure"
    assert losses_fig.__class__.__name__ == "Figure"
