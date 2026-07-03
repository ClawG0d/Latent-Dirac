"""Minimal visualization backend protocol and dependency helpers."""

from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable


@runtime_checkable
class RendererBackend(Protocol):
    """Minimal protocol shared by optional renderer backends."""

    name: str


def import_optional(module_name: str, package_name: str):
    """Import an optional visualization dependency with a clear install hint."""

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_name = exc.name or package_name
        if missing_name == package_name or missing_name.startswith(f"{package_name}."):
            raise ImportError(
                f"{package_name} is required for this visualization backend. "
                'Install it with `pip install "latent-dirac[viz]"`.'
            ) from exc
        raise


def particle_cloud_from_result_or_cloud(result_or_cloud):
    """Accept either a ParticleCloud or an object with `final_cloud`."""

    return getattr(result_or_cloud, "final_cloud", result_or_cloud)
