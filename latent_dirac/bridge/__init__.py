"""Local engine bridge: line-delimited JSON over stdin/stdout.

The desktop client spawns `python -m latent_dirac.bridge` (or the frozen
binary) and exchanges one JSON request/response per line — no HTTP, no port,
no web framework. Localhost only by construction: the process talks solely to
its parent over pipes. See `handler.handle_request` for the op contract.
"""

from __future__ import annotations

from latent_dirac.bridge.handler import handle_request

__all__ = ["handle_request"]
