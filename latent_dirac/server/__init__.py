"""Local sim-engine HTTP API for the desktop client.

A thin FastAPI wrapper over the existing core: validate a scene (the AI's
output contract), run it, and return the text report plus a self-contained
(offline) interactive 3D HTML. Bind to 127.0.0.1 only; simulations run on
the user's machine, never on a remote server. Behind the optional
`[server]` extra.
"""

from __future__ import annotations

from latent_dirac.server.app import create_app

__all__ = ["create_app"]
