"""PyInstaller entry point for the frozen local sim engine.

The desktop client spawns this frozen binary as its sidecar. It behaves
exactly like `python -m latent_dirac.server`: bind 127.0.0.1 on an ephemeral
port, print `PORT <n>` on stdout, then serve the FastAPI app. See
desktop/packaging/engine.spec and desktop/packaging/README.md.
"""

from __future__ import annotations

import sys

from latent_dirac.server.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
