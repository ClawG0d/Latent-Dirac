"""PyInstaller entry point for the frozen local engine (stdio bridge).

The desktop client spawns this frozen binary as its sidecar. It behaves exactly
like `python -m latent_dirac.bridge`: print {"ready": true}, then read one JSON
request per line on stdin and write one JSON response per line on stdout — no
HTTP, no port. See desktop/packaging/engine.spec and desktop/packaging/README.md.
"""

from __future__ import annotations

import sys

from latent_dirac.bridge.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
