"""Latent Dirac AI gateway: natural language -> scene JSON.

Standalone deployable (not part of the latent-dirac package). Turns a
user's prompt plus the engine's Scene JSON Schema into a candidate scene
via a Claude tool call, and accepts prior validation errors for a
corrective retry. The simulation itself never runs here — only scene
generation. The Anthropic key lives server-side (env ANTHROPIC_API_KEY).

`create_app` lives in `services.ai_gateway.app`; it is imported there
(not here) so importing this package does not require FastAPI — the test
module's `importorskip("fastapi")` can then skip cleanly on environments
without the [server] extra.
"""

from __future__ import annotations
