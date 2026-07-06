"""Run the local engine bridge: `python -m latent_dirac.bridge`.

Prints a single `{"ready": true}` line once imports are done (the parent waits
for it), then reads one JSON request per line from stdin and writes one JSON
response per line to stdout, echoing any request `id`. A malformed line or a
handler exception yields an error response and the loop continues — the process
never dies on a single bad request. Localhost only: it speaks only to its
parent over pipes, never the network.
"""

from __future__ import annotations

import json
import sys

from latent_dirac.bridge.handler import handle_request


def _write(stdout, obj) -> None:
    stdout.write(json.dumps(obj) + "\n")
    stdout.flush()


def serve(stdin, stdout) -> None:
    _write(stdout, {"ready": True})  # the parent reads this to know we're up
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _write(stdout, {"ok": False, "error": {"type": "bad_request", "detail": f"invalid JSON: {exc}"}})
            continue
        rid = req.get("id") if isinstance(req, dict) else None
        try:
            resp = handle_request(req)
        except Exception as exc:  # never let one request kill the loop
            resp = {"ok": False, "error": {"type": "internal", "detail": str(exc)}}
        if rid is not None:
            resp = {"id": rid, **resp}
        _write(stdout, resp)


def main(argv: list[str] | None = None) -> int:
    serve(sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
