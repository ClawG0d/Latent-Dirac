"""Launch the local sim engine: `python -m latent_dirac.server`.

Binds 127.0.0.1 on an ephemeral port (or --port), prints the chosen port
as `PORT <n>` on stdout so a parent process (the Electron main process)
can read it, then serves. Localhost only — simulations run on the user's
machine, never remotely.
"""

from __future__ import annotations

import argparse
import socket
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="latent-dirac-serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port", type=int, default=0, help="0 (default) picks a free ephemeral port"
    )
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            'the local sim engine needs the [server] extra: pip install "latent-dirac[server]"'
        ) from exc

    from latent_dirac.server import create_app

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    port = sock.getsockname()[1]
    print(f"PORT {port}", flush=True)  # the Electron main process reads this

    server = uvicorn.Server(uvicorn.Config(create_app(), log_level="warning"))
    server.run(sockets=[sock])
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
