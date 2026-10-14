"""Trusted final-evaluator launch marker written before evaluator execution."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

_START_TOKEN = b"evaluator-started-v1"


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        raise SystemExit("final evaluator launcher requires a command")
    witness_path = Path(os.environ.get("AGENTIC_EVALUATOR_STARTED_SOCKET", ""))
    if witness_path != Path("/broker/evaluator-start.sock"):
        raise SystemExit("invalid final evaluator launch witness")
    witness = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        witness.connect(str(witness_path))
        witness.sendall(_START_TOKEN)
    finally:
        witness.close()
    os.execv(arguments[0], arguments)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
