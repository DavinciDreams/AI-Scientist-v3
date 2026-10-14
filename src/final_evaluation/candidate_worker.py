"""Run the frozen candidate behind a bounded JSON-lines protocol."""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, cast

_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
_CANDIDATE_MODULE_NAME = "frozen_candidate"
_MISSING_MODULE = object()


def _load_handle(program_path: Path) -> Callable[[Any], Any]:
    sys.path.insert(0, str(program_path.parent))
    spec = importlib.util.spec_from_file_location(_CANDIDATE_MODULE_NAME, program_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load candidate entrypoint")
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(_CANDIDATE_MODULE_NAME, _MISSING_MODULE)
    sys.modules[_CANDIDATE_MODULE_NAME] = module
    try:
        with redirect_stdout(sys.stderr):
            spec.loader.exec_module(module)
        handle = getattr(module, "handle", None)
        if not callable(handle):
            raise RuntimeError("candidate must define handle(request)")
    except BaseException:
        if previous_module is _MISSING_MODULE:
            sys.modules.pop(_CANDIDATE_MODULE_NAME, None)
        else:
            sys.modules[_CANDIDATE_MODULE_NAME] = cast(ModuleType, previous_module)
        raise
    return handle


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    handle = _load_handle(Path(sys.argv[1]))
    while raw := sys.stdin.buffer.readline(_MAX_MESSAGE_BYTES + 1):
        if len(raw) > _MAX_MESSAGE_BYTES or not raw.endswith(b"\n"):
            return 2
        try:
            request = json.loads(raw)
            with redirect_stdout(sys.stderr):
                value = handle(request)
            response = {"ok": True, "value": value}
            encoded = json.dumps(response, allow_nan=False, separators=(",", ":")).encode()
        except Exception as exc:
            encoded = json.dumps(
                {"ok": False, "value": f"{type(exc).__name__}: {exc}"},
                separators=(",", ":"),
            ).encode()
        if len(encoded) > _MAX_MESSAGE_BYTES:
            return 2
        sys.stdout.buffer.write(encoded + b"\n")
        sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
