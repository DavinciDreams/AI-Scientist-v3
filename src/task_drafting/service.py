"""Host-owned repair gate service entrypoint."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import os
from pathlib import Path
import signal

from src.task_drafting.checks import GateSettings, run_gate
from src.task_drafting.gate import GateServer, TaskGateReport
from src.investigation.evaluator_tree import MANAGED_RUNTIME_MANIFEST_FILENAME
from src.investigation.task_repair_authorization import DependencyRepairExecutor


class RuntimeRecordingExecutor:
    """Record only a managed runtime materialized during a gate request."""

    def __init__(
        self,
        task_dir: Path,
        marker_path: Path,
        executor: Callable[[str], TaskGateReport],
    ) -> None:
        self._task_dir = task_dir
        self._marker_path = marker_path
        self._executor = executor

    def __call__(self, operation: str) -> TaskGateReport:
        runtime_existed = self._managed_runtime_exists()
        report = self._executor(operation)
        if not runtime_existed and self._managed_runtime_exists():
            self._write_marker()
        return report

    def _managed_runtime_exists(self) -> bool:
        runtime = self._task_dir / ".venv"
        manifest = self._task_dir / MANAGED_RUNTIME_MANIFEST_FILENAME
        return (
            runtime.is_dir()
            and not runtime.is_symlink()
            and manifest.is_file()
            and not manifest.is_symlink()
        )

    def _write_marker(self) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self._marker_path, flags, 0o600)
        except FileExistsError:
            return
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write("gate-created\n")
            stream.flush()
            os.fsync(stream.fileno())


def _main() -> int:
    parser = argparse.ArgumentParser(description="Serve task drafting gates over a Unix socket")
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--source-task-dir", type=Path, required=True)
    parser.add_argument("--containerize", action="store_true")
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--container-network", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--production-entry-timeout-seconds", type=int, required=True)
    parser.add_argument("--max-runtime-bytes", type=int, required=True)
    parser.add_argument("--max-requests", type=int, required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument("--runtime-created-marker", type=Path, required=True)
    parser.add_argument("--install-cache-dir", type=Path)
    parser.add_argument("--repair-snapshot", type=Path)
    parser.add_argument("--repair-transaction", type=Path)
    parser.add_argument("--repair-authorization", type=Path)
    parser.add_argument("--repair-session-id")
    args = parser.parse_args()
    repair_arguments = (
        args.repair_snapshot, args.repair_transaction,
        args.repair_authorization, args.repair_session_id,
    )
    if any(value is not None for value in repair_arguments) and not all(repair_arguments):
        parser.error("repair authorization arguments must be supplied together")
    token = args.token_file.read_text(encoding="ascii").strip()
    settings = GateSettings(
        task_dir=args.task_dir.resolve(strict=True),
        source_task_dir=args.source_task_dir.resolve(strict=True),
        containerize=args.containerize,
        container_image=args.container_image,
        container_network=args.container_network,
        timeout_seconds=args.timeout_seconds,
        production_entry_timeout_seconds=args.production_entry_timeout_seconds,
        max_runtime_bytes=args.max_runtime_bytes,
        install_cache_dir=(
            args.install_cache_dir.resolve(strict=True) if args.install_cache_dir else None
        ),
    )
    gate_executor: Callable[[str], TaskGateReport] = lambda operation: run_gate(operation, settings)
    if args.repair_snapshot is not None:
        gate_executor = DependencyRepairExecutor(
            settings.task_dir, args.repair_snapshot, args.repair_transaction,
            args.repair_authorization, args.repair_session_id, gate_executor,
        )
    execute = RuntimeRecordingExecutor(
        settings.task_dir,
        args.runtime_created_marker,
        gate_executor,
    )
    server = GateServer(
        args.socket,
        token=token,
        executor=execute,
        max_requests=args.max_requests,
        receipt_path=args.receipt_path,
        ready_path=args.ready_file,
    )
    def terminate(_signum: int, _frame: object) -> None:
        server.close()
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, terminate)
    server.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
