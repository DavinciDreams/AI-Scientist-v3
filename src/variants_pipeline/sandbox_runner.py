"""Run a sealed evaluator and promote only canonical bounded output."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.variants_pipeline.sandbox_outputs import (  # noqa: E402
    build_promotable_frame,
    canonicalize_results,
    copy_promotable_results,
    validate_promotable_results,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--evaluator-python", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--program-name", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--evaluator-uid", type=int, required=True)
    parser.add_argument("--evaluator-gid", type=int, required=True)
    args = parser.parse_args()

    process = subprocess.Popen(
        [
            str(args.evaluator_python),
            str(args.evaluator),
            "--program_path",
            str(args.program),
            "--results_dir",
            str(args.results),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=True,
        preexec_fn=lambda: _drop_evaluator_privileges(args.evaluator_uid, args.evaluator_gid),
    )
    returncode = _wait_for_evaluator(process)
    if returncode != 0:
        return returncode

    environment = dict(os.environ)
    canonicalize_results(args.results, args.program_name, environment)
    with tempfile.TemporaryDirectory(prefix="variant-promotion-", dir="/tmp") as temporary:
        staging = Path(temporary)
        copy_promotable_results(args.results, staging)
        validate_promotable_results(staging, environment, args.program_name)
        sys.stdout.buffer.write(build_promotable_frame(staging))
    return 0


def _drop_evaluator_privileges(user_id: int, group_id: int) -> None:
    os.setgroups([])
    os.setgid(group_id)
    os.setuid(user_id)


def _stream_bounded_output(process: subprocess.Popen[bytes]) -> None:
    if process.stdout is None:
        return
    remaining = 1024 * 1024
    truncated = False
    while chunk := process.stdout.read(64 * 1024):
        sanitized = _sanitize_diagnostic(chunk)
        if remaining:
            visible = sanitized[:remaining]
            sys.stderr.write(visible)
            sys.stderr.flush()
            visible_length = len(visible)
            remaining -= visible_length
        else:
            visible_length = 0
        if len(sanitized) > visible_length:
            truncated = True
    if truncated:
        sys.stderr.write("\n[evaluator output truncated after 1 MiB]\n")


def _wait_for_evaluator(process: subprocess.Popen[bytes]) -> int:
    output_errors: list[BaseException] = []

    def stream_output() -> None:
        try:
            _stream_bounded_output(process)
        except BaseException as exc:
            output_errors.append(exc)

    output_thread = threading.Thread(target=stream_output, daemon=True)
    output_thread.start()
    try:
        while process.poll() is None:
            if output_errors:
                raise output_errors[0]
            time.sleep(0.01)
    finally:
        _terminate_remaining_processes(process.pid)

    output_thread.join(timeout=1)
    if output_thread.is_alive():
        raise RuntimeError("evaluator output stream did not close")
    if output_errors:
        raise output_errors[0]
    return process.wait()


def _sanitize_diagnostic(payload: bytes) -> str:
    text = payload.decode("utf-8", errors="backslashreplace")
    return "".join(
        character
        if character in {"\n", "\t"} or (character.isascii() and character.isprintable())
        else f"\\u{ord(character):04x}"
        for character in text
    )


def _terminate_remaining_processes(evaluator_pid: int) -> None:
    try:
        os.killpg(evaluator_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 1.0
    while True:
        process_ids = {
            int(process_dir.name)
            for process_dir in Path("/proc").glob("[0-9]*")
            if int(process_dir.name) not in {1, os.getpid()}
        }
        if not process_ids:
            break
        for process_id in process_ids:
            try:
                os.kill(process_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _reap_children()
        if time.monotonic() >= deadline:
            raise RuntimeError("evaluator descendants did not terminate")
        time.sleep(0.01)


def _reap_children() -> None:
    while True:
        try:
            child, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if child == 0:
            break


if __name__ == "__main__":
    raise SystemExit(main())
