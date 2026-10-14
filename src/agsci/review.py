"""``agsci review``: run the standalone OpenAI-backed reviewer on a paper PDF.

The command wraps ``src/aireview.py``, the same PEP 723 ``uv`` script the
pipeline's final AI review phase runs, without touching paper manifests or
cost ledgers: it validates the PDF, builds the reviewer's restricted
environment (only OpenAI settings are loaded from the repo ``.env`` or
``--env-file``), and executes ``uv run --script src/aireview.py <pdf>`` with
inherited standard streams. The reviewer prints the review JSON itself and
``--output`` additionally writes it to a file.

The ``uv`` lookup and the subprocess runner are injectable so the command is
testable without ``uv`` or the network. Messages never print secret values,
only variable names.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from src.investigation.final_ai_review import final_ai_review_environment

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEWER_SCRIPT = REPO_ROOT / "src" / "aireview.py"

Runner = Callable[[list[str], dict[str, str]], int]


def _run_inherit_streams(command: list[str], environment: dict[str, str]) -> int:
    """Run the reviewer with the caller's stdin/stdout/stderr and return its exit code."""
    return subprocess.run(command, env=environment, check=False).returncode


def reviewer_command(
    pdf_path: Path, *, uv: str, output: Path | None, reviewer_script: Path = REVIEWER_SCRIPT
) -> list[str]:
    """The ``uv run --script`` invocation the pipeline's review phase also uses."""
    command = [uv, "run", "--script", str(reviewer_script), str(pdf_path)]
    if output is not None:
        command.extend(["--output", str(output)])
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agsci review",
        description=(
            "Review a paper PDF with the standalone OpenAI-backed reviewer "
            "(src/aireview.py). Requires uv and OPENAI_API_KEY in the environment "
            "or in the repo .env / --env-file."
        ),
    )
    parser.add_argument("pdf", type=Path, help="paper PDF to review")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="also write the review JSON to this file (it is always printed to stdout)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="env file to load OpenAI settings from (default: <repo>/.env when it exists)",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
    runner: Runner = _run_inherit_streams,
    repo_root: Path = REPO_ROOT,
) -> int:
    """Validate the inputs, build the restricted environment, and run the reviewer."""
    args = _parser().parse_args(argv)
    pdf_path: Path = args.pdf
    if not pdf_path.is_file():
        print(f"agsci review: not a regular file: {pdf_path}", file=sys.stderr)
        return 2

    env_file: Path | None = args.env_file
    if env_file is None:
        default_env_file = repo_root / ".env"
        if default_env_file.is_file():
            env_file = default_env_file
    elif not env_file.is_file():
        print(f"agsci review: env file not found: {env_file}", file=sys.stderr)
        return 2

    source = os.environ if environ is None else environ
    environment = final_ai_review_environment(source, env_file=env_file)
    if not environment.get("OPENAI_API_KEY"):
        where = f" or in {env_file}" if env_file is not None else ""
        print(
            "agsci review: OPENAI_API_KEY is not set in the environment"
            f"{where}; export it or pass --env-file",
            file=sys.stderr,
        )
        return 1

    uv = which("uv")
    if uv is None:
        print(
            "agsci review: 'uv' not found on PATH; the reviewer runs as a uv script "
            "(see docs/install.md)",
            file=sys.stderr,
        )
        return 1

    reviewer_script = repo_root / "src" / "aireview.py"
    command = reviewer_command(pdf_path, uv=uv, output=args.output, reviewer_script=reviewer_script)
    return runner(command, environment)


if __name__ == "__main__":
    raise SystemExit(main())
