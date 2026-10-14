"""Entry point for the ``agsci`` console script.

Usage::

    agsci run [task_dir] [--dry-run] key=value ...   # the pipeline (src.pipeline.cli)
    agsci investigate key=value ...               # the investigation loop alone (src.investigation.cli)
    agsci doctor [--json] [--image REF]              # host readiness checks
    agsci inspect [build|start|dev] [options]        # run inspector web UI
    agsci review <paper.pdf> [--output review.json]  # standalone OpenAI-backed reviewer

Subcommand modules are imported lazily so ``agsci doctor``, ``agsci inspect``,
and ``agsci review`` do not pay for the pipeline's Hydra imports.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

USAGE = """\
Usage: agsci <command> [arguments]

Commands:
  run          Run the pipeline: agsci run [task_dir] [--dry-run] key=value ...
  investigate  Run the investigation loop alone: agsci investigate [HYDRA_OVERRIDES...]
  doctor       Check this host for the tools, images, and GPUs the pipeline needs
  inspect      Build and serve the run inspector web UI: agsci inspect [build|start|dev]
  review       Review a paper PDF with the OpenAI-backed reviewer: agsci review <paper.pdf> [--output review.json]

Run `agsci <command> --help` for that command's options.
"""


def _looks_like_pipeline_argument(argument: str) -> bool:
    """A task directory, Hydra override, or flag that ``agsci run`` would accept."""
    return (
        "=" in argument
        or argument.startswith("-")
        or "/" in argument
        or Path(argument).is_dir()
    )


def main(argv: list[str] | None = None) -> int:
    """Dispatch to a subcommand and return its exit code."""
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print(USAGE, end="", file=sys.stderr)
        return 2
    command, rest = argv[0], argv[1:]
    if command in {"-h", "--help", "help"}:
        print(USAGE, end="")
        return 0
    if command == "run":
        from src.pipeline import cli as pipeline_cli

        return pipeline_cli.main(rest)
    if command == "investigate":
        from src.investigation import cli as investigation_cli

        return investigation_cli.run_cli(rest)
    if command == "doctor":
        from src.agsci import doctor

        return doctor.main(rest)
    if command == "inspect":
        from src.agsci import inspector

        return inspector.main(rest)
    if command == "review":
        from src.agsci import review

        return review.main(rest)
    print(f"agsci: unknown command {command!r}", file=sys.stderr)
    if _looks_like_pipeline_argument(command):
        print(
            "The pipeline is now the 'run' subcommand: "
            f"agsci run {shlex.join(argv)}".rstrip(),
            file=sys.stderr,
        )
    print(file=sys.stderr)
    print(USAGE, end="", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
