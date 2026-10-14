"""Build the generic, candidate-visible search-control runtime."""

from __future__ import annotations

import ast
import shutil
import stat
from pathlib import Path

_MODULES = (
    "evaluate_program.py",
    "evaluation_evidence.py",
    "program_clone.py",
    "program_spec.py",
)
_MAX_MODULE_BYTES = 2 * 1024 * 1024


def _copy_regular_module(source: Path, destination: Path) -> None:
    metadata = source.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_MODULE_BYTES:
        raise ValueError(f"unsafe generic search runtime module: {source.name}")
    shutil.copy2(source, destination, follow_symlinks=False)


def _write_minimal_files_module(source: Path, destination: Path) -> None:
    metadata = source.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_MODULE_BYTES:
        raise ValueError("unsafe generic files module")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=source.name)
    retained_functions = {"evaluation_results_complete", "read_json"}
    tree.body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.ImportFrom)
            and node.module in {"__future__", "pathlib", "typing"}
            or isinstance(node, ast.Import)
            and any(alias.name in {"json", "stat"} for alias in node.names)
            or isinstance(node, ast.FunctionDef)
            and node.name in retained_functions
        )
    ]
    ast.fix_missing_locations(tree)
    rendered = ast.unparse(tree) + "\n"
    compile(rendered, destination.name, "exec")
    destination.write_text(rendered, encoding="utf-8")


def package_search_runtime(repo_root: Path, destination: Path) -> None:
    """Publish only generic harness control modules and a constrained launcher."""
    source_package = repo_root / "src" / "investigation"
    runtime_package = destination / "src" / "investigation"
    runtime_package.mkdir(parents=True)
    (destination / "src" / "__init__.py").write_text("", encoding="utf-8")
    (runtime_package / "__init__.py").write_text("", encoding="utf-8")
    for module_name in _MODULES:
        _copy_regular_module(source_package / module_name, runtime_package / module_name)
    _write_minimal_files_module(source_package / "files.py", runtime_package / "files.py")

    launcher = destination / ".search-runtime" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(
        """#!/bin/sh
set -eu
python_bin="${WRAPPER_PYTHON:-python3}"
if [ "$#" -ge 3 ] && [ "$1" = "-I" ] && [ "$2" = "-m" ]; then
  module="$3"
  shift 3
  case "$module" in
    src.investigation.program_clone)
      exec "$python_bin" -I -c 'import os, sys; sys.path.insert(0, os.environ.get("TASK_SOURCE_DIR", "/task-source")); from src.investigation.program_clone import main; raise SystemExit(main(sys.argv[1:]))' "$@"
      ;;
    src.investigation.evaluate_program)
      echo "contracted task evaluation is harness-owned" >&2
      exit 64
      ;;
  esac
fi
echo "unsupported contracted task Python invocation" >&2
exit 64
""",
        encoding="utf-8",
    )
    launcher.chmod(0o555)
