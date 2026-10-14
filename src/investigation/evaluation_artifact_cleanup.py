"""Cleanup for evaluator outputs created outside the managed archive."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import os
from pathlib import Path
import re
import secrets
import shutil
import stat


_IDEA_DIR_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EVALUATOR_PARENT_RE = re.compile(r"^\.dev_eval_program_[0-9]+$")
_EVALUATOR_RUN_RE = re.compile(r"^run_[0-9]{3,}_.+$")
_RUN_RESULT_MARKERS = frozenset({"program_results.json", "run_summary.json"})
_RUN_LOG_MARKERS = frozenset({"program_stderr.txt", "program_stdout.txt"})


@dataclass(frozen=True)
class CleanupReport:
    removed_roots: tuple[Path, ...]
    removed_bytes: int
    failures: tuple[str, ...] = ()


@dataclass
class ExperimentDirectoryGuard:
    """Pinned results hierarchy retained across untrusted agent execution."""

    results_dir: Path
    ideas_dir: Path
    experiment_dir: Path
    _results_fd: int
    _ideas_fd: int

    @classmethod
    def prepare(cls, results_dir: Path, idea_dir_name: str) -> ExperimentDirectoryGuard:
        if not _IDEA_DIR_NAME_RE.fullmatch(idea_dir_name):
            raise ValueError(f"Unsafe idea directory name: {idea_dir_name!r}")
        results_root = results_dir.expanduser().resolve()
        results_root.mkdir(parents=True, exist_ok=True)
        results_fd = _open_directory(results_root)
        try:
            try:
                os.mkdir("ideas", mode=0o700, dir_fd=results_fd)
            except FileExistsError:
                pass
            ideas_fd = _open_child_directory(results_fd, "ideas")
        except BaseException:
            os.close(results_fd)
            raise
        experiment_dir = results_root / "ideas" / idea_dir_name
        try:
            _require_optional_child_directory(
                ideas_fd,
                idea_dir_name,
                path=experiment_dir,
                label="experiment directory",
            )
        except BaseException:
            os.close(ideas_fd)
            os.close(results_fd)
            raise
        return cls(
            results_dir=results_root,
            ideas_dir=results_root / "ideas",
            experiment_dir=experiment_dir,
            _results_fd=results_fd,
            _ideas_fd=ideas_fd,
        )

    def open_experiment(self) -> int | None:
        _require_same_directory(self.results_dir, self._results_fd)
        _require_same_directory(self.ideas_dir, self._ideas_fd)
        try:
            experiment_fd = _open_child_directory(
                self._ideas_fd,
                self.experiment_dir.name,
            )
        except FileNotFoundError:
            return None
        try:
            _require_same_directory(self.experiment_dir, experiment_fd)
        except BaseException:
            os.close(experiment_fd)
            raise
        return experiment_fd

    def close(self) -> None:
        os.close(self._ideas_fd)
        os.close(self._results_fd)


def cleanup_unmanaged_evaluation_outputs(
    experiment_dir: Path,
    *,
    experiment_fd: int | None = None,
) -> CleanupReport:
    """Remove proven evaluator runs from explicitly named development scratch roots."""
    root = experiment_dir.absolute()
    removed_roots: list[Path] = []
    removed_bytes = 0
    failures: list[str] = []
    root_fd = (
        os.dup(experiment_fd)
        if experiment_fd is not None
        else _open_directory(_real_directory(root, label="experiment directory"))
    )
    if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
        os.close(root_fd)
        raise NotADirectoryError(f"experiment directory must be real: {experiment_dir}")
    try:
        for parent_name in os.listdir(root_fd):
            if not _EVALUATOR_PARENT_RE.fullmatch(parent_name):
                continue
            try:
                parent_fd = _open_child_directory(root_fd, parent_name)
            except OSError as exc:
                failures.append(_failure(parent_name, exc))
                continue
            try:
                try:
                    runs_fd = _open_child_directory(parent_fd, "runs")
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    failures.append(_failure(f"{parent_name}/runs", exc))
                    continue
                try:
                    removed, removed_size, run_failures = _cleanup_runs(
                        runs_fd,
                        experiment_dir / parent_name / "runs",
                    )
                finally:
                    os.close(runs_fd)
                removed_roots.extend(removed)
                removed_bytes += removed_size
                failures.extend(run_failures)
                try:
                    os.rmdir("runs", dir_fd=parent_fd)
                except OSError as exc:
                    if exc.errno not in {errno.ENOENT, errno.ENOTEMPTY}:
                        failures.append(_failure(f"{parent_name}/runs", exc))
            finally:
                os.close(parent_fd)
    finally:
        os.close(root_fd)
    return CleanupReport(tuple(removed_roots), removed_bytes, tuple(failures))


def _cleanup_runs(
    runs_fd: int,
    runs_path: Path,
) -> tuple[list[Path], int, list[str]]:
    removed: list[Path] = []
    removed_bytes = 0
    failures: list[str] = []
    for run_name in os.listdir(runs_fd):
        if not _EVALUATOR_RUN_RE.fullmatch(run_name):
            continue
        try:
            run_fd = _open_child_directory(runs_fd, run_name)
        except OSError as exc:
            failures.append(_failure(str(runs_path / run_name), exc))
            continue
        if not _is_evaluator_run(run_fd):
            os.close(run_fd)
            continue
        verified_stat = os.fstat(run_fd)
        run_bytes = _regular_file_bytes(run_fd)
        quarantine_name = f".cleanup_{secrets.token_hex(8)}"
        try:
            os.rename(
                run_name,
                quarantine_name,
                src_dir_fd=runs_fd,
                dst_dir_fd=runs_fd,
            )
            quarantined_stat = os.stat(
                quarantine_name,
                dir_fd=runs_fd,
                follow_symlinks=False,
            )
            if (quarantined_stat.st_dev, quarantined_stat.st_ino) != (
                verified_stat.st_dev,
                verified_stat.st_ino,
            ):
                os.rename(
                    quarantine_name,
                    run_name,
                    src_dir_fd=runs_fd,
                    dst_dir_fd=runs_fd,
                )
                failures.append(f"{runs_path / run_name}: directory changed before cleanup")
                continue
            shutil.rmtree(quarantine_name, dir_fd=runs_fd)
        except OSError as exc:
            failures.append(_failure(str(runs_path / run_name), exc))
            continue
        finally:
            os.close(run_fd)
        removed.append(runs_path / run_name)
        removed_bytes += run_bytes
    return removed, removed_bytes, failures


def _is_evaluator_run(run_fd: int) -> bool:
    regular_names: set[str] = set()
    for name in os.listdir(run_fd):
        try:
            entry_stat = os.stat(name, dir_fd=run_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISREG(entry_stat.st_mode):
            regular_names.add(name)
    has_output = bool(regular_names & _RUN_RESULT_MARKERS) or _RUN_LOG_MARKERS <= regular_names
    return "config.json" in regular_names and has_output


def _regular_file_bytes(directory_fd: int) -> int:
    total = 0
    for name in os.listdir(directory_fd):
        try:
            entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISREG(entry_stat.st_mode):
            total += entry_stat.st_size
        elif stat.S_ISDIR(entry_stat.st_mode):
            try:
                child_fd = _open_child_directory(directory_fd, name)
            except OSError:
                continue
            try:
                total += _regular_file_bytes(child_fd)
            finally:
                os.close(child_fd)
    return total


def _require_optional_child_directory(
    parent_fd: int,
    name: str,
    *,
    path: Path,
    label: str,
) -> None:
    try:
        mode = os.stat(name, dir_fd=parent_fd, follow_symlinks=False).st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(mode):
        raise NotADirectoryError(f"{label} must be a real directory: {path}")


def _require_same_directory(path: Path, directory_fd: int) -> None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        raise RuntimeError(f"Directory changed during agent execution: {path}") from None
    descriptor_stat = os.fstat(directory_fd)
    if not stat.S_ISDIR(path_stat.st_mode) or (
        path_stat.st_dev,
        path_stat.st_ino,
    ) != (descriptor_stat.st_dev, descriptor_stat.st_ino):
        raise RuntimeError(f"Directory changed during agent execution: {path}")


def _real_directory(path: Path, *, label: str) -> Path:
    expanded = path.expanduser()
    try:
        mode = expanded.lstat().st_mode
    except FileNotFoundError:
        raise NotADirectoryError(f"{label} does not exist: {expanded}") from None
    if not stat.S_ISDIR(mode):
        raise NotADirectoryError(f"{label} must be a real directory: {expanded}")
    return expanded.absolute()


def _open_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags)


def _open_child_directory(parent_fd: int, name: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    return os.open(name, flags, dir_fd=parent_fd)


def _failure(path: str, exc: OSError) -> str:
    return f"{path}: {type(exc).__name__}"
