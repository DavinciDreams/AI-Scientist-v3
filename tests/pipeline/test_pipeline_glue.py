"""Pipeline glue: run/paper task snapshots, aggregated CSV links, memory copies, and the paper workspace."""

import errno
import json
import os
import shutil
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Callable

import pytest

from src.investigation.evaluator_tree import managed_evaluator_tree_digest
from src.investigation.files import task_copy_ignore
from src.pipeline import safe_fs, task_publication, task_snapshot, task_tree, task_tree_merge
from src.pipeline.config import PipelineConfig
from src.pipeline.glue import (
    UnsafeGluePathError,
    copy_investigation_memory,
    link_aggregated_csvs,
    populate_task_dir,
    prepare_run_task,
    preview_run_task,
    read_task_snapshot_name,
    reuse_paper_task_dir,
    reuse_run_task,
    run_glue,
)
from src.pipeline.paper_workspace import (
    WRITABLE_GLOBAL_MEMORY_FILES,
    PaperContainerMountPlan,
    paper_container_mount_plan,
    validate_existing_global_memory_files,
    workspace_validation_errors,
    write_workspace_marker,
)
from src.pipeline.paths import PipelinePaths
from src.pipeline.task_publication import manifest_digest
from src.pipeline.task_snapshot import reseal_run_task, task_snapshot_is_complete

METADATA = task_snapshot.TASK_SNAPSHOT_METADATA_FILENAME
CSV_NAMES = ("run_summary.csv", "metric_values.csv", "timeseries.csv")
UTF8 = {"encoding": "utf-8"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _paths(tmp_path: Path, task_name: str = "circpack") -> PipelinePaths:
    """Paths rooted at ``tmp_path`` (created so sections can use ``tmp_path / name`` roots)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    investigation = tmp_path / "investigation"
    ablations = tmp_path / "ablations"
    return PipelinePaths(
        task_name=task_name,
        run_id="20260101_000000",
        source_task_dir=tmp_path / "task_src",
        task_dir=investigation / "task",
        investigation_results_dir=investigation,
        ablations_output_dir=ablations,
        ablations_aggregated_task_dir=ablations / "aggregated" / task_name,
        final_output_dir=investigation / "final_evaluation",
        posthoc_output_dir=investigation / "posthoc" / "test-run",
        paper_task_dir=ablations / "task",
        paper_output_dir=ablations / "paper" / "final",
    )


def _write_files(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, **UTF8)


def _source(
    tmp_path: Path,
    files: dict[str, str] | None = None,
    *,
    task_name: str = "circpack",
    mode: int | None = None,
) -> PipelinePaths:
    """Paths whose ``source_task_dir`` exists and holds ``files`` (default: task.md)."""
    paths = _paths(tmp_path, task_name)
    if mode is None:
        paths.source_task_dir.mkdir()
    else:
        paths.source_task_dir.mkdir(mode=mode)
    _write_files(paths.source_task_dir, {"task.md": "source"} if files is None else files)
    return paths


def _run_task(tmp_path: Path, files: dict[str, str] | None = None) -> PipelinePaths:
    """Paths whose run ``task_dir`` exists and holds ``files`` (default: task.md)."""
    paths = _paths(tmp_path)
    paths.task_dir.mkdir(parents=True)
    _write_files(paths.task_dir, {"task.md": "trusted"} if files is None else files)
    return paths


def _memory(tmp_path: Path, files: dict[str, str] | None = None) -> PipelinePaths:
    """Paths with an investigation workspace (holding ``files``) and an ablations output dir."""
    paths = _paths(tmp_path)
    paths.investigation_workspace_dir.mkdir(parents=True)
    paths.ablations_output_dir.mkdir()
    _write_files(paths.investigation_workspace_dir, files or {})
    return paths


def _redirect(tmp_path: Path) -> tuple[Path, Path]:
    """``controlled/redirect`` symlink pointing at ``outside``; returns (redirect, real_output)."""
    controlled_dir = tmp_path / "controlled"
    controlled_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    real_output = outside_dir / "real-output"
    real_output.mkdir()
    (controlled_dir / "redirect").symlink_to(outside_dir, target_is_directory=True)
    return controlled_dir / "redirect" / "real-output", real_output


def _disable_exclusive_rename(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable exclusive rename syscalls so publication takes the portable path."""
    monkeypatch.setattr(safe_fs, "_renameat2_no_replace", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(safe_fs, "_renameatx_np_no_replace", lambda *_args, **_kwargs: False)


@pytest.fixture
def portable_rename(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_exclusive_rename(monkeypatch)


def _unsupported_link(*_args: object, **_kwargs: object) -> None:
    raise OSError(errno.EOPNOTSUPP, "hard links unsupported")


def _assert_rejected(operation: Callable[[], object], match: str, absent: Path) -> None:
    with pytest.raises(UnsafeGluePathError, match=match):
        operation()
    assert not absent.exists()


def _publish_outcome(paths: PipelinePaths) -> str:
    try:
        prepare_run_task(paths)
    except UnsafeGluePathError:
        return "lost"
    return "won"


def _swapping_open(monkeypatch, target: str, name: str, swap: Callable[[], None], *, flag: int):
    """Patch ``target`` os.open to run ``swap`` once before the first matching open of ``name``."""
    real_open = os.open
    swapped = False

    def swap_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == name and flags & flag and not swapped:
            swapped = True
            swap()
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(target, swap_before_open)


def _after_nth_call(real: Callable, on_call: int, action: Callable[[], object]) -> Callable:
    """Wrap ``real`` so that ``action`` runs right after its ``on_call``-th invocation."""
    calls = 0

    def wrapper(*args, **kwargs):
        nonlocal calls
        result = real(*args, **kwargs)
        calls += 1
        if calls == on_call:
            action()
        return result

    return wrapper


def _after_each_call(real: Callable, action: Callable[[], object]) -> Callable:
    """Wrap ``real`` so that ``action`` runs right after every invocation."""

    def wrapper(*args, **kwargs):
        result = real(*args, **kwargs)
        action()
        return result

    return wrapper


# ---------------------------------------------------------------------------
# prepare_run_task / preview_run_task / reuse_run_task
# ---------------------------------------------------------------------------


def test_prepare_run_task_copies_source_honors_ignore_file_and_persists_metadata(tmp_path: Path) -> None:
    # copies the source tree and honors .agentic_task_ignore
    paths = _source(
        tmp_path / "ignore",
        {
            "task.md": "source",
            "keep.txt": "keep",
            "ignored.txt": "ignore",
            ".agentic_task_ignore": "ignored.txt\n",
        },
    )

    prepare_run_task(paths)

    assert (paths.task_dir / "task.md").read_text(**UTF8) == "source"
    assert (paths.task_dir / "keep.txt").is_file()
    assert not (paths.task_dir / "ignored.txt").exists()
    assert (paths.source_task_dir / "ignored.txt").read_text(**UTF8) == "ignore"

    # persists the original task name and source identity in the metadata
    paths = _source(tmp_path / "metadata", task_name="original-task")

    prepare_run_task(paths)

    payload = json.loads((paths.task_dir / METADATA).read_text(**UTF8))
    assert payload["task_name"] == "original-task"
    assert payload["version"] == 2
    assert len(payload["source_digest"]) == 64
    assert payload["source_root_mode"] == stat.S_IMODE(paths.source_task_dir.stat().st_mode)

    # dry run creates nothing
    paths = _source(tmp_path / "dry-run")

    preview_run_task(paths)

    assert not paths.investigation_results_dir.exists()


def test_prepare_run_task_preserves_symlinks_and_modes_and_skips_ignored_external_links(tmp_path: Path) -> None:
    # preserves an internal relative symlink
    paths = _source(tmp_path / "symlink", {"target.txt": "target"})
    (paths.source_task_dir / "linked.txt").symlink_to("target.txt")

    prepare_run_task(paths)

    assert paths.task_dir.joinpath("linked.txt").is_symlink()
    assert os.readlink(paths.task_dir / "linked.txt") == "target.txt"

    # copies a read-only directory with its mode
    paths = _source(tmp_path / "read-only", {"resources/data.txt": "data"})
    read_only = paths.source_task_dir / "resources"
    read_only.chmod(0o555)

    try:
        prepare_run_task(paths)
    finally:
        read_only.chmod(0o755)

    copied = paths.task_dir / "resources"
    assert (copied / "data.txt").read_text(**UTF8) == "data"
    assert copied.stat().st_mode & 0o777 == 0o555

    # skips an ignored directory holding an external symlink
    paths = _source(tmp_path / "ignored-external", {".agentic_task_ignore": ".venv/\n"})
    (paths.source_task_dir / ".venv").mkdir()
    (paths.source_task_dir / ".venv" / "external").symlink_to(tmp_path / "outside")

    prepare_run_task(paths)

    assert not (paths.task_dir / ".venv").exists()


def test_reseal_run_task_publishes_drafted_tree_and_preserves_evaluator_digest(
    tmp_path: Path,
) -> None:
    paths = _source(tmp_path)
    prepare_run_task(paths)
    (paths.task_dir / "evaluate.py").write_text("drafted", **UTF8)
    qualified_digest = managed_evaluator_tree_digest(paths.task_dir)

    reseal_run_task(paths)

    assert task_snapshot_is_complete(paths.task_dir)
    assert managed_evaluator_tree_digest(paths.task_dir) == qualified_digest
    reuse_run_task(paths)


MALFORMED_METADATA_CASES = [
    ("missing-version", json.dumps({"task_name": "t"}), "version"),
    ("bool-version", json.dumps({"task_name": "t", "version": True}), "version"),
    ("future-version", json.dumps({"task_name": "t", "version": 3}), "version"),
    ("str-version", json.dumps({"task_name": "t", "version": "1"}), "version"),
    ("oversized", " " * 5000, "too large"),
    ("deep-nesting", "[" * 1500 + "]" * 1500, "invalid|unsupported version"),
]


def test_task_snapshot_metadata_rejects_malformed_files_and_opens_nonblocking(tmp_path: Path, monkeypatch) -> None:
    # malformed metadata files are rejected
    for name, content, match in MALFORMED_METADATA_CASES:
        snapshot = tmp_path / name / "task"
        snapshot.mkdir(parents=True)
        (snapshot / METADATA).write_text(content, **UTF8)

        with pytest.raises(UnsafeGluePathError, match=match):
            read_task_snapshot_name(snapshot)

    # the metadata file is opened non-blocking
    snapshot = tmp_path / "nonblocking" / "task"
    snapshot.mkdir(parents=True)
    (snapshot / METADATA).write_text('{"task_name":"original-task","version":1}', **UTF8)
    real_open = os.open

    def require_nonblocking(path, flags, *args, **kwargs):
        if path == METADATA:
            assert flags & os.O_NONBLOCK
        return real_open(path, flags, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.task_snapshot.os.open", require_nonblocking)
        assert read_task_snapshot_name(snapshot) == "original-task"

    # an empty task snapshot is incomplete
    (tmp_path / "task").mkdir()
    assert not task_snapshot.task_snapshot_is_complete(tmp_path / "task")


def _ignored_dir_symlink(source: Path, tmp_path: Path) -> None:
    (tmp_path / "outside").mkdir()
    (source / "cache").symlink_to(tmp_path / "outside", target_is_directory=True)
    (source / ".agentic_task_ignore").write_text("cache/\n", **UTF8)


def _chained_escape(source: Path, _tmp_path: Path) -> None:
    (source / "a").symlink_to(".")
    (source / "x").symlink_to("a/../outside-secret")


def _symlink_cycle(source: Path, _tmp_path: Path) -> None:
    (source / "a").symlink_to("b")
    (source / "b").symlink_to("a")


def _fifo(source: Path, _tmp_path: Path) -> None:
    os.mkfifo(source / "pipe")


def _absolute_file_symlink(source: Path, tmp_path: Path) -> None:
    (tmp_path / "outside.txt").write_text("outside", **UTF8)
    (source / "linked.txt").symlink_to(tmp_path / "outside.txt")


UNSAFE_SOURCE_CASES = [
    ("ignored-dir-symlink", _ignored_dir_symlink, "absolute symlink", preview_run_task),
    ("chained-escape", _chained_escape, "escaping symlink", preview_run_task),
    ("symlink-cycle", _symlink_cycle, "invalid symlink", preview_run_task),
    ("fifo", _fifo, "unsupported entry", preview_run_task),
    ("absolute-symlink", _absolute_file_symlink, "absolute symlink", prepare_run_task),
]


def test_run_task_rejects_unsafe_source_entries_and_ignore_pattern_does_not_follow_symlink(
    tmp_path: Path,
) -> None:
    for name, layout, match, operation in UNSAFE_SOURCE_CASES:
        root = tmp_path / name
        paths = _source(root, {})
        layout(paths.source_task_dir, root)

        with pytest.raises(UnsafeGluePathError, match=match):
            operation(paths)

    # a directory ignore pattern does not follow a symlink
    source = tmp_path / "task"
    source.mkdir()
    (tmp_path / "outside").mkdir()
    (source / "cache").symlink_to(tmp_path / "outside", target_is_directory=True)

    assert task_copy_ignore(source, ("cache/",))(str(source), ["cache"]) == set()


def test_run_task_validation_rejects_trees_changed_while_validating(tmp_path: Path, monkeypatch) -> None:
    # reuse_run_task rejects a symlink replaced after readlink
    paths = _run_task(tmp_path / "readlink")
    link = paths.task_dir / "link"
    link.symlink_to("task.md")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", **UTF8)
    real_readlink = os.readlink
    swapped = False

    def replace_after_readlink(path, *args, **kwargs):
        nonlocal swapped
        target = real_readlink(path, *args, **kwargs)
        if path == "link" and not swapped:
            swapped = True
            link.unlink()
            link.symlink_to(outside)
        return target

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.safe_fs.os.readlink", replace_after_readlink)
        with pytest.raises(UnsafeGluePathError, match="changed while validating"):
            reuse_run_task(paths)

    # preview / reuse reject an unreadable tree
    def fail_listdir(_directory_fd):
        raise PermissionError("injected unreadable tree")

    for operation in (preview_run_task, reuse_run_task):
        paths = _paths(tmp_path / operation.__name__)
        target = paths.source_task_dir if operation is preview_run_task else paths.task_dir
        target.mkdir(parents=True)
        (target / "task.md").write_text("trusted", **UTF8)

        with monkeypatch.context() as m:
            m.setattr("src.pipeline.safe_fs.os.listdir", fail_listdir)
            with pytest.raises(UnsafeGluePathError, match="changed while validating"):
                operation(paths)


def test_run_task_resume_and_snapshot_presence_rules(tmp_path: Path) -> None:
    # resume reuses the snapshot without refreshing from the source
    paths = _source(tmp_path / "reuse", {"task.md": "new source"})
    paths.task_dir.mkdir(parents=True)
    (paths.task_dir / "task.md").write_text("run snapshot", **UTF8)

    reuse_run_task(paths)

    assert (paths.task_dir / "task.md").read_text(**UTF8) == "run snapshot"

    # prepare resumes an existing empty snapshot
    paths = _source(tmp_path / "empty")
    paths.task_dir.mkdir(parents=True)

    prepare_run_task(paths)

    assert (paths.task_dir / "task.md").read_text(**UTF8) == "source"
    assert task_snapshot.task_snapshot_ready_path(paths.task_dir).is_file()

    # a destination inside the source is rejected
    paths = _source(tmp_path / "overlap", {})
    nested_results = paths.source_task_dir / "results"
    overlapping = replace(paths, investigation_results_dir=nested_results, task_dir=nested_results / "task")

    with pytest.raises(UnsafeGluePathError, match="must not overlap"):
        prepare_run_task(overlapping)

    # resume requires an existing snapshot
    paths = _source(tmp_path / "missing", {})

    with pytest.raises(UnsafeGluePathError, match="does not exist"):
        reuse_run_task(paths)


def test_prepare_run_task_rejects_source_entries_swapped_during_copy(tmp_path: Path, monkeypatch) -> None:
    # a directory swapped to a symlink
    paths = _source(tmp_path / "dir-swap", {"nested/trusted.txt": "trusted"})
    nested = paths.source_task_dir / "nested"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", **UTF8)

    def swap_dir() -> None:
        nested.rename(paths.source_task_dir / "nested-original")
        nested.symlink_to(outside, target_is_directory=True)

    with monkeypatch.context() as m:
        _swapping_open(m, "src.pipeline.task_tree.os.open", "nested", swap_dir, flag=os.O_DIRECTORY)
        _assert_rejected(lambda: prepare_run_task(paths), "changed while", paths.task_dir)

    # a file added after the copy listing
    paths = _source(tmp_path / "file-added")
    late_paths = paths
    list_calls = 0
    real_list_directory = task_tree._list_directory

    def add_file_after_copy_listing(directory_fd, relative_dir):
        nonlocal list_calls
        names = real_list_directory(directory_fd, relative_dir)
        list_calls += 1
        if list_calls == 2:
            (late_paths.source_task_dir / "late.txt").write_text("late", **UTF8)
        return names

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.task_tree._list_directory", add_file_after_copy_listing)
        _assert_rejected(lambda: prepare_run_task(paths), "changed while copying", paths.task_dir)

    # a regular file replaced before it is opened
    paths = _source(tmp_path / "file-replaced", {"task.md": "trusted"})
    source_file = paths.source_task_dir / "task.md"
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("replacement", **UTF8)

    def swap_file() -> None:
        source_file.unlink()
        replacement.rename(source_file)

    with monkeypatch.context() as m:
        _swapping_open(m, "src.pipeline.task_tree.os.open", "task.md", swap_file, flag=os.O_NOFOLLOW)
        _assert_rejected(lambda: prepare_run_task(paths), "changed while copying", paths.task_dir)


SOURCE_MUTATION_CASES = [
    ("in-place-write", "copyfileobj", 1, "task.md"),
    ("earlier-file-while-copying-later", "copyfileobj", 2, "a.txt"),
    ("during-final-digest", "file_digest", 1, "task.md"),
    ("earlier-file-during-verification", "file_digest", 2, "a.txt"),
]


def test_prepare_run_task_rejects_source_mutation_during_copy_verification_and_validation(
    tmp_path: Path, monkeypatch
) -> None:
    # mutation while copying / digesting
    for name, hook, mutate_on_call, mutated_file in SOURCE_MUTATION_CASES:
        files = {"task.md": "trusted"} if mutated_file == "task.md" else {"a.txt": "trusted", "b.txt": "later"}
        paths = _source(tmp_path / name, files)
        real = shutil.copyfileobj if hook == "copyfileobj" else task_tree.hashlib.file_digest
        mutated = paths.source_task_dir / mutated_file
        mutate_after_call = _after_nth_call(real, mutate_on_call, lambda: mutated.write_text("changed", **UTF8))

        with monkeypatch.context() as m:
            if hook == "copyfileobj":
                m.setattr("src.pipeline.task_tree.shutil.copyfileobj", mutate_after_call)
            else:
                m.setattr(task_tree.hashlib, "file_digest", mutate_after_call)
            _assert_rejected(lambda: prepare_run_task(paths), "changed while copying", paths.task_dir)

    # root mode changed after verification
    paths = _source(tmp_path / "root-mode", {"task.md": "trusted"}, mode=0o700)
    source_root = paths.source_task_dir
    change_root_mode = _after_each_call(task_snapshot.verify_copied_task_tree, lambda: source_root.chmod(0o777))

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.task_snapshot.verify_copied_task_tree", change_root_mode)
        _assert_rejected(lambda: prepare_run_task(paths), "changed while copying", paths.task_dir)

    # source changed (file added / file changed) during staged validation
    for name, late_file in (("file-added", "late.txt"), ("file-changed", "task.md")):
        paths = _source(tmp_path / f"staged-{name}", {"task.md": "trusted"})
        late = paths.source_task_dir / late_file
        mutate_source = _after_each_call(
            task_snapshot.validate_tree_symlinks_at, partial(late.write_text, "changed", **UTF8)
        )

        with monkeypatch.context() as m:
            m.setattr("src.pipeline.task_snapshot.validate_tree_symlinks_at", mutate_source)
            _assert_rejected(lambda: prepare_run_task(paths), "changed while copying", paths.task_dir)


CHMOD_FAILURE_CASES: list[tuple[str, Callable[[Path], PipelinePaths], Callable[[PipelinePaths], object], str, str, str]] = [
    (
        "prepare-run-task",
        lambda tmp: _source(tmp, {"task.md": "trusted"}),
        prepare_run_task,
        "src.pipeline.task_snapshot.os.fchmod",
        "task_dir",
        "investigation_results_dir",
    ),
    (
        "populate-task-dir",
        _run_task,
        lambda paths: populate_task_dir(paths, dry_run=False),
        "src.pipeline.glue.os.fchmod",
        "paper_task_dir",
        "ablations_output_dir",
    ),
]


def test_prepare_run_task_staging_root_stays_private_and_chmod_failure_does_not_publish(
    tmp_path: Path, monkeypatch
) -> None:
    # the staging root stays private until publication
    paths = _source(tmp_path / "private", {"task.md": "trusted"}, mode=0o777)
    paths.source_task_dir.chmod(0o777)
    real_validate = task_snapshot.validate_tree_symlinks_at

    def require_private_staging(parent_fd, name, *, role):
        staging_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        assert stat.S_IMODE(staging_stat.st_mode) == 0o700
        return real_validate(parent_fd, name, role=role)

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.task_snapshot.validate_tree_symlinks_at", require_private_staging)
        prepare_run_task(paths)

    assert stat.S_IMODE(paths.task_dir.stat().st_mode) == 0o777

    # a chmod failure on the final root does not publish (run task and paper task)
    real_fchmod = os.fchmod

    def fail_final_root_mode(fd, mode):
        if stat.S_ISDIR(os.fstat(fd).st_mode) and mode == 0o755:
            raise OSError("injected chmod failure")
        return real_fchmod(fd, mode)

    for name, make_paths, operation, fchmod_target, destination, staging_root in CHMOD_FAILURE_CASES:
        paths = make_paths(tmp_path / name)
        (paths.source_task_dir if destination == "task_dir" else paths.task_dir).chmod(0o755)

        with monkeypatch.context() as m:
            m.setattr(fchmod_target, fail_final_root_mode)
            _assert_rejected(lambda: operation(paths), "permissions", getattr(paths, destination))
        assert not list(getattr(paths, staging_root).glob(".task.*.tmp")), name


def test_prepare_run_task_opens_source_files_nonblocking_and_normalizes_manifest_open_race(
    tmp_path: Path, monkeypatch
) -> None:
    real_open = os.open

    # regular files are opened non-blocking
    paths = _source(tmp_path / "nonblocking", {"task.md": "trusted"})

    def require_nonblocking(path, flags, *args, **kwargs):
        if path == "task.md" and flags & os.O_ACCMODE == os.O_RDONLY:
            assert flags & os.O_NONBLOCK
        return real_open(path, flags, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.task_tree.os.open", require_nonblocking)
        prepare_run_task(paths)

    # a file swapped to a symlink during the manifest open is rejected
    paths = _source(tmp_path / "manifest-race", {"task.md": "trusted"})
    source_file = paths.source_task_dir / "task.md"
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", **UTF8)
    source_open_calls = 0

    def swap_during_manifest_open(path, flags, *args, **kwargs):
        nonlocal source_open_calls
        if path == "task.md" and flags & os.O_ACCMODE == os.O_RDONLY:
            source_open_calls += 1
            if source_open_calls == 2:
                source_file.unlink()
                source_file.symlink_to(outside)
        return real_open(path, flags, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.task_tree.os.open", swap_during_manifest_open)
        _assert_rejected(lambda: prepare_run_task(paths), "changed while copying", paths.task_dir)


def test_prepare_run_task_ignore_file_short_reads_and_changed_rules(tmp_path: Path, monkeypatch) -> None:
    files = {".agentic_task_ignore": "secret.txt\n", "secret.txt": "secret", "task.md": "trusted"}

    # short reads of the ignore file are handled
    paths = _source(tmp_path / "short-reads", files)
    real_read = os.read

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.task_snapshot.os.read", lambda fd, size: real_read(fd, min(size, 3)))
        prepare_run_task(paths)

    assert not (paths.task_dir / "secret.txt").exists()

    # ignore rules changed during the copy are rejected
    paths = _source(tmp_path / "changed-rules", files)
    ignore_file = paths.source_task_dir / ".agentic_task_ignore"
    change_ignore_rules = _after_each_call(
        task_tree.copy_task_tree_at, lambda: ignore_file.write_text("other.txt\n", **UTF8)
    )

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.task_snapshot.copy_task_tree_at", change_ignore_rules)
        _assert_rejected(lambda: prepare_run_task(paths), "ignore patterns changed", paths.task_dir)


# ---------------------------------------------------------------------------
# atomic / portable publication
# ---------------------------------------------------------------------------

DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def test_rename_directory_no_replace_keeps_existing_and_fails_closed_without_exclusive_rename(
    tmp_path: Path, monkeypatch
) -> None:
    # an existing destination directory is never replaced
    root = tmp_path / "existing"
    (root / "staged").mkdir(parents=True)
    (root / "task").mkdir()
    marker = root / "task" / "marker.txt"
    marker.write_text("keep", **UTF8)
    parent_fd = os.open(root, DIR_FLAGS)
    try:
        with pytest.raises(UnsafeGluePathError, match="already exists"):
            safe_fs.rename_directory_no_replace(parent_fd, "staged", "task", role="fresh run task snapshot")
    finally:
        os.close(parent_fd)

    assert marker.read_text(**UTF8) == "keep"

    # without an exclusive rename syscall the publication fails closed
    root = tmp_path / "portable"
    (root / "staged").mkdir(parents=True)
    parent_fd = os.open(root, DIR_FLAGS)
    try:
        with monkeypatch.context() as m:
            _disable_exclusive_rename(m)
            with pytest.raises(UnsafeGluePathError, match="not supported"):
                safe_fs.rename_directory_no_replace(parent_fd, "staged", "task", role="fresh run task snapshot")
    finally:
        os.close(parent_fd)

    assert (root / "staged").is_dir()
    assert not (root / "task").exists()


@pytest.mark.usefixtures("portable_rename")
def test_prepare_run_task_portable_publication_copies_trees_read_only_roots_and_without_hard_links(
    tmp_path: Path, monkeypatch
) -> None:
    # nested trees are published without an exclusive rename
    paths = _source(tmp_path / "nested", {"task.md": "trusted", "resources/data.txt": "nested"})

    prepare_run_task(paths)

    assert (paths.task_dir / "task.md").read_text(**UTF8) == "trusted"
    assert (paths.task_dir / "resources" / "data.txt").read_text(**UTF8) == "nested"
    assert task_snapshot.task_snapshot_ready_path(paths.task_dir).is_file()
    reuse_run_task(paths)

    # a read-only source root is preserved
    paths = _source(tmp_path / "read-only", {"task.md": "trusted"})
    paths.source_task_dir.chmod(0o555)

    try:
        prepare_run_task(paths)
        reuse_run_task(paths)
        assert stat.S_IMODE(paths.task_dir.stat().st_mode) == 0o555
    finally:
        paths.source_task_dir.chmod(0o755)
        if paths.task_dir.exists():
            paths.task_dir.chmod(0o755)

    # publication works without hard links
    paths = _source(tmp_path / "no-hard-links", {"task.md": "trusted"})
    with monkeypatch.context() as m:
        m.setattr(task_tree_merge.os, "link", _unsupported_link)
        m.setattr(task_publication.os, "link", _unsupported_link)
        prepare_run_task(paths)

    assert (paths.task_dir / "task.md").read_text(**UTF8) == "trusted"
    reuse_run_task(paths)


def test_publication_races_select_one_winner(tmp_path: Path, monkeypatch) -> None:
    # portable publication selects one concurrent winner
    paths = _source(tmp_path / "concurrent", {"task.md": "trusted"})
    with monkeypatch.context() as m:
        _disable_exclusive_rename(m)
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = sorted(executor.map(lambda _index: _publish_outcome(paths), range(2)))

    assert outcomes == ["lost", "won"]
    reuse_run_task(paths)

    # no-hard-link publication serializes the active copy behind the lock
    paths = _source(tmp_path / "serialized", {"task.md": "trusted payload"})
    real_copy = task_tree_merge.copy_regular_file_no_replace_at
    copy_paused = threading.Event()
    release_copy = threading.Event()
    second_lock_attempted = threading.Event()
    lock_calls = 0
    lock_calls_guard = threading.Lock()
    real_flock = task_publication.fcntl.flock

    def observed_flock(descriptor: int, operation: int) -> None:
        nonlocal lock_calls
        with lock_calls_guard:
            lock_calls += 1
            if lock_calls == 2:
                second_lock_attempted.set()
        real_flock(descriptor, operation)

    def paused_copy(*args, **kwargs) -> None:
        if args[3] == "task.md" and not copy_paused.is_set():
            copy_paused.set()
            assert release_copy.wait(timeout=5)
        real_copy(*args, **kwargs)

    with monkeypatch.context() as m:
        _disable_exclusive_rename(m)
        m.setattr(task_tree_merge.os, "link", _unsupported_link)
        m.setattr(task_publication.fcntl, "flock", observed_flock)
        m.setattr(task_tree_merge, "copy_regular_file_no_replace_at", paused_copy)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(_publish_outcome, paths)
            assert copy_paused.wait(timeout=5)
            second = executor.submit(_publish_outcome, paths)
            assert second_lock_attempted.wait(timeout=5)
            assert not second.done()
            release_copy.set()
            outcomes = sorted([first.result(timeout=5), second.result(timeout=5)])

    assert outcomes == ["lost", "won"]
    reuse_run_task(paths)

    # the portable completion marker selects one barrier-raced winner
    destination = tmp_path / "marker" / "task"
    destination.mkdir(parents=True)
    (destination / METADATA).write_text("metadata")
    parent_fd = os.open(destination.parent, DIR_FLAGS)
    destination_fd = os.open(destination, DIR_FLAGS)
    barrier = threading.Barrier(2)

    def publish(_index: int) -> str:
        barrier.wait(timeout=5)
        try:
            task_publication._publish_ready_marker(parent_fd, "task", destination_fd, role="task snapshot")
        except UnsafeGluePathError:
            return "lost"
        return "won"

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = sorted(executor.map(publish, range(2)))
    finally:
        os.close(destination_fd)
        os.close(parent_fd)

    assert outcomes == ["lost", "won"]
    assert task_snapshot.task_snapshot_ready_path(destination).is_file()


@pytest.mark.usefixtures("portable_rename")
def test_portable_publication_syncs_nested_directories_and_claim_before_destination(
    tmp_path: Path, monkeypatch
) -> None:
    real_fsync = os.fsync

    # no-hard-link publication fsyncs every nested directory
    paths = _source(tmp_path / "nested", {"task.md": "trusted", "resources/data.txt": "nested"})
    synced_directories: set[tuple[int, int]] = set()

    def record_directory_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if stat.S_ISDIR(metadata.st_mode):
            synced_directories.add((metadata.st_dev, metadata.st_ino))
        real_fsync(descriptor)

    with monkeypatch.context() as m:
        m.setattr(task_tree_merge.os, "link", _unsupported_link)
        m.setattr(task_tree_merge.os, "fsync", record_directory_fsync)
        prepare_run_task(paths)

    expected = {
        (directory.stat().st_dev, directory.stat().st_ino)
        for directory in (paths.task_dir, paths.task_dir / "resources")
    }
    assert expected <= synced_directories

    # the claim is synced before the destination is mutated
    paths = _source(tmp_path / "claim", {"task.md": "trusted"})
    events: list[str] = []
    real_mkdir = os.mkdir

    def record_mkdir(name, *args, **kwargs) -> None:
        if isinstance(name, str) and ".agentic-task-source-" in name:
            events.append("claim")
        elif name == paths.task_dir.name:
            events.append("destination")
        real_mkdir(name, *args, **kwargs)

    def record_claim_fsync(descriptor: int) -> None:
        if events and events[-1] == "claim":
            events.append("claim-parent-sync")
        real_fsync(descriptor)

    with monkeypatch.context() as m:
        m.setattr(task_publication.os, "mkdir", record_mkdir)
        m.setattr(task_publication.os, "fsync", record_claim_fsync)
        prepare_run_task(paths)

    claim_index = events.index("claim")
    sync_index = events.index("claim-parent-sync", claim_index + 1)
    destination_index = events.index("destination", sync_index + 1)
    assert claim_index < sync_index < destination_index


def test_prepare_run_task_recovers_interrupted_publication(tmp_path: Path, monkeypatch) -> None:
    # an interrupted no-hard-link (portable) copy is recovered on the next attempt
    paths = _source(tmp_path / "portable", {"task.md": "trusted payload"})
    real_copy = task_tree_merge.copy_regular_file_no_replace_at
    interrupted = False

    def interrupt_task_copy(source_parent_fd, source_name, destination_parent_fd, destination_name, **kwargs):
        nonlocal interrupted
        if destination_name == "task.md" and not interrupted:
            interrupted = True
            descriptor = os.open(
                destination_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=destination_parent_fd,
            )
            os.write(descriptor, b"partial")
            os.close(descriptor)
            raise OSError("injected interruption")
        return real_copy(source_parent_fd, source_name, destination_parent_fd, destination_name, **kwargs)

    with monkeypatch.context() as m:
        _disable_exclusive_rename(m)
        m.setattr(task_tree_merge.os, "link", _unsupported_link)
        m.setattr(task_tree_merge, "copy_regular_file_no_replace_at", interrupt_task_copy)
        with pytest.raises(UnsafeGluePathError, match="could not publish"):
            prepare_run_task(paths)

        m.setattr(task_tree_merge, "copy_regular_file_no_replace_at", real_copy)
        prepare_run_task(paths)

    assert (paths.task_dir / "task.md").read_text(**UTF8) == "trusted payload"
    reuse_run_task(paths)

    # a native publish interrupted before the ready marker is recovered
    paths = _source(tmp_path / "native", {"task.md": "trusted"})

    def interrupt_ready(*_args, **_kwargs) -> None:
        raise OSError("injected interruption")

    with monkeypatch.context() as m:
        m.setattr(task_publication, "_publish_ready_marker", interrupt_ready)
        with pytest.raises(UnsafeGluePathError, match="could not publish"):
            prepare_run_task(paths)

    assert paths.task_dir.is_dir()
    assert not list(paths.task_dir.parent.glob(".task.agentic_task_ready.*"))

    prepare_run_task(paths)

    reuse_run_task(paths)


def test_ready_marker_is_bound_to_task_directory_inode_and_content(tmp_path: Path) -> None:
    # a recreated task directory does not reuse the stale ready marker
    paths = _source(tmp_path / "recreated", {"task.md": "trusted"})
    prepare_run_task(paths)
    original = paths.task_dir.with_name("original-task")
    paths.task_dir.rename(original)
    paths.task_dir.mkdir(mode=stat.S_IMODE(original.stat().st_mode))
    shutil.copyfile(original / METADATA, paths.task_dir / METADATA)

    with pytest.raises(UnsafeGluePathError, match="source digest"):
        task_snapshot.task_snapshot_is_complete(paths.task_dir)

    # an orphaned ready marker from an old inode is ignored
    paths = _source(tmp_path / "orphaned", {"task.md": "trusted"})
    prepare_run_task(paths)
    shutil.rmtree(paths.task_dir)

    prepare_run_task(paths)

    reuse_run_task(paths)

    # completion survives a copy without hard links
    paths = _source(tmp_path / "copied", {"task.md": "trusted"})
    prepare_run_task(paths)
    source_marker = task_snapshot.task_snapshot_ready_path(paths.task_dir)
    restored_task = tmp_path / "copied" / "restored" / "task"
    shutil.copytree(paths.task_dir, restored_task)
    shutil.copy2(source_marker, restored_task.parent / source_marker.name)

    assert task_snapshot.task_snapshot_is_complete(restored_task)
    restored_task.chmod(0o700)
    assert task_snapshot.task_snapshot_publication_is_complete(restored_task)

    # a replaced ready marker is rejected on reuse
    paths = _source(tmp_path / "replaced", {"task.md": "trusted"})
    prepare_run_task(paths)
    marker = task_snapshot.task_snapshot_ready_path(paths.task_dir)
    marker.unlink()
    marker.write_text("unsafe", **UTF8)

    with pytest.raises(UnsafeGluePathError, match="does not match"):
        reuse_run_task(paths)


def test_prepare_run_task_handles_incomplete_destinations(tmp_path: Path, monkeypatch) -> None:
    # a matching incomplete portable copy is resumed
    paths = _source(tmp_path / "matching", {"a.txt": "already copied", "task.md": "still missing"})
    paths.task_dir.mkdir(parents=True, mode=0o700)
    source_fd = os.open(paths.source_task_dir, DIR_FLAGS)
    try:
        source_manifest = task_tree.task_tree_manifest_at(source_fd, relative_dir=Path(), ignore_patterns=())
    finally:
        os.close(source_fd)
    incomplete_metadata = paths.task_dir / METADATA
    incomplete_metadata.write_text(
        json.dumps(
            {
                "source_digest": manifest_digest(source_manifest),
                "source_root_mode": stat.S_IMODE(paths.source_task_dir.stat().st_mode),
                "task_name": paths.task_name,
                "version": 2,
            },
            separators=(",", ":"),
        )
        + "\n",
        **UTF8,
    )
    incomplete_metadata.chmod(0o600)
    (paths.task_dir / "a.txt").write_text("already copied", **UTF8)

    with monkeypatch.context() as m:
        _disable_exclusive_rename(m)
        prepare_run_task(paths)

    assert (paths.task_dir / "task.md").read_text(**UTF8) == "still missing"
    assert task_snapshot.task_snapshot_ready_path(paths.task_dir).is_file()

    # a mismatched incomplete portable copy is rejected and left untouched
    paths = _source(tmp_path / "mismatched", {"task.md": "trusted"})
    paths.task_dir.mkdir(parents=True, mode=0o700)
    (paths.task_dir / "task.md").write_text("attacker", **UTF8)

    with monkeypatch.context() as m:
        _disable_exclusive_rename(m)
        with pytest.raises(UnsafeGluePathError, match="does not match|conflict"):
            prepare_run_task(paths)

    assert (paths.task_dir / "task.md").read_text(**UTF8) == "attacker"
    assert not list(paths.task_dir.parent.glob(".task.agentic_task_ready.*"))

    # a rejected unclaimed destination keeps its permissions
    paths = _source(tmp_path / "unclaimed", {"task.md": "trusted"})
    paths.task_dir.mkdir(parents=True)
    (paths.task_dir / "task.md").write_text("conflict", **UTF8)
    paths.task_dir.chmod(0o555)

    try:
        with pytest.raises(UnsafeGluePathError, match="unclaimed incomplete"):
            prepare_run_task(paths)
        assert stat.S_IMODE(paths.task_dir.stat().st_mode) == 0o555
    finally:
        paths.task_dir.chmod(0o755)


FAILED_PUBLICATION_CASES = [
    ("run-task", "source_task_dir", prepare_run_task, "task_dir", "investigation_results_dir"),
    (
        "paper-task",
        "task_dir",
        lambda paths: populate_task_dir(paths, dry_run=False),
        "paper_task_dir",
        "ablations_output_dir",
    ),
]


def test_failed_task_publication_cleans_read_only_temporary_tree(tmp_path: Path, monkeypatch) -> None:
    def fail_publication(*_args, **_kwargs) -> None:
        raise UnsafeGluePathError("injected publication failure")

    for name, source_attr, operation, destination, staging_root in FAILED_PUBLICATION_CASES:
        paths = _paths(tmp_path / name)
        read_only = getattr(paths, source_attr) / "resources"
        read_only.mkdir(parents=True)
        (read_only / "data.txt").write_text("data", **UTF8)
        read_only.chmod(0o555)

        with monkeypatch.context() as m:
            m.setattr(task_publication, "rename_directory_between_no_replace", fail_publication)
            try:
                with pytest.raises(UnsafeGluePathError, match="injected publication failure"):
                    operation(paths)
            finally:
                read_only.chmod(0o755)

        assert not getattr(paths, destination).exists(), name
        assert not list(getattr(paths, staging_root).glob(".task.*.tmp")), name


# ---------------------------------------------------------------------------
# populate_task_dir / reuse_paper_task_dir
# ---------------------------------------------------------------------------


def _populate(paths: PipelinePaths) -> None:
    populate_task_dir(paths, dry_run=False)


def test_populate_task_dir_copies_run_task_with_identity_metadata_and_symlinks(
    tmp_path: Path, monkeypatch
) -> None:
    # copies the run task
    paths = _run_task(tmp_path / "copy", {"task.md": "hello"})
    _populate(paths)
    assert (paths.paper_task_dir / "task.md").read_text() == "hello"

    # uses portable publication without an exclusive rename
    paths = _run_task(tmp_path / "portable", {"task.md": "hello"})
    with monkeypatch.context() as m:
        _disable_exclusive_rename(m)
        _populate(paths)

    assert (paths.paper_task_dir / "task.md").read_text(**UTF8) == "hello"
    assert task_snapshot.task_snapshot_ready_path(paths.paper_task_dir).is_file()
    reuse_paper_task_dir(paths)

    # preserves the generated identity metadata through an ignore pattern
    paths = _source(tmp_path / "identity", {"task.md": "hello", ".agentic_task_ignore": "*.json\n"}, task_name="stored-task")
    prepare_run_task(paths)
    _populate(paths)

    assert read_task_snapshot_name(paths.paper_task_dir) == "stored-task"

    # preserves internal symlinks
    paths = _run_task(tmp_path / "symlinks", {"source.txt": "task data"})
    (paths.task_dir / "linked.txt").symlink_to("source.txt")

    _populate(paths)

    copied_link = paths.paper_task_dir / "linked.txt"
    assert copied_link.is_symlink()
    assert os.readlink(copied_link) == "source.txt"


def test_paper_task_dir_rejects_content_changed_after_publication(tmp_path: Path) -> None:
    # reuse rejects tampered content
    paths = _run_task(tmp_path / "reuse")
    _populate(paths)
    (paths.paper_task_dir / "task.md").write_text("tampered", **UTF8)

    with pytest.raises(UnsafeGluePathError, match="source digest"):
        reuse_paper_task_dir(paths)

    # populate rejects a tampered snapshot when the source is missing
    paths = _run_task(tmp_path / "populate")
    _populate(paths)
    paths.task_dir.rename(paths.task_dir.with_name("removed-task"))
    (paths.paper_task_dir / "task.md").write_text("tampered", **UTF8)

    with pytest.raises(UnsafeGluePathError, match="source digest"):
        _populate(paths)


def _seed_via_prepare(tmp_path: Path) -> PipelinePaths:
    paths = _source(tmp_path, {"task.md": "hello"})
    prepare_run_task(paths)
    return paths


def _seed_via_legacy_populate(tmp_path: Path) -> PipelinePaths:
    paths = _run_task(tmp_path, {"task.md": "hello"})
    _populate(paths)
    return paths


def test_populate_task_dir_excludes_ignored_and_runtime_entries(tmp_path: Path) -> None:
    # honors .agentic_task_ignore from the source task
    paths = _source(tmp_path / "ignore", {"task.md": "hello", ".agentic_task_ignore": ".venv/\nresources/\n"})
    (paths.source_task_dir / ".venv/bin").mkdir(parents=True)
    (paths.source_task_dir / ".venv/bin/python").symlink_to("/usr/bin/python3")
    (paths.source_task_dir / "resources").mkdir()
    (paths.source_task_dir / "resources/data.bin").write_bytes(b"large")

    prepare_run_task(paths)
    _populate(paths)

    assert (paths.paper_task_dir / "task.md").is_file()
    assert not (paths.paper_task_dir / ".venv").exists()
    assert not (paths.paper_task_dir / "resources").exists()

    # excludes runtime entries added to the run task (after snapshot / legacy copy)
    for name, seed in (("after-snapshot", _seed_via_prepare), ("legacy-copy", _seed_via_legacy_populate)):
        paths = seed(tmp_path / name)
        (paths.task_dir / ".venv/bin").mkdir(parents=True)
        (paths.task_dir / ".venv/bin/python").write_text("runtime", **UTF8)
        (paths.task_dir / ".agentic_runtime.json").write_text("{}", **UTF8)

        _populate(paths)

        assert (paths.paper_task_dir / "task.md").is_file(), name
        assert not (paths.paper_task_dir / ".venv").exists(), name
        assert not (paths.paper_task_dir / ".agentic_runtime.json").exists(), name

    # preserves nested entries that merely share runtime names
    paths = _run_task(
        tmp_path / "nested",
        {
            "fixtures/.venv/runtime.txt": "fixture",
            "resources/.agentic_runtime.json": '{"fixture": true}',
            ".agentic_runtime.json": "{}",
        },
    )
    (paths.task_dir / ".venv").mkdir()

    _populate(paths)

    assert (paths.paper_task_dir / "fixtures/.venv/runtime.txt").is_file()
    assert (paths.paper_task_dir / "resources/.agentic_runtime.json").is_file()
    assert not (paths.paper_task_dir / ".venv").exists()
    assert not (paths.paper_task_dir / ".agentic_runtime.json").exists()


def test_paper_task_dir_rejects_managed_runtime_entries(tmp_path: Path) -> None:
    # populate rejects a runtime entry added to an existing destination
    paths = _run_task(tmp_path / "populate", {"task.md": "hello"})
    _populate(paths)
    (paths.paper_task_dir / ".agentic_runtime.json").write_text("{}", **UTF8)

    with pytest.raises(UnsafeGluePathError, match="contains managed runtime entry"):
        _populate(paths)

    # reuse rejects a managed runtime directory
    paths = _paths(tmp_path / "reuse")
    paths.paper_task_dir.mkdir(parents=True)
    (paths.paper_task_dir / "task.md").write_text("hello", **UTF8)
    (paths.paper_task_dir / ".venv").mkdir()

    with pytest.raises(UnsafeGluePathError, match="contains managed runtime entry"):
        reuse_paper_task_dir(paths)


def test_populate_task_dir_existing_destination_rules(tmp_path: Path) -> None:
    # skips when a matching destination is present
    paths = _run_task(tmp_path / "present")
    paths.paper_task_dir.mkdir(parents=True)
    (paths.paper_task_dir / "task.md").write_text("trusted")
    _populate(paths)
    assert (paths.paper_task_dir / "task.md").read_text() == "trusted"

    # rejects a mismatched existing destination and leaves it untouched
    paths = _run_task(tmp_path / "mismatched")
    paths.paper_task_dir.mkdir(parents=True)
    (paths.paper_task_dir / "task.md").write_text("attacker")

    with pytest.raises(UnsafeGluePathError, match="does not match trusted task source"):
        _populate(paths)

    assert (paths.paper_task_dir / "task.md").read_text() == "attacker"

    # reuses a read-only destination without leaving staging trees
    paths = _run_task(tmp_path / "read-only", {"data/data.txt": "trusted"})
    nested_source = paths.task_dir / "data"
    nested_source.chmod(0o555)

    try:
        _populate(paths)
        _populate(paths)
    finally:
        nested_source.chmod(0o755)
        if paths.paper_task_dir.exists():
            (paths.paper_task_dir / "data").chmod(0o755)

    assert not list(paths.ablations_output_dir.glob(".task.*.tmp"))

    # resumes a pre-existing empty destination
    paths = _run_task(tmp_path / "empty")
    paths.paper_task_dir.mkdir(parents=True)

    _populate(paths)

    assert (paths.paper_task_dir / "task.md").read_text(**UTF8) == "trusted"
    assert task_snapshot.task_snapshot_ready_path(paths.paper_task_dir).is_file()


def _external_source_symlink(paths: PipelinePaths, tmp_path: Path) -> None:
    (tmp_path / "external.txt").write_text("host data")
    (paths.task_dir / "linked.txt").symlink_to(tmp_path / "external.txt")


def _absolute_internal_source_symlink(paths: PipelinePaths, _tmp_path: Path) -> None:
    (paths.task_dir / "source.txt").write_text("task data")
    (paths.task_dir / "linked.txt").symlink_to((paths.task_dir / "source.txt").resolve())


def _runtime_symlink_in_destination_without_source(paths: PipelinePaths, _tmp_path: Path) -> None:
    shutil.rmtree(paths.task_dir)
    paths.paper_task_dir.mkdir(parents=True)
    (paths.paper_task_dir / "task.md").write_text("hello", **UTF8)
    (paths.paper_task_dir / ".agentic_runtime.json").symlink_to("/outside/runtime.json")


def _escaping_symlink_in_destination(paths: PipelinePaths, tmp_path: Path) -> None:
    (tmp_path / "outside.txt").write_text("host data")
    paths.paper_task_dir.mkdir(parents=True)
    (paths.paper_task_dir / "task.md").symlink_to(tmp_path / "outside.txt")


def _symlink_destination(paths: PipelinePaths, tmp_path: Path) -> None:
    (tmp_path / "outside").mkdir()
    paths.paper_task_dir.parent.mkdir(parents=True)
    paths.paper_task_dir.symlink_to(tmp_path / "outside")


UNSAFE_PAPER_TASK_CASES = [
    ("external-source-symlink", _external_source_symlink, "task source contains absolute symlink"),
    ("absolute-internal-source-symlink", _absolute_internal_source_symlink, "task source contains absolute symlink"),
    ("runtime-symlink-no-source", _runtime_symlink_in_destination_without_source, "absolute symlink"),
    ("escaping-destination-symlink", _escaping_symlink_in_destination, "task destination contains absolute symlink"),
    ("symlink-destination", _symlink_destination, "task destination must be a real directory"),
]


def test_populate_task_dir_rejects_unsafe_symlinks(tmp_path: Path) -> None:
    for name, layout, match in UNSAFE_PAPER_TASK_CASES:
        root = tmp_path / name
        paths = _run_task(root, {})
        layout(paths, root)

        with pytest.raises(UnsafeGluePathError, match=match):
            _populate(paths)

    # a symlinked destination ancestor is rejected before anything is written
    root = tmp_path / "ancestor"
    paths = _run_task(root)
    redirect, real_output = _redirect(root)

    with pytest.raises(UnsafeGluePathError, match="could not safely create task destination"):
        _populate(replace(paths, paper_task_dir=redirect / "task"))

    assert not list(real_output.iterdir())


def test_populate_task_dir_pins_parent_and_source_across_races(tmp_path: Path, monkeypatch) -> None:
    # publishes under the pinned parent even if the visible parent is swapped
    root = tmp_path / "pinned-parent"
    paths = _run_task(root)
    paths.ablations_output_dir.mkdir()
    pinned_parent = root / "pinned-ablations"
    outside = root / "outside"
    outside.mkdir()
    real_rename = os.rename
    real_publish = safe_fs.rename_directory_between_no_replace
    swapped = False

    def swap_visible_parent_before_publish(*args, role):
        nonlocal swapped
        if not swapped:
            swapped = True
            real_rename(paths.ablations_output_dir, pinned_parent)
            paths.ablations_output_dir.symlink_to(outside, target_is_directory=True)
        return real_publish(*args, role=role)

    with monkeypatch.context() as m:
        m.setattr(
            "src.pipeline.task_publication.rename_directory_between_no_replace",
            swap_visible_parent_before_publish,
        )
        _populate(paths)

    assert (pinned_parent / "task" / "task.md").read_text(**UTF8) == "trusted"
    assert not (outside / "task").exists()

    # copies from the pinned source even if the visible source is swapped
    root = tmp_path / "pinned-source"
    paths = _run_task(root)
    outside = root / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("host secret", **UTF8)
    real_copy_tree_at = task_tree.copy_task_tree_at
    source_paths = paths

    def swap_visible_source_before_copy(*args, **kwargs):
        source_paths.task_dir.rename(source_paths.task_dir.with_name("task-original"))
        source_paths.task_dir.symlink_to(outside, target_is_directory=True)
        return real_copy_tree_at(*args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.glue._copy_task_tree_at", swap_visible_source_before_copy)
        _populate(paths)

    assert (paths.paper_task_dir / "task.md").read_text(**UTF8) == "trusted"
    assert not (paths.paper_task_dir / "secret.txt").exists()

    # rejects a file swapped to a symlink during the copy
    root = tmp_path / "file-swap"
    paths = _run_task(root)
    source_file = paths.task_dir / "task.md"
    secret = root / "secret.txt"
    secret.write_text("host secret", **UTF8)

    def swap() -> None:
        source_file.unlink()
        source_file.symlink_to(secret)

    with monkeypatch.context() as m:
        _swapping_open(m, "src.pipeline.task_tree.os.open", "task.md", swap, flag=os.O_NOFOLLOW)
        _assert_rejected(lambda: _populate(paths), "changed while copying", paths.paper_task_dir)

    # resumes a destination raced into existence (empty) during the copy
    paths = _run_task(tmp_path / "raced-empty")
    raced_paths = paths

    def create_destination_during_copy(*args, **kwargs):
        result = real_copy_tree_at(*args, **kwargs)
        raced_paths.paper_task_dir.mkdir()
        return result

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.glue._copy_task_tree_at", create_destination_during_copy)
        _populate(paths)

    assert (paths.paper_task_dir / "task.md").read_text(**UTF8) == "trusted"
    assert task_snapshot.task_snapshot_ready_path(paths.paper_task_dir).is_file()


# ---------------------------------------------------------------------------
# link_aggregated_csvs
# ---------------------------------------------------------------------------


def test_link_aggregated_csvs_links_relative_warns_when_missing_and_dry_runs(tmp_path: Path, caplog) -> None:
    # links are relative to the task name, not hardcoded
    for task_name in ("circpack", "grpo"):
        paths = _paths(tmp_path / task_name, task_name=task_name)
        agg_task = paths.ablations_aggregated_task_dir
        agg_task.mkdir(parents=True)
        for name in CSV_NAMES:
            (agg_task / name).write_text("x")

        link_aggregated_csvs(paths, dry_run=False)

        for name in CSV_NAMES:
            link = paths.ablations_output_dir / "aggregated" / name
            assert link.is_symlink(), (task_name, name)
            assert os.readlink(link) == f"{task_name}/{name}"
            assert link.resolve() == (agg_task / name).resolve()

    # warns when the aggregated task dir is missing
    paths = _paths(tmp_path / "missing")
    with caplog.at_level("WARNING"):
        link_aggregated_csvs(paths, dry_run=False)
    assert any("aggregated task dir missing" in r.message for r in caplog.records)
    assert not (paths.ablations_output_dir / "aggregated").exists()

    # dry run creates nothing
    paths = _paths(tmp_path / "dry-run")
    paths.ablations_aggregated_task_dir.mkdir(parents=True)
    (paths.ablations_aggregated_task_dir / "run_summary.csv").write_text("x")
    link_aggregated_csvs(paths, dry_run=True)
    assert not (paths.ablations_output_dir / "aggregated" / "run_summary.csv").is_symlink()


def test_link_aggregated_csvs_rejects_symlinked_root_and_source_csv(tmp_path: Path) -> None:
    # a symlinked aggregated root is rejected without touching its target
    root = tmp_path / "root"
    paths = _paths(root)
    outside_root = root / "outside"
    outside_task = outside_root / paths.task_name
    outside_task.mkdir(parents=True)
    for name in CSV_NAMES:
        (outside_task / name).write_text("source", **UTF8)
        (outside_root / name).write_text("sentinel", **UTF8)
    paths.ablations_output_dir.mkdir()
    (paths.ablations_output_dir / "aggregated").symlink_to(outside_root, target_is_directory=True)

    with pytest.raises(UnsafeGluePathError, match="aggregated root"):
        link_aggregated_csvs(paths, dry_run=False)

    for name in CSV_NAMES:
        assert (outside_root / name).read_text(**UTF8) == "sentinel"

    # a symlinked source csv is rejected without touching its target
    root = tmp_path / "csv"
    paths = _paths(root)
    paths.ablations_aggregated_task_dir.mkdir(parents=True)
    outside_file = root / "outside.csv"
    outside_file.write_text("outside", **UTF8)
    (paths.ablations_aggregated_task_dir / "run_summary.csv").symlink_to(outside_file)

    with pytest.raises(UnsafeGluePathError, match="aggregated csv"):
        link_aggregated_csvs(paths, dry_run=False)

    assert outside_file.read_text(**UTF8) == "outside"


# ---------------------------------------------------------------------------
# copy_investigation_memory / run_glue
# ---------------------------------------------------------------------------


def _copy_memory(paths: PipelinePaths) -> None:
    copy_investigation_memory(paths, dry_run=False)


def test_copy_investigation_memory_copies_refreshes_and_reuses(tmp_path: Path) -> None:
    # copies memory.jsonl; an absent memory.md is skipped, not an error
    paths = _memory(tmp_path / "copy", {"memory.jsonl": "j"})
    _copy_memory(paths)
    assert (paths.ablations_output_dir / "memory.jsonl").read_text() == "j"
    assert not (paths.ablations_output_dir / "memory.md").exists()

    # refreshes a changed source
    paths = _memory(tmp_path / "refresh", {"memory.md": "old"})
    _copy_memory(paths)
    (paths.investigation_workspace_dir / "memory.md").write_text("new", **UTF8)

    _copy_memory(paths)

    assert (paths.ablations_output_dir / "memory.md").read_text(**UTF8) == "new"

    # reuses a matching destination (same inode)
    paths = _memory(tmp_path / "reuse", {"memory.md": "same"})
    destination = paths.ablations_output_dir / "memory.md"
    _copy_memory(paths)
    destination_inode = destination.stat().st_ino

    _copy_memory(paths)

    assert destination.stat().st_ino == destination_inode


def test_run_glue_does_not_copy_task_or_memory_and_paper_only_reuses_bundle_task(tmp_path: Path) -> None:
    # run_glue copies neither the task nor the investigation memory
    paths = _run_task(tmp_path / "glue", {"task.md": "t"})
    paths.investigation_workspace_dir.mkdir(parents=True)
    (paths.investigation_workspace_dir / "memory.jsonl").write_text("j")

    run_glue(paths, PipelineConfig(), dry_run=False)

    assert not (paths.paper_task_dir / "task.md").exists()
    assert not (paths.ablations_output_dir / "memory.jsonl").exists()

    # paper-only glue reuses the bundle task without comparing the run snapshot
    paths = _run_task(tmp_path / "paper-only", {"task.md": "run snapshot"})
    paths.paper_task_dir.mkdir(parents=True)
    (paths.paper_task_dir / "task.md").write_text("paper snapshot", **UTF8)
    cfg = PipelineConfig()
    cfg.execution.phases = "paper"

    run_glue(paths, cfg, dry_run=False)

    assert (paths.paper_task_dir / "task.md").read_text(**UTF8) == "paper snapshot"


def _memory_symlink_source(paths: PipelinePaths, tmp_path: Path) -> tuple[PipelinePaths, Path]:
    (tmp_path / "host-secret").write_text("secret")
    (paths.investigation_workspace_dir / "memory.md").symlink_to(tmp_path / "host-secret")
    return paths, paths.ablations_output_dir / "memory.md"


def _memory_symlink_destination(paths: PipelinePaths, tmp_path: Path) -> tuple[PipelinePaths, Path]:
    (paths.investigation_workspace_dir / "memory.md").write_text("agent data")
    (paths.ablations_output_dir / "memory.md").symlink_to(tmp_path / "outside")
    return paths, tmp_path / "outside"


def _memory_symlinked_source_directory(paths: PipelinePaths, tmp_path: Path) -> tuple[PipelinePaths, Path]:
    shutil.rmtree(paths.investigation_workspace_dir)
    real_source = tmp_path / "real-source"
    real_source.mkdir()
    (real_source / "memory.md").write_text("secret")
    paths.investigation_workspace_dir.symlink_to(real_source)
    return paths, paths.ablations_output_dir / "memory.md"


def _memory_symlinked_destination_directory(paths: PipelinePaths, tmp_path: Path) -> tuple[PipelinePaths, Path]:
    (paths.investigation_workspace_dir / "memory.md").write_text("agent data")
    paths.ablations_output_dir.rmdir()
    (tmp_path / "outside").mkdir()
    paths.ablations_output_dir.symlink_to(tmp_path / "outside")
    return paths, tmp_path / "outside" / "memory.md"


def _memory_symlinked_destination_ancestor(paths: PipelinePaths, tmp_path: Path) -> tuple[PipelinePaths, Path]:
    (paths.investigation_workspace_dir / "memory.md").write_text("agent data")
    redirect, real_output = _redirect(tmp_path)
    return replace(paths, ablations_output_dir=redirect), real_output / "memory.md"


UNSAFE_MEMORY_CASES = [
    ("symlink-source", _memory_symlink_source, "source must be a regular file"),
    ("symlink-destination", _memory_symlink_destination, "destination must be a regular file"),
    ("symlinked-source-directory", _memory_symlinked_source_directory, "could not safely open memory source directory"),
    ("symlinked-destination-directory", _memory_symlinked_destination_directory, "could not safely open memory destination"),
    ("symlinked-destination-ancestor", _memory_symlinked_destination_ancestor, "could not safely open memory destination"),
]


def test_copy_investigation_memory_rejects_unsafe_layouts(tmp_path: Path) -> None:
    for name, layout, match in UNSAFE_MEMORY_CASES:
        root = tmp_path / name
        paths, must_not_exist = layout(_memory(root), root)

        with pytest.raises(UnsafeGluePathError, match=match):
            _copy_memory(paths)

        assert not must_not_exist.exists(), name


def test_copy_investigation_memory_rejects_swaps_and_cleans_partial_copies(tmp_path: Path, monkeypatch) -> None:
    # a source swapped to a symlink before the open is rejected
    paths = _memory(tmp_path / "source-swap", {"memory.md": "safe"})
    source = paths.investigation_workspace_dir / "memory.md"
    host_secret = tmp_path / "host-secret"
    host_secret.write_text("secret")
    real_open = os.open
    swapped = False

    def swap_before_source_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "memory.md" and flags & os.O_ACCMODE == os.O_RDONLY and not swapped:
            swapped = True
            source.unlink()
            source.symlink_to(host_secret)
        return real_open(path, flags, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.glue.os.open", swap_before_source_open)
        _assert_rejected(
            lambda: _copy_memory(paths),
            "could not safely open memory source",
            paths.ablations_output_dir / "memory.md",
        )

    # a destination swapped to a symlink before publication is rejected
    paths = _memory(tmp_path / "destination-swap", {"memory.md": "agent data"})
    destination = paths.ablations_output_dir / "memory.md"
    outside_file = tmp_path / "outside"
    real_link = os.link
    link_swapped = False

    def swap_before_destination_publish(source, destination_name, *args, **kwargs):
        nonlocal link_swapped
        if destination_name == "memory.md" and not link_swapped:
            link_swapped = True
            destination.symlink_to(outside_file)
        return real_link(source, destination_name, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.glue.os.link", swap_before_destination_publish)
        _assert_rejected(
            lambda: _copy_memory(paths), "could not safely publish memory destination", outside_file
        )

    # a partial file is removed after a copy failure, and a retry succeeds
    paths = _memory(tmp_path / "partial", {"memory.md": "complete"})

    def fail_after_partial_copy(source, destination) -> None:
        destination.write(source.read(3))
        raise OSError("injected copy failure")

    with monkeypatch.context() as m:
        m.setattr("src.pipeline.safe_fs.shutil.copyfileobj", fail_after_partial_copy)
        with pytest.raises(UnsafeGluePathError, match="could not safely publish memory destination"):
            _copy_memory(paths)

    destination = paths.ablations_output_dir / "memory.md"
    assert not destination.exists()
    assert not list(paths.ablations_output_dir.glob(".memory.md.*.tmp"))

    _copy_memory(paths)
    assert destination.read_text() == "complete"


# ---------------------------------------------------------------------------
# paper workspace: marker validation and container mount plan
# ---------------------------------------------------------------------------


class _Workspace:
    """Paper workspace layout; ``memory`` defaults to ``results``."""

    def __init__(
        self,
        output: Path,
        task: Path,
        results: Path,
        memory: Path | None = None,
        ablations: Path | None = None,
    ) -> None:
        self.output, self.task, self.results = output, task, results
        self.memory = results if memory is None else memory
        self.ablations = ablations

    def mark(self) -> None:
        write_workspace_marker(
            output_dir=self.output, task_dir=self.task, results_dir=self.results, global_memory_dir=self.memory
        )

    def errors(self) -> list[str]:
        kwargs = {} if self.ablations is None else {"ablations_dir": self.ablations}
        return workspace_validation_errors(
            output_dir=self.output,
            task_dir=self.task,
            results_dir=self.results,
            global_memory_dir=self.memory,
            **kwargs,
        )

    def plan(self, **kwargs: Path | None) -> PaperContainerMountPlan:
        return paper_container_mount_plan(
            output_dir=self.output,
            task_dir=self.task,
            results_dir=self.results,
            global_memory_dir=self.memory,
            **kwargs,
        )


def _flat_workspace(tmp_path: Path, *, mkdir_output: bool = True) -> _Workspace:
    """paper/ next to task/ and results/ directly under tmp_path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    if mkdir_output:
        (tmp_path / "paper").mkdir()
    return _Workspace(tmp_path / "paper", tmp_path / "task", tmp_path / "results")


def _run_workspace(
    tmp_path: Path, *, output_name: str = "paper", policy: bool = False, memory_files: bool = False
) -> _Workspace:
    """run/{paper,task,investigation} phase layout, optionally with a resume policy and memory files."""
    run_dir = tmp_path / "run"
    workspace = _Workspace(
        run_dir / output_name, run_dir / "task", run_dir / "investigation", ablations=run_dir / "ablations"
    )
    for directory in (workspace.output, workspace.task, workspace.results):
        directory.mkdir(parents=True)
    if policy:
        (run_dir / ".agentic_resume_policy.json").write_text('{"schema_version":1}\n', **UTF8)
    if memory_files:
        for name in WRITABLE_GLOBAL_MEMORY_FILES:
            (workspace.results / name).touch()
    return workspace


def _nested_workspace(tmp_path: Path) -> _Workspace:
    """Legacy results/paper/final_run layout with memory files in results/."""
    results_dir = tmp_path / "results"
    workspace = _Workspace(results_dir / "paper" / "final_run", results_dir / "task", results_dir)
    for directory in (workspace.output, workspace.task):
        directory.mkdir(parents=True)
    for name in WRITABLE_GLOBAL_MEMORY_FILES:
        (results_dir / name).touch()
    return workspace


UNEXPECTED_SYMLINK_CASES = [
    ("child-file", "paper_data.json", "outside.json"),
    ("wrong-alias-target", "results", "unrelated"),
    ("dangling-merged-results", "results", "paper/_merged_parallel_missing"),
]


def test_workspace_validation_accepts_marker_and_rejects_unexpected_symlinks(tmp_path: Path) -> None:
    # a matching marker validates
    workspace = _flat_workspace(tmp_path / "matching")
    workspace.mark()

    assert workspace.errors() == []

    # a symlinked output directory is rejected
    root = tmp_path / "output-symlink"
    root.mkdir()
    (root / "real-paper").mkdir()
    workspace = _flat_workspace(root, mkdir_output=False)
    workspace.output.symlink_to(root / "real-paper", target_is_directory=True)

    assert workspace.errors() == [f"paper output must not be a symlink: {workspace.output}"]

    # unexpected symlinks inside the workspace are rejected
    for case, name, target in UNEXPECTED_SYMLINK_CASES:
        root = tmp_path / case
        workspace = _flat_workspace(root)
        workspace.mark()
        (workspace.output / name).symlink_to(root / target, target_is_directory=name == "results")

        assert f"unexpected symlink in paper workspace: {name}" in workspace.errors(), case

    # dangling ablation aliases are rejected
    for alias_name in ("aggregated", "extra_tasks"):
        root = tmp_path / f"dangling-{alias_name}"
        workspace = _flat_workspace(root)
        workspace.ablations = root / "ablations"
        workspace.ablations.mkdir()
        workspace.mark()
        (workspace.output / alias_name).symlink_to(
            workspace.ablations / f"missing-{alias_name}", target_is_directory=True
        )

        assert f"unexpected symlink in paper workspace: {alias_name}" in workspace.errors(), alias_name


def test_workspace_validation_allows_pipeline_alias_layouts(tmp_path: Path) -> None:
    # the pipeline alias layout (task/results/aggregated/extra_tasks/global_memory)
    root = tmp_path / "aliases"
    results_dir = root / "results"
    workspace = _Workspace(root / "paper", root / "task-source", results_dir, memory=root / "memory")
    aliases = {
        "task": results_dir / "task",
        "results": results_dir / "paper" / "_merged_parallel_run",
        "aggregated": results_dir / "aggregated",
        "extra_tasks": results_dir / "extra_tasks",
        "global_memory": workspace.memory,
    }
    for directory in (workspace.output, workspace.task, *aliases.values()):
        directory.mkdir(parents=True, exist_ok=True)
    workspace.mark()
    for name, target in aliases.items():
        (workspace.output / name).symlink_to(target, target_is_directory=True)

    assert workspace.errors() == []

    # producer-owned ablation and merged aliases
    workspace = _run_workspace(tmp_path / "producer-owned")
    assert workspace.ablations is not None
    merged_results = workspace.output / "_merged_parallel_20260826_120000"
    extra_tasks = workspace.ablations / "extra_tasks"
    for directory in (merged_results, extra_tasks):
        directory.mkdir(parents=True)
    workspace.mark()
    (workspace.output / "results").symlink_to(merged_results, target_is_directory=True)
    (workspace.output / "extra_tasks").symlink_to(extra_tasks, target_is_directory=True)

    assert workspace.errors() == []

    # merged parallel idea links into results are allowed; strays are not
    root = tmp_path / "merged-ideas"
    workspace = _run_workspace(root)
    workspace.ablations = None
    worker_idea = workspace.results / "worker_01" / "ideas" / "01_idea"
    merged_ideas = workspace.output / "_merged_parallel_20261011_101125" / "ideas"
    for directory in (worker_idea, merged_ideas):
        directory.mkdir(parents=True)
    workspace.mark()
    (merged_ideas / "worker_01__01_idea").symlink_to(worker_idea, target_is_directory=True)
    (merged_ideas / "stray").symlink_to(root, target_is_directory=True)

    assert workspace.errors() == [
        "unexpected symlink in paper workspace: _merged_parallel_20261011_101125/ideas/stray"
    ]

    # a sibling ablations alias is allowed
    workspace = _run_workspace(tmp_path / "sibling", policy=True)
    assert workspace.ablations is not None
    (workspace.ablations / "aggregated").mkdir(parents=True)
    workspace.mark()
    (workspace.output / "aggregated").symlink_to(workspace.ablations / "aggregated", target_is_directory=True)

    assert workspace.errors() == []


def test_container_mount_plan_layouts(tmp_path: Path) -> None:
    # a read-only root with narrow write overlays
    root = tmp_path / "phase"
    workspace = _run_workspace(root, policy=True, memory_files=True)
    assert workspace.ablations is not None
    (workspace.ablations / "aggregated").mkdir(parents=True)

    plan = workspace.plan(aggregated_dir=workspace.ablations / "aggregated")

    assert plan.mount_root == (root / "run").resolve()
    assert plan.readonly_mounts == ()
    assert plan.writable_mounts == (
        f"{workspace.output.resolve()}:/workspace/paper:rw",
        f"{(workspace.results / 'memory.md').resolve()}:/workspace/investigation/memory.md:rw",
        f"{(workspace.results / 'memory.jsonl').resolve()}:/workspace/investigation/memory.jsonl:rw",
        f"{(workspace.results / 'conclusions_history.jsonl').resolve()}:/workspace/investigation/conclusions_history.jsonl:rw",
    )

    # the legacy nested output layout is supported
    workspace = _nested_workspace(tmp_path / "nested")

    plan = workspace.plan()

    assert plan.mount_root == workspace.results.resolve()
    assert plan.readonly_mounts == ()
    assert plan.writable_mounts[0] == f"{workspace.output.resolve()}:/workspace/paper/final_run:rw"

    # a custom output name is allowed
    root = tmp_path / "custom"
    workspace = _run_workspace(root, output_name="custom-paper", policy=True, memory_files=True)

    plan = workspace.plan()

    assert plan.mount_root == (root / "run").resolve()
    assert plan.writable_mounts[0].endswith(":/workspace/custom-paper:rw")

    # external ablations are bound read-only
    root = tmp_path / "external"
    workspace = _run_workspace(root, policy=True, memory_files=True)
    aggregated_dir = root / "external" / "aggregated"
    aggregated_dir.mkdir(parents=True)

    plan = workspace.plan(aggregated_dir=aggregated_dir)

    assert plan.readonly_mounts == (f"{aggregated_dir.resolve()}:/external/aggregated:ro",)


def test_container_mount_plan_rejects_unbounded_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # an unbounded layout (task outside the run) is rejected
    root = tmp_path / "unbounded"
    workspace = _run_workspace(root, memory_files=True)
    workspace.task = root / "external-task"
    workspace.task.mkdir()

    with pytest.raises(ValueError, match="bounded container root"):
        workspace.plan()

    # a phase layout without an owned marker is rejected
    workspace = _run_workspace(tmp_path / "no-marker", memory_files=True)

    with pytest.raises(ValueError, match="bounded container root"):
        workspace.plan()

    # the home directory is rejected as a mount root
    home = tmp_path / "home"
    home.mkdir()
    workspace = _Workspace(home / "paper", home / "task", home / "investigation")
    for directory in (workspace.output, workspace.task, workspace.results):
        directory.mkdir()
    (home / ".agentic_resume_policy.json").write_text('{"schema_version":1}\n', **UTF8)
    for name in WRITABLE_GLOBAL_MEMORY_FILES:
        (workspace.results / name).touch()

    with monkeypatch.context() as m:
        m.setattr(Path, "home", classmethod(lambda cls: home))
        with pytest.raises(ValueError, match="broad paper container mount root"):
            workspace.plan()


def test_writable_memory_files_must_be_unlinked_regular_files(tmp_path: Path) -> None:
    # a hard-linked memory file is rejected by the mount plan
    workspace = _nested_workspace(tmp_path / "hardlink")
    os.link(workspace.results / "memory.md", workspace.results / "memory-copy.md")

    with pytest.raises(ValueError, match="Unsafe writable paper file"):
        workspace.plan()

    # a FIFO is rejected before it is opened
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    os.mkfifo(memory_dir / "memory.md")

    with pytest.raises(ValueError, match="Unsafe writable paper file"):
        validate_existing_global_memory_files(memory_dir)
