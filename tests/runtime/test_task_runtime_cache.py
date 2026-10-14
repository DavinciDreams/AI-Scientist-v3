"""Task runtime install cache: publication, eviction, recovery and deadline handling."""
from __future__ import annotations

import errno
import fcntl
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.investigation import task_runtime_cache as cache_module
from src.investigation.task_runtime_cache import (
    TaskRuntimeInstallCache,
    TaskRuntimeInstallCacheDeadlineExceeded,
    TaskRuntimeInstallCacheKey,
    discard_task_runtime_install_cache,
    prepare_task_runtime_install_cache,
    task_runtime_install_cache_root,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _cache_root(tmp_path: Path) -> Path:
    root = tmp_path / "cache"
    root.mkdir(mode=0o700, parents=True)
    return root


def _runtime(tmp_path: Path, name: str, *, size: int | None = None) -> Path:
    runtime = tmp_path / name / ".venv"
    python = runtime / "bin" / "python"
    python.parent.mkdir(parents=True)
    if size is None:
        python.write_text("python", encoding="utf-8")
    else:
        python.write_bytes(b"x" * size)
    return runtime


def _key(index: int) -> TaskRuntimeInstallCacheKey:
    return TaskRuntimeInstallCacheKey(
        image_id=f"sha256:{index:064x}",
        requirements_sha256=f"{index + 10:064x}",
        tooling_sha256=f"{index + 20:064x}",
    )


@pytest.fixture
def cache(tmp_path: Path) -> TaskRuntimeInstallCache:
    return TaskRuntimeInstallCache(_cache_root(tmp_path))


def _fresh_cache(tmp_path: Path, name: str, **kwargs: Any) -> TaskRuntimeInstallCache:
    """An independent cache under ``tmp_path / name`` for a sequential test section."""
    return TaskRuntimeInstallCache(_cache_root(tmp_path / name), **kwargs)


def _owned_source(cache: TaskRuntimeInstallCache) -> tuple[Path, Path]:
    """A cache-local build root and the runtime it owns."""
    build_root = cache.create_build_root()
    return build_root, _runtime(cache.root, build_root.name)


def _fail_eviction(cache: TaskRuntimeInstallCache, monkeypatch, *, expected_exclude=None) -> None:
    def fail(*, exclude: Path, deadline: float | None = None) -> None:
        if expected_exclude is not None:
            assert exclude == expected_exclude
            assert deadline is None
        raise OSError("injected eviction failure")

    monkeypatch.setattr(cache, "_evict", fail)


def _fail_replace(monkeypatch, predicate, error: OSError) -> None:
    """Make ``Path.replace`` raise ``error`` when ``predicate(path, target)`` holds."""
    original_replace = Path.replace

    def replace(path: Path, target: Path) -> Path:
        if predicate(path, target):
            raise error
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace)


def _cross_device(path: Path, target: Path) -> OSError:
    return OSError(errno.EXDEV, "Invalid cross-device link", str(path), str(target))


def _root_entries(cache: TaskRuntimeInstallCache) -> set[str]:
    return {path.name for path in cache.root.iterdir()}


def test_install_cache_stores_evicts_materializes_and_creates_private_build_roots(cache, tmp_path):
    # retains only the two most recent entries
    first, second, third = _key(1), _key(2), _key(3)
    cache.store(first, _runtime(tmp_path, "first"))
    cache.store(second, _runtime(tmp_path, "second"))
    assert cache.lookup(first) is not None
    cache.store(third, _runtime(tmp_path, "third"))
    assert cache.lookup(first) is not None
    assert cache.lookup(second) is None
    assert cache.lookup(third) is not None

    # materializes an isolated runtime copy
    destination = tmp_path / "task" / ".venv"
    destination.parent.mkdir()
    cache.materialize(first, destination)
    (destination / "bin" / "python").write_text("poisoned", encoding="utf-8")
    clean_destination = tmp_path / "next-task" / ".venv"
    clean_destination.parent.mkdir()
    cache.materialize(first, clean_destination)
    assert (clean_destination / "bin" / "python").read_text(encoding="utf-8") == "python"
    assert cache.lookup(first) is not None

    # creates private cache-local build roots
    build_a = cache.create_build_root()
    build_b = cache.create_build_root()
    assert build_a.parent == cache.root
    assert build_b.parent == cache.root
    assert build_a != build_b
    assert build_a.name.startswith(".build-")
    assert build_a.stat().st_mode & 0o077 == 0

    # store_owned consumes the cache-local runtime
    _, source = _owned_source(cache)
    entry = cache.store_owned(_key(4), source)
    assert not source.exists()
    assert (entry / ".venv/bin/python").read_text(encoding="utf-8") == "python"


def test_install_cache_rejects_public_root_symlink_escape_foreign_build_root_and_expired_deadline(
    cache, tmp_path
):
    # rejects a non-private root
    root = _cache_root(tmp_path / "public")
    root.chmod(0o755)
    with pytest.raises(ValueError, match="must be private"):
        TaskRuntimeInstallCache(root)
    # rejects a runtime symlink escape
    runtime = _runtime(tmp_path, "unsafe")
    (runtime / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="absolute symlink"):
        cache.store(_key(1), runtime)
    assert runtime.is_dir()
    # build root respects an expired deadline
    with pytest.raises(TaskRuntimeInstallCacheDeadlineExceeded):
        cache.create_build_root(deadline=0.0)
    # store_owned rejects a runtime outside a build root
    source = _runtime(tmp_path, "source")
    with pytest.raises(ValueError, match="cache-owned build root"):
        cache.store_owned(_key(1), source)
    assert source.is_dir()


def test_install_cache_store_owned_failures_restore_source(tmp_path, monkeypatch):
    key = _key(1)
    # eviction failure restores the source
    with monkeypatch.context() as m:
        cache = _fresh_cache(tmp_path, "eviction")
        build_root, source = _owned_source(cache)
        _fail_eviction(cache, m, expected_exclude=cache.root / key.digest)
        with pytest.raises(OSError, match="injected eviction failure"):
            cache.store_owned(key, source)
        assert (source / "bin/python").read_text(encoding="utf-8") == "python"
        assert not (cache.root / key.digest).exists()
        assert _root_entries(cache) == {".lock", build_root.name}
    # move failure preserves the source
    with monkeypatch.context() as m:
        cache = _fresh_cache(tmp_path, "move")
        build_root, source = _owned_source(cache)
        _fail_replace(m, lambda path, _target: path == source, _cross_device(source, source))
        with pytest.raises(OSError, match="Invalid cross-device link"):
            cache.store_owned(key, source)
        assert (source / "bin/python").is_file()
        assert _root_entries(cache) == {".lock", build_root.name}
    # entry is tracked before the post-publish sync
    with monkeypatch.context() as m:
        cache = _fresh_cache(tmp_path, "sync")
        _, source = _owned_source(cache)
        entry = cache.root / key.digest
        original_sync = cache._sync_directory

        def fail_post_publish_sync(path: Path) -> None:
            if path == cache.root and entry.is_dir():
                raise OSError("injected post-publish sync failure")
            original_sync(path)

        m.setattr(cache, "_sync_directory", fail_post_publish_sync)
        with pytest.raises(OSError, match="post-publish sync failure"):
            cache.store_owned(key, source)
        assert (source / "bin/python").is_file()
        assert not entry.exists()


def test_install_cache_store_owned_retains_recovery_state_when_rollback_fails(tmp_path, monkeypatch):
    key = _key(1)
    # rollback failure retains only the runtime
    with monkeypatch.context() as m:
        cache = _fresh_cache(tmp_path, "rollback")
        _, source = _owned_source(cache)
        _fail_eviction(cache, m)
        _fail_replace(m, lambda _path, target: target == source, OSError("injected restore failure"))
        with pytest.raises(OSError, match="recovery state retained"):
            cache.store_owned(key, source)
        recovery_roots = list(cache.root.glob(".recovery-*"))
        assert not source.exists()
        assert not (cache.root / key.digest).exists()
        assert len(recovery_roots) == 1
        assert (recovery_roots[0] / ".venv/bin/python").is_file()
    # failed recovery rename reports the actual runtime location
    with monkeypatch.context() as m:
        cache = _fresh_cache(tmp_path, "rename")
        _, source = _owned_source(cache)
        entry = cache.root / key.digest
        _fail_eviction(cache, m)
        _fail_replace(
            m,
            lambda path, target: path == entry and target.name.startswith(".recovery-"),
            OSError("injected recovery rename failure"),
        )
        with pytest.raises(OSError, match="recovery state retained") as error:
            cache.store_owned(key, source)
        assert f"recovery state retained at {entry}:" in str(error.value)
        assert (entry / ".venv/bin/python").is_file()
        assert not source.exists()
        assert not list(cache.root.glob(".recovery-*"))
    # compound recovery failure keeps the marker, and a later lookup completes recovery
    cache = _fresh_cache(tmp_path, "compound")
    _, source = _owned_source(cache)
    entry = cache.root / key.digest
    with monkeypatch.context() as m:
        original_sync = cache._sync_directory

        def fail_entry_sync(path: Path) -> None:
            if path == entry:
                raise OSError("injected entry sync failure")
            original_sync(path)

        m.setattr(cache, "_sync_directory", fail_entry_sync)
        _fail_replace(m, lambda _path, target: target == source, OSError("injected restore failure"))
        with pytest.raises(OSError, match="recovery state retained"):
            cache.store_owned(key, source)
        recovery_root = next(cache.root.glob(".recovery-*"))
        assert (recovery_root / ".owned-runtime-recovery.json").is_file()
    assert cache.lookup(key) is None
    assert (source / "bin/python").is_file()
    assert not recovery_root.exists()
    # recovery rejects a symlink at the restored destination
    cache = _fresh_cache(tmp_path, "symlink")
    build_root = cache.create_build_root()
    (build_root / ".venv").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    recovery_root = cache.root / f".{key.digest}.staging-{'a' * 32}"
    recovery_root.mkdir(mode=0o700)
    cache._write_owned_recovery_record(recovery_root, key=key, source_build_root=build_root.name)
    with pytest.raises(OSError, match="recovery is incomplete"):
        cache._restore_owned_runtime(recovery_root, source_build_root=build_root.name, deadline=None)
    assert (recovery_root / ".owned-runtime-recovery.json").is_file()


def test_install_cache_store_owned_recovers_subprocess_interrupted_move(cache):
    key = _key(1)
    _, source = _owned_source(cache)
    script = f"""
import os
from pathlib import Path
from src.investigation.task_runtime_cache import TaskRuntimeInstallCache, TaskRuntimeInstallCacheKey

cache = TaskRuntimeInstallCache(Path({str(cache.root)!r}))
key = TaskRuntimeInstallCacheKey(
    image_id={key.image_id!r},
    requirements_sha256={key.requirements_sha256!r},
    tooling_sha256={key.tooling_sha256!r},
)
source = Path({str(source)!r})
original_replace = Path.replace

def interrupt_after_owned_move(path, target):
    result = original_replace(path, target)
    if path == source:
        os._exit(23)
    return result

Path.replace = interrupt_after_owned_move
cache.store_owned(key, source)
"""

    interrupted = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, env=os.environ.copy(), check=False)

    assert interrupted.returncode == 23
    restarted_cache = TaskRuntimeInstallCache(cache.root)
    assert restarted_cache.lookup(key) is None
    assert (source / "bin/python").read_text(encoding="utf-8") == "python"
    assert not list(cache.root.glob(".*.staging-*"))


def test_install_cache_store_failures_preserve_source_and_roll_back_staging(tmp_path, monkeypatch):
    key = _key(1)
    # does not rename the source across filesystems
    with monkeypatch.context() as m:
        cache = _fresh_cache(tmp_path, "xdev")
        source = _runtime(tmp_path / "xdev", "source")
        _fail_replace(m, lambda path, _target: path == source, _cross_device(source, source))
        entry = cache.store(key, source)
        assert source.is_dir()
        assert (entry / ".venv/bin/python").read_text(encoding="utf-8") == "python"
    # copy failure preserves the source and removes staging
    with monkeypatch.context() as m:
        cache = _fresh_cache(tmp_path, "copy")
        source = _runtime(tmp_path / "copy", "source")

        def fail_partial_copy(_source: str, destination: str, **_kwargs: object) -> str:
            Path(destination).write_text("partial", encoding="utf-8")
            raise OSError("injected cache copy failure")

        m.setattr(TaskRuntimeInstallCache, "_copy_file_before_deadline", fail_partial_copy)
        with pytest.raises(OSError, match="injected cache copy failure"):
            cache.store(key, source)
        assert source.is_dir()
        assert _root_entries(cache) == {".lock"}
    # post-publish failure rolls back the cached copy
    with monkeypatch.context() as m:
        cache = _fresh_cache(tmp_path, "publish")
        source = _runtime(tmp_path / "publish", "source")

        def fail_eviction(*, exclude: Path, deadline: float | None = None) -> None:
            assert exclude.parent == cache.root
            assert deadline is None
            raise OSError("injected eviction failure")

        m.setattr(cache, "_evict", fail_eviction)
        with pytest.raises(OSError, match="injected eviction failure"):
            cache.store(key, source)
        assert source.is_dir()
        assert _root_entries(cache) == {".lock"}
    # interrupted rollback never leaves a visible partial entry
    with monkeypatch.context() as m:
        cache = _fresh_cache(tmp_path, "interrupted")
        source = _runtime(tmp_path / "interrupted", "source")
        (source / "lib").mkdir()
        (source / "lib/payload").write_text("payload", encoding="utf-8")
        real_rmtree = cache_module.shutil.rmtree
        _fail_eviction(cache, m)

        def interrupt_rollback(path: Path, ignore_errors: bool = False) -> None:
            target = Path(path)
            if target == cache.root / key.digest or target.name.startswith(".evict-"):
                next(item for item in target.rglob("*") if item.is_file()).unlink()
                raise OSError("injected rollback interruption")
            real_rmtree(target, ignore_errors=ignore_errors)

        m.setattr(cache_module.shutil, "rmtree", interrupt_rollback)
        with pytest.raises(OSError, match="injected eviction failure"):
            cache.store(key, source)
        assert not (cache.root / key.digest).exists()
        assert list(cache.root.glob(".evict-*"))


def test_install_cache_deadlines_interrupt_lookup_locking_validation_and_copy(tmp_path, monkeypatch):
    key = _key(1)
    # lookup deadline interrupts cached tree validation
    with monkeypatch.context() as m:
        cache = _fresh_cache(tmp_path, "lookup")
        cache.store(key, _runtime(tmp_path / "lookup", "source"))
        clock = iter((0.0, 0.0, 2.0))
        m.setattr(cache_module.time, "monotonic", lambda: next(clock, 2.0))
        with pytest.raises(TaskRuntimeInstallCacheDeadlineExceeded, match="deadline"):
            cache.lookup(key, deadline=1.0)
    # deadline bounds lock contention
    with monkeypatch.context() as m:
        cache = _fresh_cache(tmp_path, "lock")
        source = _runtime(tmp_path / "lock", "source")
        clock = iter((0.0, 2.0))

        def contended_lock(_descriptor: int, operation: int) -> None:
            assert operation & fcntl.LOCK_NB
            raise BlockingIOError

        m.setattr(cache_module.time, "monotonic", lambda: next(clock, 2.0))
        m.setattr(cache_module.fcntl, "flock", contended_lock)
        with pytest.raises(TaskRuntimeInstallCacheDeadlineExceeded, match="deadline"):
            cache.store(key, source, deadline=1.0)
        assert source.is_dir()
    # deadline interrupts runtime tree validation
    with monkeypatch.context() as m:
        runtime = _runtime(tmp_path / "validate", "source")
        clock = iter((0.0, 0.0, 2.0))
        m.setattr(cache_module.time, "monotonic", lambda: next(clock, 2.0))
        with pytest.raises(TaskRuntimeInstallCacheDeadlineExceeded, match="deadline"):
            TaskRuntimeInstallCache._validate_runtime_tree(runtime, deadline=1.0)
    # copy deadline preserves the source and removes staging
    with monkeypatch.context() as m:
        cache = _fresh_cache(tmp_path, "copy")
        source = _runtime(tmp_path / "copy", "source", size=2 * 1024 * 1024)

        def monotonic() -> float:
            staging_files = list(cache.root.glob(".*.staging-*/.venv/bin/python"))
            return 2.0 if staging_files and staging_files[0].stat().st_size > 0 else 0.0

        m.setattr(cache_module.time, "monotonic", monotonic)
        with pytest.raises(TaskRuntimeInstallCacheDeadlineExceeded, match="deadline"):
            cache.store(key, source, deadline=1.0)
        assert source.is_dir()
        assert _root_entries(cache) == {".lock"}


def test_install_cache_eviction_and_materialize_honor_deadlines_and_metadata_failures(tmp_path, monkeypatch):
    first, second = _key(1), _key(2)
    # eviction honors the publication deadline
    with monkeypatch.context() as m:
        cache = _fresh_cache(tmp_path, "evict", max_entries=1)
        cache.store(first, _runtime(tmp_path / "evict", "first"))
        source = _runtime(tmp_path / "evict", "second")

        def monotonic() -> float:
            published = [path for path in cache.root.iterdir() if not path.name.startswith(".")]
            return 2.0 if len(published) > 1 else 0.0

        m.setattr(cache_module.time, "monotonic", monotonic)
        with pytest.raises(TaskRuntimeInstallCacheDeadlineExceeded, match="deadline"):
            cache.store(second, source, deadline=1.0)
        assert source.is_dir()
        assert cache.lookup(first) is not None
        assert cache.lookup(second) is None
    # interrupted eviction hides and reaps the tombstone
    with monkeypatch.context() as m:
        cache = _fresh_cache(tmp_path, "tombstone", max_entries=1)
        first_source = _runtime(tmp_path / "tombstone", "first")
        (first_source / "lib/data").mkdir(parents=True)
        (first_source / "lib/data/payload").write_text("payload", encoding="utf-8")
        first_entry = cache.store(first, first_source)
        original_paths = {path.relative_to(first_entry) for path in first_entry.rglob("*")}
        second_source = _runtime(tmp_path / "tombstone", "second")

        def tombstone_monotonic() -> float:
            tombstones = list(cache.root.glob(".evict-*"))
            if not tombstones:
                return 0.0
            current_paths = {path.relative_to(tombstones[0]) for path in tombstones[0].rglob("*")}
            return 2.0 if current_paths != original_paths else 0.0

        m.setattr(cache_module.time, "monotonic", tombstone_monotonic)
        with pytest.raises(TaskRuntimeInstallCacheDeadlineExceeded, match="deadline"):
            cache.store(second, second_source, deadline=1.0)
        assert cache.lookup(first) is None
        assert cache.lookup(second) is None
        assert list(cache.root.glob(".evict-*"))
        m.setattr(cache_module.time, "monotonic", lambda: 0.0)
        assert cache.store(second, second_source, deadline=1.0).is_dir()
        assert not list(cache.root.glob(".evict-*"))
    # materialize metadata failure does not publish the destination
    with monkeypatch.context() as m:
        cache = _fresh_cache(tmp_path, "metadata")
        cache.store(first, _runtime(tmp_path / "metadata", "source"))
        destination = tmp_path / "metadata" / "task" / ".venv"
        destination.parent.mkdir()
        real_utime = cache_module.os.utime

        def fail_metadata_update(
            path: Any,
            times: tuple[float, float] | None = None,
            *,
            ns: tuple[int, int] | None = None,
            dir_fd: int | None = None,
            follow_symlinks: bool = True,
        ) -> None:
            if Path(path) == cache.root / first.digest:
                raise OSError("metadata update failed")
            if ns is None:
                real_utime(path, times, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
            else:
                real_utime(path, ns=ns, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

        m.setattr(cache_module.os, "utime", fail_metadata_update)
        with pytest.raises(OSError, match="metadata update failed"):
            cache.materialize(first, destination)
        assert not destination.exists()
        assert (cache.root / first.digest / ".venv/bin/python").is_file()
    # materialize deadline removes partial destination staging
    with monkeypatch.context() as m:
        cache = _fresh_cache(tmp_path, "materialize")
        cache.store(first, _runtime(tmp_path / "materialize", "source", size=2 * 1024 * 1024))
        destination = tmp_path / "materialize" / "task/.venv"
        destination.parent.mkdir()

        def staging_monotonic() -> float:
            staging_files = list(destination.parent.glob("..venv.materialize-*/bin/python"))
            return 2.0 if staging_files and staging_files[0].stat().st_size > 0 else 0.0

        m.setattr(cache_module.time, "monotonic", staging_monotonic)
        with pytest.raises(TaskRuntimeInstallCacheDeadlineExceeded, match="deadline"):
            cache.materialize(first, destination, deadline=1.0)
        assert not destination.exists()
        assert not list(destination.parent.glob("..venv.materialize-*"))


def test_cross_phase_cache_root_is_deterministic_private_scoped_and_rejects_unsafe_stale_root(
    tmp_path: Path,
) -> None:
    # deterministic, private and scoped
    task_dir = tmp_path / "run" / "task"
    task_dir.mkdir(parents=True)
    root = prepare_task_runtime_install_cache(task_dir, reset=True)
    try:
        assert root == task_runtime_install_cache_root(task_dir)
        assert root.parent == Path("/tmp")
        assert root.stat().st_mode & 0o077 == 0
        assert prepare_task_runtime_install_cache(task_dir, reset=False) == root
    finally:
        discard_task_runtime_install_cache(task_dir)
    assert not root.exists()
    # rejects an unsafe stale root
    stale_task_dir = tmp_path / "stale" / "task"
    stale_task_dir.mkdir(parents=True)
    stale_root = task_runtime_install_cache_root(stale_task_dir)
    stale_root.mkdir(mode=0o755)
    try:
        with pytest.raises(ValueError, match="must be private"):
            prepare_task_runtime_install_cache(stale_task_dir, reset=True)
    finally:
        stale_root.chmod(0o700)
        discard_task_runtime_install_cache(stale_task_dir)
