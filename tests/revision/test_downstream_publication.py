"""Frozen selections stay unpublished until a portable copy is verified."""

import errno
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from src.pipeline import downstream_publication, downstream_selection, safe_fs
from tests.revision.test_downstream_selection import candidate, search


@pytest.fixture
def unsupported_rename(monkeypatch):
    monkeypatch.setattr(safe_fs, "_renameat2_no_replace", lambda *a, **k: False)
    monkeypatch.setattr(safe_fs, "_renameatx_np_no_replace", lambda *a, **k: False)


def test_publishes_verified_selection_and_never_adopts_existing_destination(tmp_path, unsupported_rename):
    # publishes the verified selection without an exclusive rename
    base = tmp_path / "publish"
    root = search(base)
    source = candidate(root)
    expected = downstream_selection.preview_downstream_selection(root)
    selected = downstream_selection.prepare_downstream_selection(root)
    assert selected == expected
    frozen = Path(selected["program_dir"])
    assert (frozen / "src/program.py").read_bytes() == (source / "src/program.py").read_bytes()
    assert set(p.name for p in frozen.iterdir()) == {"src", "program_spec.toml"}
    assert downstream_selection.prepare_downstream_selection(root) == selected

    # an existing destination is never adopted, whatever its kind
    for kind in ("empty", "nonempty", "file", "symlink"):
        base = tmp_path / kind
        root = search(base)
        candidate(root)
        destination = base / "downstream/program"
        destination.parent.mkdir()
        outside = base / "outside"
        outside.mkdir()
        (outside / "keep").write_text("untouched")
        if kind in {"empty", "nonempty"}:
            destination.mkdir()
            if kind == "nonempty":
                (destination / "keep").write_text("untouched")
        elif kind == "file":
            destination.write_text("untouched")
        else:
            destination.symlink_to(outside, target_is_directory=True)
        before = destination.lstat()
        with pytest.raises(ValueError, match="freeze downstream"):
            downstream_selection.prepare_downstream_selection(root)
        assert destination.lstat() == before, kind
        assert (outside / "keep").read_text() == "untouched", kind
        assert not (base / ".downstream_selection.json").exists(), kind


def test_interrupted_or_raced_fallback_copy_is_never_committed(tmp_path, unsupported_rename, monkeypatch):
    # a partial copy is invisible and a retry preserves it
    base = tmp_path / "partial"
    root = search(base)
    candidate(root)

    def interrupt_copy(source_fd, destination_fd, **kwargs):
        assert downstream_selection.resolve_downstream_program(root) is None
        fd = os.open("partial", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
                     dir_fd=destination_fd)
        os.close(fd)
        raise OSError(errno.ENOSPC, "injected full filesystem")

    with monkeypatch.context() as context:
        context.setattr(downstream_publication, "copy_task_tree_at", interrupt_copy)
        with pytest.raises(ValueError, match="freeze downstream"):
            downstream_selection.prepare_downstream_selection(root)
        assert downstream_selection.resolve_downstream_program(root) is None
        partial = base / "downstream/program/partial"
        before = partial.stat()
        with pytest.raises(ValueError, match="freeze downstream"):
            downstream_selection.prepare_downstream_selection(root)
        assert partial.stat() == before

    original = downstream_publication.copy_task_tree_at

    # a destination replaced during the copy is not published
    for replacement in ("directory", "symlink"):
        base = tmp_path / f"replaced_{replacement}"
        root = search(base)
        candidate(root)
        destination = base / "downstream/program"

        def replace_destination(*args, destination=destination, replacement=replacement, **kwargs):
            original(*args, **kwargs)
            destination.rename(destination.with_name("displaced"))
            if replacement == "directory":
                destination.mkdir()
            else:
                destination.symlink_to(destination.with_name("displaced"))

        with monkeypatch.context() as context:
            context.setattr(downstream_publication, "copy_task_tree_at", replace_destination)
            with pytest.raises(ValueError, match="changed|unsafe"):
                downstream_selection.prepare_downstream_selection(root)
        assert not (base / ".downstream_selection.json").exists(), replacement

    # evidence changed during the fallback copy is not committed
    for change in ("source", "history", "baseline"):
        base = tmp_path / f"changed_{change}"
        root = search(base)
        source = candidate(root)
        paths = {"source": source / "src/program.py", "history": root / "attempts_history.jsonl",
                 "baseline": root / ".baseline/program_0/src/program.py"}

        def change_evidence(*args, path=paths[change], **kwargs):
            original(*args, **kwargs)
            with path.open("a") as stream:
                stream.write(" ")

        with monkeypatch.context() as context:
            context.setattr(downstream_publication, "copy_task_tree_at", change_evidence)
            with pytest.raises(ValueError, match="changed|mismatch"):
                downstream_selection.prepare_downstream_selection(root)
        assert not (base / ".downstream_selection.json").exists(), change

    # a manifest appearing at commit time is not replaced
    base = tmp_path / "manifest"
    root = search(base)
    candidate(root)
    original_link = downstream_publication.os.link
    manifest = base / ".downstream_selection.json"

    def competing_manifest(*args, **kwargs):
        manifest.write_text("independent publication")
        return original_link(*args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(downstream_publication.os, "link", competing_manifest)
        with pytest.raises(ValueError, match="freeze downstream"):
            downstream_selection.prepare_downstream_selection(root)
    assert manifest.read_text() == "independent publication"


def test_competing_publisher_cannot_replace_in_progress_copy(tmp_path, unsupported_rename,
                                                          monkeypatch):
    root = search(tmp_path)
    candidate(root)
    entered, release = Event(), Event()
    original = downstream_publication.copy_task_tree_at

    def pause_copy(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(downstream_publication, "copy_task_tree_at", pause_copy)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(downstream_selection.prepare_downstream_selection, root)
        try:
            assert entered.wait(10)
            assert downstream_selection.resolve_downstream_program(root) is None
            with pytest.raises(ValueError, match="freeze downstream"):
                downstream_selection.prepare_downstream_selection(root)
        finally:
            release.set()
        selected = first.result(timeout=10)
    assert downstream_selection.load_downstream_selection(root) == selected


def test_unrelated_rename_failure_does_not_fall_back(tmp_path, monkeypatch):
    root = search(tmp_path)
    candidate(root)

    def fail(*args, **kwargs):
        raise safe_fs.UnsafeGluePathError("injected I/O failure")

    monkeypatch.setattr(safe_fs, "_renameat2_no_replace", fail)
    with pytest.raises(ValueError, match="injected I/O failure"):
        downstream_selection.prepare_downstream_selection(root)
    assert not (tmp_path / "downstream/program").exists()
    assert not (tmp_path / ".downstream_selection.json").exists()
