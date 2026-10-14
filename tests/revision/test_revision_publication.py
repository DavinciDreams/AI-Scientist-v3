"""Revision publication: promotion, ancillary archive, audit-bound copies, portable exchange, cleanup."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from src.orchestrator.paper_evidence_audit import audit_input_sha256
from src.pipeline import (
    portable_publication,
    publication_cleanup,
    publication_tree,
    revision_ancillary_archive,
    revision_audit_publication,
    revision_publication,
)
from src.pipeline.paths import derive_paths
from src.pipeline.publication_journal import DirectoryIdentity, PublicationRecord
from src.pipeline.revision import promote_revision_workspace, run_pipeline_revision
from src.pipeline.revision_ancillary_archive import archive_original_only_inputs
from src.pipeline.revision_audit_publication import copy_audit_bound_inputs
from src.pipeline.revision_evidence import validate_published_revision_evidence
from src.pipeline.revision_publication import recover_incomplete_promotion
from src.investigation.final_ai_review import FinalAiReviewResult
from src.task_drafting.publication_fs import TaskPublicationDeadlineExceeded, regular_tree_files
from tests.revision_test_support import (
    refresh_evidence_audit,
    revision_config,
    review_payload,
    write_workspace,
)

JOURNAL = ".paper.revision-publication.json"
BEFORE = "\\title{Before}\n"
AFTER = "\\title{After}\n"


# --- helpers -----------------------------------------------------------------------------


def _write_review(paper_dir: Path) -> Path:
    review = paper_dir / "ai_review.before_revision.json"
    review.write_text(json.dumps(review_payload()), encoding="utf-8")
    return review


def _pair(root: Path, revision_name: str = "revision") -> tuple[Path, Path]:
    paper_dir = root / "paper"
    revision_dir = root / revision_name
    write_workspace(paper_dir, title="Before")
    write_workspace(revision_dir, title="After")
    return paper_dir, revision_dir


def _promote(paper_dir: Path, revision_dir: Path, **kwargs: Any) -> Path:
    return promote_revision_workspace(
        paper_dir=paper_dir,
        revision_dir=revision_dir,
        before_review_path=_write_review(paper_dir),
        **kwargs,
    )


def _tex(directory: Path) -> str:
    return (directory / "paper.tex").read_text(encoding="utf-8")


def _raise(exc: BaseException):
    def raiser(*_args: object, **_kwargs: object) -> None:
        raise exc

    return raiser


def _workspace_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _write_original_only_files(paper_dir: Path) -> dict[str, bytes]:
    files = {
        "memory.md": b"Original author notes.\n",
        "reference_plan.json": b'{"reference": "original"}\n',
        "references/planning/queries.json": b'["original search"]\n',
        "scripts/build_figures.py": b"print('original figure source')\n",
    }
    for relative, content in files.items():
        target = paper_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    hashes = refresh_evidence_audit(paper_dir)
    assert all(f"evidence-source/{relative}" in hashes for relative in files)
    return files


def _rename_exchange(tmp_path: Path):
    """Exchange paper/replacement via plain renames (stands in for the atomic syscall)."""

    def exchange(record, *, parent: Path) -> None:
        left = parent / record.paper_name
        right = parent / record.replacement_name
        temporary = tmp_path / ".exchange-temporary"
        left.rename(temporary)
        right.rename(left)
        temporary.rename(right)

    return exchange


# --- promote_revision_workspace ----------------------------------------------------------


def test_promote_revision_publishes_manifest_last_and_every_audit_bound_input(tmp_path: Path) -> None:
    # preserves the original snapshot and publishes the manifest last
    root = tmp_path / "manifest"
    paper_dir, revision_dir = _pair(root)
    before_review = _write_review(paper_dir)

    snapshot = promote_revision_workspace(
        paper_dir=paper_dir, revision_dir=revision_dir, before_review_path=before_review
    )

    assert _tex(snapshot) == BEFORE
    assert _tex(paper_dir) == AFTER
    manifest = json.loads((paper_dir / "latest_manifest.json").read_text(encoding="utf-8"))
    assert manifest["paper_pdf"] == str((paper_dir / "paper.pdf").resolve())
    assert manifest["revision"]["status"] == "promoted"
    assert manifest["revision"]["before_review"] == str(before_review.resolve())
    assert (snapshot / "paper_prompt.md").read_text(encoding="utf-8") == "original prompt\n"
    original_manifest = json.loads((snapshot / "original_manifest.json").read_text(encoding="utf-8"))
    assert original_manifest["paper_pdf"] == str((paper_dir / "paper.pdf").resolve())
    assert not list(root.glob(".paper.revision-*"))

    # publishes every audit-bound input
    paper_dir, revision_dir = _pair(tmp_path / "audit")
    (revision_dir / "artifact.bin").write_bytes(b"evidence-file\n")
    for relative, content in (
        ("artifacts/nested/result.bin", b"evidence-tree\n"),
        ("sections/details.tex", b"Detailed evidence.\n"),
        ("styles/custom.bst", b"ENTRY{}{}{}\n"),
        ("figures/result.png", b"PNG evidence\n"),
    ):
        target = revision_dir / relative
        target.parent.mkdir(parents=True)
        target.write_bytes(content)
    (revision_dir / "paper.tex").write_text(
        "\\title{After}\n"
        "\\input{sections/details}\n"
        "\\bibliographystyle{styles/custom}\n"
        "\\includegraphics{figures/result.png}\n",
        encoding="utf-8",
    )
    scientific_evidence = revision_dir / "scientific-evidence"
    scientific_evidence.mkdir()
    (scientific_evidence / "verification-report.json").write_text(
        json.dumps(
            {
                "schema_name": "scientific-evidence.verification-report",
                "summary": {"dangling_references": 0},
                "results": [
                    {"status": "resolved", "resolved_path": "artifact.bin"},
                    {"status": "resolved", "resolved_path": "artifacts"},
                ],
            }
        ),
        encoding="utf-8",
    )
    expected_hashes = refresh_evidence_audit(revision_dir)

    _promote(paper_dir, revision_dir)

    published_hashes = audit_input_sha256(paper_dir)
    for path, digest in expected_hashes.items():
        if path != "evidence-source/latest_manifest.json":
            assert published_hashes.get(path) == digest
    assert {
        "evidence-source/artifact.bin",
        "evidence-tree/artifacts",
        "tex-source/sections/details.tex",
        "compile-input/styles/custom.bst",
        "figure-asset/figures/result.png",
        "scientific-evidence/verification-report.json",
    } <= expected_hashes.keys()


def test_promote_revision_preserves_canonical_aliases_and_rejects_wrong_targets(tmp_path: Path) -> None:
    # preserves canonical workspace aliases
    root = tmp_path / "canonical"
    paper_dir, revision_dir = _pair(root)
    aliases = {
        "task": root / "task",
        "results": root / "investigation",
        "global_memory": root / "investigation",
        "aggregated": root / "ablations" / "aggregated",
        "extra_tasks": root / "ablations" / "extra_tasks",
        "ablations": root / "ablations",
        "posthoc": root / "posthoc",
    }
    for target in set(aliases.values()):
        target.mkdir(parents=True, exist_ok=True)
    for name, target in aliases.items():
        (paper_dir / name).symlink_to(target, target_is_directory=True)

    _promote(
        paper_dir,
        revision_dir,
        workspace_alias_targets={name: (target,) for name, target in aliases.items()},
    )

    for name, target in aliases.items():
        assert (paper_dir / name).is_symlink()
        assert (paper_dir / name).resolve() == target.resolve()

    # rejects a workspace alias with the wrong target
    root = tmp_path / "wrong"
    paper_dir, revision_dir = _pair(root)
    unexpected = root / "unexpected"
    unexpected.mkdir()
    (root / "posthoc").mkdir()
    (paper_dir / "posthoc").symlink_to(unexpected, target_is_directory=True)

    with pytest.raises(ValueError, match="noncanonical workspace alias"):
        _promote(paper_dir, revision_dir, workspace_alias_targets={"posthoc": (root / "posthoc",)})


def test_promote_revision_rejects_symlinks_and_oversized_trees(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # rejects a nested symlink inside the paper tree
    root = tmp_path / "nested"
    paper_dir, revision_dir = _pair(root)
    assets = paper_dir / "assets"
    assets.mkdir()
    target = root / "target.txt"
    target.write_text("target\n", encoding="utf-8")
    (assets / "linked.txt").symlink_to(target)

    with pytest.raises(ValueError, match="unexpected symlink"):
        _promote(paper_dir, revision_dir)

    # rejects a symlinked authoritative artifact
    root = tmp_path / "artifact"
    paper_dir, revision_dir = _pair(root)
    (revision_dir / "paper.pdf").unlink()
    (revision_dir / "paper.pdf").symlink_to(root / "victim.pdf")

    with pytest.raises(ValueError, match="regular"):
        _promote(paper_dir, revision_dir)

    assert _tex(paper_dir) == BEFORE

    # rejects a symlinked paper root
    root = tmp_path / "paper-root"
    real_paper_dir = root / "real-paper"
    revision_dir = root / "revision"
    write_workspace(real_paper_dir, title="Before")
    write_workspace(revision_dir, title="After")
    paper_link = root / "paper"
    paper_link.symlink_to(real_paper_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="paper directory"):
        promote_revision_workspace(
            paper_dir=paper_link,
            revision_dir=revision_dir,
            before_review_path=_write_review(real_paper_dir),
        )

    # the existing paper tree is bounded before publication
    paper_dir, revision_dir = _pair(tmp_path / "bounded")
    (paper_dir / "extra-a.txt").write_text("a", encoding="utf-8")
    (paper_dir / "extra-b.txt").write_text("b", encoding="utf-8")
    monkeypatch.setattr(revision_publication, "_MAX_COPY_ENTRIES", 1)

    with pytest.raises(ValueError, match="too many entries"):
        _promote(paper_dir, revision_dir)

    assert _tex(paper_dir) == BEFORE


def test_promote_revision_handles_injected_copy_cleanup_and_validation_races(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # succeeds when post-commit cleanup fails
    paper_dir, revision_dir = _pair(tmp_path / "cleanup")
    real_remove = revision_publication.shutil.rmtree

    def fail_old_paper_cleanup(path: Path) -> None:
        if path.name.startswith(".paper.revision-"):
            raise OSError("cleanup unavailable")
        real_remove(path)

    with monkeypatch.context() as patch:
        patch.setattr(revision_publication.shutil, "rmtree", fail_old_paper_cleanup)
        snapshot = _promote(paper_dir, revision_dir)

    assert snapshot == paper_dir / "revisions" / "revision_01" / "before"
    assert _tex(paper_dir) == AFTER

    # publishes the validated snapshot when the source changes after validation
    paper_dir, revision_dir = _pair(tmp_path / "mutated")
    real_validate = revision_publication._validate_revision_workspace

    def mutate_source_after_validation(workspace: Path, *, manifest_artifact_root=None):
        manifest = real_validate(workspace, manifest_artifact_root=manifest_artifact_root)
        if workspace != revision_dir:
            (revision_dir / "paper.tex").write_text("\\title{Mutated}\n", encoding="utf-8")
        return manifest

    with monkeypatch.context() as patch:
        patch.setattr(revision_publication, "_validate_revision_workspace", mutate_source_after_validation)
        _promote(paper_dir, revision_dir)

    assert _tex(paper_dir) == AFTER

    # rolls back after a mid-publication failure
    paper_dir, revision_dir = _pair(tmp_path / "rollback")
    original_manifest = (paper_dir / "latest_manifest.json").read_bytes()
    real_copy = revision_publication._atomic_copy_file_at

    def fail_on_pdf(source_fd: int, name: str, source: Path, target: Path, expected) -> None:
        if source.parent.name.startswith(".revision-source-") and target.name == "paper.pdf":
            raise OSError("injected publication failure")
        real_copy(source_fd, name, source, target, expected)

    with monkeypatch.context() as patch:
        patch.setattr(revision_publication, "_atomic_copy_file_at", fail_on_pdf)
        with pytest.raises(OSError, match="injected"):
            _promote(paper_dir, revision_dir)

    assert _tex(paper_dir) == BEFORE
    assert (paper_dir / "latest_manifest.json").read_bytes() == original_manifest


# --- original-only audit inputs (ancillary archive via promotion) -------------------------


def test_publication_archives_original_only_and_displaced_audit_inputs(tmp_path: Path) -> None:
    # archives original-only audit inputs under the snapshot's ancillary tree
    paper_dir, revision_dir = _pair(tmp_path / "original-only")
    original_only = _write_original_only_files(paper_dir)
    revision_before = _workspace_bytes(revision_dir)

    snapshot = _promote(paper_dir, revision_dir)

    for relative, content in original_only.items():
        assert not (paper_dir / relative).exists()
        assert (snapshot / "ancillary" / relative).read_bytes() == content
    validate_published_revision_evidence(snapshot, paper_dir)
    assert _workspace_bytes(revision_dir) == revision_before

    # uses the revised bytes for shared audit inputs
    paper_dir, revision_dir = _pair(tmp_path / "shared")
    original_only = _write_original_only_files(paper_dir)
    revised_memory = b"Revised author notes.\n"
    (revision_dir / "memory.md").write_bytes(revised_memory)
    refresh_evidence_audit(revision_dir)

    snapshot = _promote(paper_dir, revision_dir)

    assert (paper_dir / "memory.md").read_bytes() == revised_memory
    assert not (snapshot / "ancillary" / "memory.md").exists()
    for relative, content in original_only.items():
        if relative != "memory.md":
            assert (snapshot / "ancillary" / relative).read_bytes() == content
    validate_published_revision_evidence(snapshot, paper_dir)

    # archives displaced binary tree inputs for both revision layouts
    for revision_layout in ("kept_file", "replaced_directory"):
        paper_dir, revision_dir = _pair(tmp_path / revision_layout)
        for workspace in (paper_dir, revision_dir):
            (workspace / "evidence").mkdir()
            report_dir = workspace / "scientific-evidence"
            report_dir.mkdir()
            (report_dir / "verification-report.json").write_text(
                json.dumps({"results": [{"resolved_path": "evidence"}]}), encoding="utf-8"
            )
        (paper_dir / "evidence" / "nested").mkdir()
        (paper_dir / "evidence" / "nested" / "old.bin").write_bytes(b"old binary evidence")
        if revision_layout == "kept_file":
            (paper_dir / "evidence" / "kept.bin").write_bytes(b"original kept")
            (revision_dir / "evidence" / "kept.bin").write_bytes(b"revised kept")
        else:
            (revision_dir / "evidence" / "nested").write_bytes(b"new file")
        refresh_evidence_audit(paper_dir)
        refresh_evidence_audit(revision_dir)

        snapshot = _promote(paper_dir, revision_dir)

        assert (snapshot / "ancillary" / "evidence" / "nested" / "old.bin").read_bytes() == (
            b"old binary evidence"
        ), revision_layout
        validate_published_revision_evidence(snapshot, paper_dir)


def test_completed_revision_with_original_only_inputs_reuses_attempt_and_reviews(
    tmp_path: Path,
) -> None:
    cfg = revision_config(tmp_path)
    paths = derive_paths(cfg)
    write_workspace(paths.paper_dir, title="Before")
    original_only = _write_original_only_files(paths.paper_dir)
    _write_review(paths.paper_dir)
    reusable = paths.revision_dir / "attempt_01"
    write_workspace(reusable, title="Reusable")
    reviews: list[str] = []

    def review(**kwargs: object) -> FinalAiReviewResult:
        assert kwargs["artifact_stem"] == "ai_review.after_revision"
        assert "Reusable" in _tex(paths.paper_dir)
        reviews.append("after_revision")
        return FinalAiReviewResult(
            status="completed",
            review_path=paths.paper_dir / "ai_review.after_revision.json",
            log_path=paths.paper_dir / "ai_review.after_revision.log",
        )

    result = run_pipeline_revision(
        cfg=cfg,
        paths=paths,
        execute_command=lambda _command: pytest.fail("completed attempt must not edit again"),
        review_runner=review,
    )

    assert result.status == "completed"
    assert result.revision_dir == reusable
    assert reviews == ["after_revision"]
    snapshot = paths.paper_dir / "revisions" / "revision_01" / "before"
    for relative, content in original_only.items():
        assert (snapshot / "ancillary" / relative).read_bytes() == content
    validate_published_revision_evidence(snapshot, paths.paper_dir)


def test_staging_failures_and_tampered_inputs_preserve_original_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # a staging failure after archival preserves the original publication
    root = tmp_path / "staging"
    paper_dir, revision_dir = _pair(root)
    original_only = _write_original_only_files(paper_dir)
    _write_review(paper_dir)
    original = _workspace_bytes(paper_dir)

    def fail_sync(staged_dir: Path) -> None:
        ancillary = staged_dir / "revisions" / "revision_01" / "before" / "ancillary"
        for relative, content in original_only.items():
            assert (ancillary / relative).read_bytes() == content
            assert not (staged_dir / relative).exists()
        raise OSError("injected staging failure after archival")

    with monkeypatch.context() as patch:
        patch.setattr(revision_publication, "_sync_publication_tree", fail_sync)
        with pytest.raises(OSError, match="injected staging failure after archival"):
            _promote(paper_dir, revision_dir)

    assert _workspace_bytes(paper_dir) == original
    assert not list(root.glob(".paper.revision-*"))

    # unexpected staged evidence still rejects publication
    paper_dir, revision_dir = _pair(tmp_path / "unexpected")
    _write_original_only_files(paper_dir)
    _write_review(paper_dir)
    original = _workspace_bytes(paper_dir)
    real_sync = revision_publication._sync_publication_tree

    def add_unexpected_evidence(staged_dir: Path) -> None:
        (staged_dir / "unexpected.json").write_text('{"unaudited": true}\n', encoding="utf-8")
        real_sync(staged_dir)

    with monkeypatch.context() as patch:
        patch.setattr(revision_publication, "_sync_publication_tree", add_unexpected_evidence)
        with pytest.raises(ValueError, match="current passing agentic evidence audit"):
            _promote(paper_dir, revision_dir)

    assert _workspace_bytes(paper_dir) == original

    # a tampered revision input rejects publication
    paper_dir, revision_dir = _pair(tmp_path / "tampered")
    _write_original_only_files(paper_dir)
    (revision_dir / "memory.md").write_text("Audited notes.\n", encoding="utf-8")
    refresh_evidence_audit(revision_dir)
    (revision_dir / "memory.md").write_text("Changed after audit.\n", encoding="utf-8")
    _write_review(paper_dir)
    original = _workspace_bytes(paper_dir)

    with pytest.raises(ValueError):
        _promote(paper_dir, revision_dir)

    assert _workspace_bytes(paper_dir) == original


# --- archive_original_only_inputs ----------------------------------------------------------


def _archive_dirs(root: Path, *, memory: bool = True, snapshot: bool = True) -> tuple[Path, Path, Path]:
    staged = root / "staged"
    staged.mkdir(parents=True)
    if memory:
        (staged / "memory.md").write_text("original")
    revision = root / "revision"
    revision.mkdir()
    snapshot_dir = staged / "revisions" / "before"
    if snapshot:
        snapshot_dir.mkdir(parents=True)
    return staged, revision, snapshot_dir


def _evidence_tree_dirs(root: Path) -> tuple[Path, Path, Path]:
    staged = root / "staged"
    (staged / "evidence" / "nested").mkdir(parents=True)
    (staged / "evidence" / "nested" / "old.md").write_text("old tree")
    revision = root / "revision"
    (revision / "evidence").mkdir(parents=True)
    snapshot = staged / "revisions" / "before"
    snapshot.mkdir(parents=True)
    return staged, revision, snapshot


def test_archive_rejects_existing_destination_symlinked_parent_and_unsafe_input(tmp_path: Path) -> None:
    # rejects an existing destination
    staged, revision, snapshot = _archive_dirs(tmp_path / "existing", snapshot=False)
    (snapshot / "ancillary").mkdir(parents=True)
    (snapshot / "ancillary" / "memory.md").write_text("preserved")

    with pytest.raises(FileExistsError):
        archive_original_only_inputs(staged, revision, snapshot, {})

    assert (staged / "memory.md").read_text() == "original"
    assert (snapshot / "ancillary" / "memory.md").read_text() == "preserved"

    # rejects a symlinked snapshot parent
    root = tmp_path / "symlinked"
    staged, revision, _ = _archive_dirs(root, snapshot=False)
    outside = root / "outside"
    outside.mkdir()
    (staged / "revisions").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="regular directory"):
        archive_original_only_inputs(staged, revision, staged / "revisions", {})

    assert not list(outside.iterdir())
    assert (staged / "memory.md").read_text() == "original"

    # rejects an unsafe revision input path
    staged, revision, snapshot = _archive_dirs(tmp_path / "unsafe", memory=False)

    with pytest.raises(ValueError, match="unsafe audit-bound path"):
        archive_original_only_inputs(
            staged, revision, snapshot, {"../outside": hashlib.sha256(b"").hexdigest()}
        )


def test_archive_rejects_concurrent_changes_oversized_files_and_revision_symlinks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # rejects an original that changed during the copy
    staged, revision, snapshot = _archive_dirs(tmp_path / "changed")
    real_copy = revision_ancillary_archive._copy_regular_file

    def change_after_copy(source_fd, source_name, target_fd, target_name, budget) -> None:
        real_copy(source_fd, source_name, target_fd, target_name, budget)
        replacement = staged / "replacement"
        replacement.write_text("concurrently replaced")
        os.replace(replacement, staged / source_name)

    with monkeypatch.context() as patch:
        patch.setattr(revision_ancillary_archive, "_copy_regular_file", change_after_copy)
        with pytest.raises(ValueError, match="changed during archival"):
            archive_original_only_inputs(staged, revision, snapshot, {})

    assert (staged / "memory.md").read_text() == "concurrently replaced"
    assert (snapshot / "ancillary" / "memory.md").read_text() == "original"

    # a copy failure keeps the active original
    staged, revision, snapshot = _archive_dirs(tmp_path / "oversized")
    with monkeypatch.context() as patch:
        patch.setattr(revision_audit_publication, "_MAX_FILE_BYTES", 1)
        with pytest.raises(ValueError, match="too large"):
            archive_original_only_inputs(staged, revision, snapshot, {})

    assert (staged / "memory.md").read_text() == "original"
    assert not (snapshot / "ancillary" / "memory.md").exists()

    # rejects symlinks in the revision tree
    for symlink_path in ("nested", "nested/old.md"):
        staged, revision, snapshot = _evidence_tree_dirs(tmp_path / symlink_path.replace("/", "-"))
        link = revision / "evidence" / symlink_path
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(staged / "evidence" / symlink_path)

        with pytest.raises((ValueError, OSError)):
            archive_original_only_inputs(staged, revision, snapshot, {"evidence-tree/evidence": "0" * 64})

        assert (staged / "evidence" / "nested" / "old.md").read_text() == "old tree", symlink_path


def test_archive_keeps_revision_bound_files_and_preserves_replaced_directories(tmp_path: Path) -> None:
    # keeps files present in the revision-bound tree
    root = tmp_path / "kept"
    staged = root / "staged"
    (staged / "evidence").mkdir(parents=True)
    (staged / "evidence" / "kept.md").write_text("original kept")
    (staged / "evidence" / "old.md").write_text("original only")
    revision = root / "revision"
    (revision / "evidence").mkdir(parents=True)
    (revision / "evidence" / "kept.md").write_text("revised kept")
    snapshot = staged / "revisions" / "before"
    snapshot.mkdir(parents=True)

    archive_original_only_inputs(staged, revision, snapshot, {"evidence-tree/evidence": "0" * 64})

    assert (staged / "evidence" / "kept.md").read_text() == "original kept"
    assert not (staged / "evidence" / "old.md").exists()
    assert (snapshot / "ancillary" / "evidence" / "old.md").read_text() == "original only"

    # preserves old files when the revision tree replaces a directory with a file
    staged, revision, snapshot = _evidence_tree_dirs(tmp_path / "replaced")
    (revision / "evidence" / "nested").write_text("new file")

    archive_original_only_inputs(staged, revision, snapshot, {"evidence-tree/evidence": "0" * 64})

    assert (snapshot / "ancillary" / "evidence" / "nested" / "old.md").read_text() == "old tree"


# --- copy_audit_bound_inputs ---------------------------------------------------------------


def _digest(content: bytes = b"input\n") -> str:
    return hashlib.sha256(content).hexdigest()


def _symlink_to_victim(tmp_path: Path, entry: Path) -> None:
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"input\n")
    entry.symlink_to(victim)


@pytest.mark.parametrize(
    ("kind", "key", "match"),
    [
        pytest.param("traversal", "evidence-source/../outside.txt", "unsafe audit-bound path", id="traversal"),
        pytest.param("symlink", "evidence-source/artifact.bin", "regular file", id="symlink"),
        pytest.param("special", "evidence-source/artifact.bin", "regular file", id="special"),
        pytest.param("oversized", "evidence-source/artifact.bin", "too large", id="oversized"),
        pytest.param("tree_symlink", "evidence-tree/artifacts", "tree entry is not regular", id="tree_symlink"),
    ],
)
def test_audit_bound_copy_rejects_unsafe_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str, key: str, match: str
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    entry = source / "artifact.bin"
    if kind == "symlink":
        _symlink_to_victim(tmp_path, entry)
    elif kind == "special":
        os.mkfifo(entry)
    elif kind == "oversized":
        entry.write_bytes(b"input\n")
        monkeypatch.setattr(revision_audit_publication, "_MAX_FILE_BYTES", 4)
    elif kind == "tree_symlink":
        tree = source / "artifacts"
        tree.mkdir()
        _symlink_to_victim(tmp_path, tree / "linked.bin")

    with pytest.raises(ValueError, match=match):
        copy_audit_bound_inputs(source, target, {key: _digest()})


# --- portable publication (atomic exchange fallback, journal, recovery) -------------------


def _unsupported_exchange(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        revision_publication,
        "_atomic_exchange",
        _raise(OSError(errno.EINVAL, "atomic exchange unsupported")),
    )


def test_atomic_exchange_fallback_only_for_unsupported_syscall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # falls back to the portable replacement when the atomic exchange is unsupported
    root = tmp_path / "unsupported"
    paper_dir, revision_dir = _pair(root)
    with monkeypatch.context() as patch:
        _unsupported_exchange(patch)
        snapshot = _promote(paper_dir, revision_dir)

    assert _tex(paper_dir) == AFTER
    assert _tex(snapshot) == BEFORE
    assert not (root / JOURNAL).exists()
    assert not list(root.glob(".paper.original-*"))

    # does not fall back for an unexpected exchange error
    root = tmp_path / "denied"
    paper_dir, revision_dir = _pair(root)
    with monkeypatch.context() as patch:
        patch.setattr(
            revision_publication, "_atomic_exchange", _raise(PermissionError(errno.EPERM, "denied"))
        )
        with pytest.raises(PermissionError, match="denied"):
            _promote(paper_dir, revision_dir)

    assert _tex(paper_dir) == BEFORE
    assert not (root / JOURNAL).exists()

    # the atomic exchange requires the prepared directory identities
    root = tmp_path / "identity"
    paper_dir, revision_dir = _pair(root)
    real_exchange = portable_publication.atomic_exchange_prepared

    def substitute_before_exchange(record, *, parent: Path):
        replacement = root / record.replacement_name
        replacement.rename(root / ".parked-replacement")
        replacement.mkdir()
        return real_exchange(record, parent=parent)

    with monkeypatch.context() as patch:
        patch.setattr(revision_publication, "_atomic_exchange", substitute_before_exchange)
        with pytest.raises(RuntimeError, match="entry changed"):
            _promote(paper_dir, revision_dir)

    assert _tex(paper_dir) == BEFORE


def test_replacement_trees_are_fsynced_before_exchange_and_prepared_journal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # the staged tree is synced before the atomic exchange
    root = tmp_path / "exchange"
    paper_dir, revision_dir = _pair(root)
    nested = revision_dir / "figures" / "nested"
    nested.mkdir(parents=True)
    (nested / "figure.txt").write_text("figure\n", encoding="utf-8")
    refresh_evidence_audit(revision_dir)
    real_fsync = publication_tree.os.fsync
    synced_inodes: set[int] = set()
    exchange = _rename_exchange(root)

    def record_fsync(descriptor: int) -> None:
        synced_inodes.add(os.fstat(descriptor).st_ino)
        real_fsync(descriptor)

    def exchange_after_sync(record, *, parent: Path) -> None:
        right = parent / record.replacement_name
        required_inodes = {
            path.stat(follow_symlinks=False).st_ino
            for path in (right, *right.rglob("*"))
            if not path.is_symlink()
        }
        assert required_inodes <= synced_inodes
        exchange(record, parent=parent)

    with monkeypatch.context() as patch:
        patch.setattr(publication_tree.os, "fsync", record_fsync)
        patch.setattr(revision_publication, "_atomic_exchange", exchange_after_sync)
        _promote(paper_dir, revision_dir)

    assert _tex(paper_dir) == AFTER

    # replacement tree directories are fsynced before the prepared journal
    paper_dir, replacement_dir = _pair(tmp_path / "journal", ".paper.revision-sync")
    nested_dir = replacement_dir / "figures" / "nested"
    nested_dir.mkdir(parents=True)
    (nested_dir / "figure.txt").write_text("figure\n", encoding="utf-8")
    expected_directory_inodes = {
        replacement_dir.stat().st_ino,
        (replacement_dir / "figures").stat().st_ino,
        nested_dir.stat().st_ino,
    }
    real_fsync = portable_publication.os.fsync
    synced_directory_inodes: set[int] = set()

    def record_directory_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if stat.S_ISDIR(metadata.st_mode):
            synced_directory_inodes.add(metadata.st_ino)
        real_fsync(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(portable_publication.os, "fsync", record_directory_fsync)
        patch.setattr(portable_publication, "_advance_publication", lambda *_args, **_kwargs: None)
        portable_publication.replace_directory_portably(paper_dir, replacement_dir)

    assert expected_directory_inodes <= synced_directory_inodes


def test_post_exchange_cleanup_and_fsync_failures_recover_from_durable_intent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # a post-exchange cleanup failure leaves a journal that recovery consumes
    root = tmp_path / "cleanup"
    paper_dir, revision_dir = _pair(root)
    with monkeypatch.context() as patch:
        patch.setattr(revision_publication, "_atomic_exchange", _rename_exchange(root))
        patch.setattr(portable_publication, "_remove_backup", _raise(OSError("cleanup failed")))
        with pytest.raises(OSError, match="cleanup failed"):
            _promote(paper_dir, revision_dir)

        assert _tex(paper_dir) == AFTER
        assert (root / JOURNAL).is_file()

    recover_incomplete_promotion(paper_dir)

    assert not (root / JOURNAL).exists()
    assert not list(root.glob(".paper.revision-*"))

    # an exchange fsync failure is recovered from the durable intent
    root = tmp_path / "fsync"
    paper_dir, revision_dir = _pair(root)
    real_sync = portable_publication._sync_directory
    failed = False

    def fail_first_post_exchange_sync(directory_fd: int) -> None:
        nonlocal failed
        try:
            paper_title = _tex(paper_dir)
        except FileNotFoundError:
            real_sync(directory_fd)
            return
        if paper_title == AFTER and not failed:
            failed = True
            raise OSError(errno.EIO, "injected post-exchange fsync failure")
        real_sync(directory_fd)

    with monkeypatch.context() as patch:
        patch.setattr(portable_publication, "_sync_directory", fail_first_post_exchange_sync)
        with pytest.raises(OSError, match="post-exchange fsync failure"):
            _promote(paper_dir, revision_dir)

        journal = root / JOURNAL
        assert journal.is_file()
    recover_incomplete_promotion(paper_dir)

    assert _tex(paper_dir) == AFTER
    assert not journal.exists()
    assert not list(root.glob(".paper.revision-*"))


@pytest.mark.parametrize(
    ("failure_point", "paper_exists_before_recovery"),
    [("prepared", True), ("original_moved", False), ("replacement_moved", True)],
)
def test_recover_incomplete_portable_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_point: str, paper_exists_before_recovery: bool
) -> None:
    paper_dir, revision_dir = _pair(tmp_path)
    _unsupported_exchange(monkeypatch)
    real_rename = portable_publication._rename_entry
    rename_calls = 0

    def interrupt_rename(parent_fd: int, source: str, destination: str) -> None:
        nonlocal rename_calls
        rename_calls += 1
        if failure_point == "prepared" and rename_calls == 1:
            raise RuntimeError("injected prepared interruption")
        if failure_point == "original_moved" and rename_calls == 2:
            raise RuntimeError("injected original-moved interruption")
        real_rename(parent_fd, source, destination)

    monkeypatch.setattr(portable_publication, "_rename_entry", interrupt_rename)
    real_remove_backup = portable_publication._remove_backup

    def interrupt_cleanup(parent_fd: int, backup_name: str, expected: DirectoryIdentity) -> None:
        if failure_point == "replacement_moved":
            raise RuntimeError("injected replacement-moved interruption")
        real_remove_backup(parent_fd, backup_name, expected)

    monkeypatch.setattr(portable_publication, "_remove_backup", interrupt_cleanup)

    with pytest.raises(RuntimeError, match="injected"):
        _promote(paper_dir, revision_dir)

    assert paper_dir.exists() is paper_exists_before_recovery
    assert (tmp_path / JOURNAL).is_file()

    monkeypatch.setattr(portable_publication, "_rename_entry", real_rename)
    monkeypatch.setattr(portable_publication, "_remove_backup", real_remove_backup)
    recover_incomplete_promotion(paper_dir)

    assert _tex(paper_dir) == AFTER
    assert not (tmp_path / JOURNAL).exists()
    assert not list(tmp_path.glob(".paper.original-*"))


def test_portable_publication_journal_is_created_exclusively_and_atomically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # does not overwrite an existing journal
    root = tmp_path / "existing"
    paper_dir, revision_dir = _pair(root)
    journal = root / JOURNAL
    journal.write_text('{"untrusted": true}\n', encoding="utf-8")
    with monkeypatch.context() as patch:
        _unsupported_exchange(patch)
        with pytest.raises(RuntimeError, match="journal already exists"):
            _promote(paper_dir, revision_dir)

    assert journal.read_text(encoding="utf-8") == '{"untrusted": true}\n'
    assert _tex(paper_dir) == BEFORE

    # creates the initial journal exclusively
    root = tmp_path / "race"
    paper_dir, replacement_dir = _pair(root, ".paper.revision-race")
    journal = root / JOURNAL
    journal.write_text('{"created_during_preflight": true}\n', encoding="utf-8")
    with monkeypatch.context() as patch:
        patch.setattr(portable_publication, "portable_replacement_pending", lambda _path: False)
        with pytest.raises(RuntimeError, match="journal already exists"):
            portable_publication.replace_directory_portably(paper_dir, replacement_dir)

    assert journal.read_text(encoding="utf-8") == '{"created_during_preflight": true}\n'
    assert _tex(paper_dir) == BEFORE

    # a partial initial journal write never exposes the journal
    root = tmp_path / "partial"
    paper_dir, replacement_dir = _pair(root, ".paper.revision-partial")
    real_write = portable_publication.os.write
    write_calls = 0

    def interrupt_write(descriptor: int, content: bytes) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return real_write(descriptor, content[:8])
        raise OSError("injected partial journal write")

    with monkeypatch.context() as patch:
        patch.setattr(portable_publication.os, "write", interrupt_write)
        with pytest.raises(OSError, match="partial journal write"):
            portable_publication.replace_directory_portably(paper_dir, replacement_dir)

    assert not (root / JOURNAL).exists()
    assert not list(root.glob("..paper.revision-publication.*.tmp"))
    assert _tex(paper_dir) == BEFORE
    assert _tex(replacement_dir) == AFTER


def _interrupt_after_journal(
    monkeypatch: pytest.MonkeyPatch, paper_dir: Path, replacement_dir: Path, message: str
):
    """Start a portable replacement that stops right after the journal is written."""
    real_rename = portable_publication._rename_entry
    monkeypatch.setattr(portable_publication, "_rename_entry", _raise(RuntimeError(message)))
    with pytest.raises(RuntimeError, match=message):
        portable_publication.replace_directory_portably(paper_dir, replacement_dir)
    return real_rename


def test_hostile_or_deeply_nested_journal_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # a hostile non-string journal stage is rejected
    root = tmp_path / "hostile"
    paper_dir, replacement_dir = _pair(root, ".paper.revision-hostile")
    with monkeypatch.context() as patch:
        _interrupt_after_journal(patch, paper_dir, replacement_dir, "stop after journal")
    journal = root / JOURNAL
    payload = json.loads(journal.read_text(encoding="utf-8"))
    payload["stage"] = []
    journal.write_text(json.dumps(payload), encoding="utf-8")

    assert not portable_publication.portable_replacement_is_recoverable(paper_dir)
    with pytest.raises(portable_publication.PortablePublicationError, match="stage is invalid"):
        portable_publication.recover_portable_replacement(paper_dir)

    # a deeply nested journal is rejected without a recursion error
    root = tmp_path / "nested"
    paper_dir = root / "paper"
    paper_dir.mkdir(parents=True)
    (root / JOURNAL).write_text("[" * 2000 + "0" + "]" * 2000, encoding="utf-8")

    assert not portable_publication.portable_replacement_is_recoverable(paper_dir)
    with pytest.raises(
        portable_publication.PortablePublicationError,
        match="journal is invalid|unsupported portable publication journal",
    ):
        portable_publication.recover_portable_replacement(paper_dir)


def test_recovery_rejects_modified_or_semantically_invalid_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # rejects a staged replacement modified after the journal was written
    paper_dir, replacement_dir = _pair(tmp_path / "modified", ".paper.revision-modified")
    with monkeypatch.context() as patch:
        _interrupt_after_journal(patch, paper_dir, replacement_dir, "stop after journal")
    (replacement_dir / "paper.tex").write_text("mutated\n", encoding="utf-8")

    with pytest.raises(portable_publication.PortablePublicationError, match="digest changed"):
        portable_publication.recover_portable_replacement(paper_dir)

    assert _tex(paper_dir) == BEFORE

    # rejects a self-consistent but semantically invalid replacement
    paper_dir, replacement_dir = _pair(tmp_path / "invalid", ".paper.revision-invalid")
    (replacement_dir / "paper_data.json").write_text("{}\n", encoding="utf-8")
    refresh_evidence_audit(replacement_dir)
    with monkeypatch.context() as patch:
        _interrupt_after_journal(patch, paper_dir, replacement_dir, "prepared interruption")

    with pytest.raises(ValueError, match="loses or changes"):
        recover_incomplete_promotion(paper_dir)

    assert _tex(paper_dir) == BEFORE


@pytest.mark.parametrize("journal_lag", ["original_moved", "replacement_moved", "cleanup"])
def test_recovery_infers_completed_transition_before_journal_update(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, journal_lag: str
) -> None:
    paper_dir, replacement_dir = _pair(tmp_path, ".paper.revision-journal-lag")
    real_write_record = portable_publication._write_record
    real_remove_journal = portable_publication._remove_journal

    def interrupt_stage_write(parent: Path, parent_fd: int, record: PublicationRecord) -> PublicationRecord:
        if record.stage == journal_lag:
            raise RuntimeError("injected journal lag")
        return real_write_record(parent, parent_fd, record)

    def interrupt_journal_removal(parent: Path, parent_fd: int, record: PublicationRecord) -> None:
        if journal_lag == "cleanup":
            raise RuntimeError("injected journal lag")
        real_remove_journal(parent, parent_fd, record)

    monkeypatch.setattr(portable_publication, "_write_record", interrupt_stage_write)
    monkeypatch.setattr(portable_publication, "_remove_journal", interrupt_journal_removal)
    with pytest.raises(RuntimeError, match="journal lag"):
        portable_publication.replace_directory_portably(paper_dir, replacement_dir)
    monkeypatch.setattr(portable_publication, "_write_record", real_write_record)
    monkeypatch.setattr(portable_publication, "_remove_journal", real_remove_journal)

    portable_publication.recover_portable_replacement(paper_dir)

    assert _tex(paper_dir) == AFTER
    assert not (tmp_path / JOURNAL).exists()


def test_recovery_resumes_backup_quarantine_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paper_dir, replacement_dir = _pair(tmp_path, ".paper.revision-quarantine")
    real_remove_contents = publication_cleanup._remove_contents
    failed = False

    def interrupt_after_quarantine(directory_fd: int) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected quarantine interruption")
        real_remove_contents(directory_fd)

    monkeypatch.setattr(publication_cleanup, "_remove_contents", interrupt_after_quarantine)
    with pytest.raises(OSError, match="quarantine interruption"):
        portable_publication.replace_directory_portably(paper_dir, replacement_dir)

    monkeypatch.setattr(publication_cleanup, "_remove_contents", real_remove_contents)
    portable_publication.recover_portable_replacement(paper_dir)

    assert _tex(paper_dir) == AFTER
    assert not (tmp_path / JOURNAL).exists()
    assert not list(tmp_path.glob(".paper.original-*"))
    assert not [path for path in tmp_path.iterdir() if ".delete-" in path.name]


# --- backup cleanup and tree traversal hardening ------------------------------------------


def test_publication_tree_traversal_enforces_deadline_for_directories(tmp_path: Path) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)

    with pytest.raises(TaskPublicationDeadlineExceeded):
        tuple(regular_tree_files(tmp_path, deadline=0.0))


def test_backup_cleanup_rejects_inode_substitution_before_pinning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backup = tmp_path / ".paper.original-backup"
    parked = tmp_path / ".paper.original-parked"
    victim = tmp_path / "victim"
    backup.mkdir()
    victim.mkdir()
    (backup / "original.txt").write_text("original\n", encoding="utf-8")
    (victim / "victim.txt").write_text("victim\n", encoding="utf-8")
    expected = portable_publication.DirectoryIdentity(
        device=backup.stat().st_dev, inode=backup.stat().st_ino
    )
    real_open = portable_publication.os.open
    swapped = False

    def swap_before_pinning(path, flags, *args, dir_fd=None, **kwargs):
        nonlocal swapped
        if path == backup.name and dir_fd is not None and not swapped:
            swapped = True
            backup.rename(parked)
            victim.rename(backup)
        return real_open(path, flags, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(portable_publication.os, "open", swap_before_pinning)

    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(portable_publication.PortablePublicationError, match="backup changed"):
            portable_publication._remove_backup(parent_fd, backup.name, expected)
    finally:
        os.close(parent_fd)

    assert (backup / "victim.txt").read_text(encoding="utf-8") == "victim\n"
    assert (parked / "original.txt").read_text(encoding="utf-8") == "original\n"


def _remove_pinned_backup(root: Path, expected: os.stat_result) -> None:
    parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(publication_cleanup.PinnedDeletionError):
            publication_cleanup.remove_pinned_directory_at(
                parent_fd, "backup", expected_device=expected.st_dev, expected_inode=expected.st_ino
            )
    finally:
        os.close(parent_fd)


def test_backup_and_child_cleanup_quarantine_before_deleting_after_identity_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # the backup itself is quarantined before deletion after the identity check
    root = tmp_path / "backup-race"
    root.mkdir()
    backup = root / "backup"
    parked = root / "parked"
    victim = root / "victim"
    backup.mkdir()
    victim.mkdir()
    expected = backup.stat()
    real_rename = publication_cleanup.os.rename
    raced = False

    def race_before_quarantine(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
        nonlocal raced
        if source == "backup" and not raced:
            raced = True
            backup.rename(parked)
            victim.rename(backup)
        return real_rename(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    with monkeypatch.context() as patch:
        patch.setattr(publication_cleanup.os, "rename", race_before_quarantine)
        _remove_pinned_backup(root, expected)

    assert parked.is_dir()
    assert any(path.is_dir() for path in root.glob(".backup.delete-*"))

    # a child entry is quarantined before deletion after the identity check
    root = tmp_path / "child-race"
    root.mkdir()
    backup = root / "backup"
    (backup / "child").mkdir(parents=True)
    (backup / "victim").mkdir()
    expected = backup.stat()
    child_raced = False

    # os.listdir order is filesystem dependent, so race whichever sibling is visited
    # first and swap the other one into its place
    def race_child_before_quarantine(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
        nonlocal child_raced
        if source in {"child", "victim"} and not child_raced:
            child_raced = True
            other = "victim" if source == "child" else "child"
            real_rename(source, "parked", src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
            real_rename(other, source, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
        return real_rename(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    with monkeypatch.context() as patch:
        patch.setattr(publication_cleanup.os, "rename", race_child_before_quarantine)
        _remove_pinned_backup(root, expected)

    quarantined_root = next(root.glob(".backup.delete-*"))
    assert (quarantined_root / "parked").is_dir()
    assert any(path.is_dir() for path in quarantined_root.glob(".*.delete-*"))
