"""paper_revision_workspace: snapshots, reference/source-evidence seeding, posthoc preservation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from src.investigation import paper_revision_workspace as workspace
from src.orchestrator.paper_data_posthoc import augment_paper_data_with_posthoc
from src.pipeline.revision_evidence import _validate_semantic_evidence

ROOT = Path(__file__).resolve().parents[2]


def _paper(tmp_path: Path) -> Path:
    """Paper directory whose manuscript references a nested readable source file."""
    paper = tmp_path / "paper"
    source = paper / "source_evidence" / "winner"
    source.mkdir(parents=True)
    (source / "main.py").write_bytes(b"# readable implementation\n")
    (paper / "paper.tex").write_text("source_evidence/winner/main.py\n")
    return paper


def _run_paper_edit_preparation(source: Path, destination: Path) -> subprocess.CompletedProcess:
    """Run the embedded python preparation step of paper_edit.sh."""
    script = (ROOT / "src/scripts/paper_edit.sh").read_text()
    marker = '"$PYTHON_BIN" - "$EDIT_PAPER_DIR" "$OUTPUT_DIR" "$FIGURE_REFRESH_MODE"'
    code = script.split(marker, 1)[1].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    return subprocess.run(
        [sys.executable, "-c", code, str(source), str(destination), "auto", "preserve", "--"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def _copy_source_evidence(operation: str, paper: Path, destination: Path) -> list[dict]:
    if operation == "snapshot":
        return workspace.snapshot_previous_paper(paper, destination)
    return workspace.seed_revision_source_evidence(paper, destination)


# --- snapshots and reference preservation -----------------------------------------------


def test_snapshot_previous_paper_skips_symlinks_and_removes_partial_copy_over_size_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # symlinks are skipped without being followed
    links = tmp_path / "links"
    source = links / "paper"
    source.mkdir(parents=True)
    (source / "paper.tex").write_text("paper\n", encoding="utf-8")
    figures = source / "figures"
    figures.mkdir()
    (figures / "plot.pdf").write_bytes(b"plot")
    linked_tree = links / "posthoc"
    linked_tree.mkdir()
    (linked_tree / "large.json").write_text("evidence\n", encoding="utf-8")
    (source / "posthoc").symlink_to(linked_tree, target_is_directory=True)
    (figures / "linked.pdf").symlink_to(linked_tree / "large.json")
    destination = links / "snapshot"

    skipped = workspace.snapshot_previous_paper(source, destination)

    assert (destination / "paper.tex").read_text(encoding="utf-8") == "paper\n"
    assert (destination / "figures" / "plot.pdf").read_bytes() == b"plot"
    assert not (destination / "posthoc").exists()
    assert not (destination / "figures" / "linked.pdf").exists()
    assert {item["path"] for item in skipped} == {str(source / "posthoc"), str(figures / "linked.pdf")}

    # a partial copy is removed when the size limit is exceeded
    limit = tmp_path / "limit"
    source = limit / "paper"
    source.mkdir(parents=True)
    (source / "paper.tex").write_text("too large\n", encoding="utf-8")
    destination = limit / "snapshot"
    monkeypatch.setattr(workspace, "MAX_SNAPSHOT_BYTES", 3)

    with pytest.raises(ValueError, match="snapshot is too large"):
        workspace.snapshot_previous_paper(source, destination)

    assert not destination.exists()


def test_preserve_revision_references_accepts_aliases_and_rejects_symlinked_artifacts(
    tmp_path: Path,
) -> None:
    # workspace aliases (symlinked posthoc) are accepted
    aliases = tmp_path / "aliases"
    source = aliases / "paper"
    source.mkdir(parents=True)
    (source / "references.bib").write_text("refs\n", encoding="utf-8")
    (source / "references_provenance.json").write_text("[]\n", encoding="utf-8")
    posthoc = aliases / "posthoc"
    posthoc.mkdir()
    (source / "posthoc").symlink_to(posthoc, target_is_directory=True)
    baseline = aliases / "baseline"
    baseline.mkdir()

    workspace.preserve_revision_references(source, baseline)

    assert (baseline / "references.bib").read_text(encoding="utf-8") == "refs\n"
    assert (baseline / "references_provenance.json").read_text(encoding="utf-8") == "[]\n"

    # a symlinked reference artifact is rejected
    for name in ("references.bib", "references_provenance.json"):
        case_dir = tmp_path / name
        source = case_dir / "paper"
        source.mkdir(parents=True)
        for artifact in ("references.bib", "references_provenance.json"):
            (source / artifact).write_text("content\n", encoding="utf-8")
        target = case_dir / "target"
        target.write_text("replacement\n", encoding="utf-8")
        (source / name).unlink()
        (source / name).symlink_to(target)
        baseline = case_dir / "baseline"
        baseline.mkdir()

        with pytest.raises(ValueError, match="Could not preserve revision reference artifact"):
            workspace.preserve_revision_references(source, baseline)


# --- source evidence --------------------------------------------------------------------


def test_source_evidence_is_snapshotted_and_seeded_at_original_relative_paths(tmp_path: Path) -> None:
    # a snapshot preserves nested source evidence
    paper = _paper(tmp_path / "snapshot")
    snapshot = tmp_path / "snapshot" / "snapshot"

    assert workspace.snapshot_previous_paper(paper, snapshot) == []

    relative = Path("source_evidence/winner/main.py")
    assert (snapshot / relative).read_bytes() == (paper / relative).read_bytes()

    # a revision seeds readable sources at their original relative paths
    paper = _paper(tmp_path / "seed")
    revision = tmp_path / "seed" / "revision"
    revision.mkdir()
    snapshot = revision / "previous_paper"
    workspace.snapshot_previous_paper(paper, snapshot)

    assert workspace.seed_revision_source_evidence(snapshot, revision) == []

    seeded_relative = (paper / "paper.tex").read_text().strip()
    assert (revision / seeded_relative).read_bytes() == (paper / seeded_relative).read_bytes()

    # an older paper without source evidence is accepted
    paper = tmp_path / "older" / "paper"
    paper.mkdir(parents=True)
    revision = tmp_path / "older" / "revision"
    revision.mkdir()

    assert workspace.seed_revision_source_evidence(paper, revision) == []
    assert not (revision / "source_evidence").exists()


def test_source_evidence_skips_links_and_special_files(tmp_path: Path) -> None:
    for operation in ("snapshot", "seed"):
        case_dir = tmp_path / operation
        paper = _paper(case_dir)
        source = paper / "source_evidence"
        target = case_dir / "outside.py"
        target.write_text("not copied")
        (source / "linked.py").symlink_to(target)
        (source / "linked_directory").symlink_to(case_dir, target_is_directory=True)
        os.mkfifo(source / "pipe")
        destination = case_dir / "destination"
        if operation == "seed":
            destination.mkdir()
        skipped = _copy_source_evidence(operation, paper, destination)

        assert {row["reason"] for row in skipped} == {"symlink", "named_pipe"}, operation
        assert sorted(path.name for path in (destination / "source_evidence").iterdir()) == [
            "winner"
        ], operation


def test_source_evidence_limits_remove_only_owned_partial_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for operation in ("snapshot", "seed"):
        for limit in ("MAX_SNAPSHOT_BYTES", "MAX_SNAPSHOT_FILES"):
            case_dir = tmp_path / operation / limit
            paper = _paper(case_dir)
            destination = case_dir / "destination"
            (paper / "paper.tex").write_text("")
            allowed = 1 if operation == "snapshot" and limit == "MAX_SNAPSHOT_FILES" else 0
            if operation == "seed":
                destination.mkdir()
                (destination / "unrelated").write_text("keep")
            with monkeypatch.context() as m:
                m.setattr(workspace, limit, allowed)
                with pytest.raises(ValueError, match="too (large|many files)"):
                    _copy_source_evidence(operation, paper, destination)

            assert not (destination / "source_evidence").exists(), (operation, limit)
            if operation == "seed":
                assert (destination / "unrelated").read_text() == "keep", limit
            assert (paper / "source_evidence/winner/main.py").exists(), (operation, limit)


def test_destination_conflicts_and_linked_destinations_preserve_existing_content(
    tmp_path: Path,
) -> None:
    # an existing destination is never overwritten
    for operation in ("snapshot", "seed"):
        case_dir = tmp_path / operation
        paper = _paper(case_dir)
        destination = case_dir / "destination"
        conflict = destination if operation == "snapshot" else destination / "source_evidence"
        conflict.mkdir(parents=True)
        (conflict / "existing").write_text("keep")

        with pytest.raises(FileExistsError):
            _copy_source_evidence(operation, paper, destination)

        assert (conflict / "existing").read_text() == "keep", operation

    # a symlinked source_evidence destination is rejected without writing through it
    case_dir = tmp_path / "linked"
    paper = _paper(case_dir)
    destination = case_dir / "destination"
    destination.mkdir()
    outside = case_dir / "outside"
    outside.mkdir()
    (destination / "source_evidence").symlink_to(outside, target_is_directory=True)

    with pytest.raises(FileExistsError):
        workspace.seed_revision_source_evidence(paper, destination)

    assert not list(outside.iterdir())

    # a symlinked destination ancestor is rejected
    case_dir = tmp_path / "ancestor"
    paper = _paper(case_dir)
    outside = case_dir / "outside"
    outside.mkdir()
    linked = case_dir / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        workspace.seed_revision_source_evidence(paper, linked)

    assert not list(outside.iterdir())


def test_revision_seed_reports_invalid_source_root(tmp_path: Path) -> None:
    reasons = {"file": "not_directory", "symlink": "symlink", "pipe": "named_pipe"}
    for kind in ("file", "symlink", "pipe"):
        case_dir = tmp_path / kind
        paper = case_dir / "paper"
        paper.mkdir(parents=True)
        source = paper / "source_evidence"
        if kind == "file":
            source.write_text("not a directory")
        elif kind == "symlink":
            source.symlink_to(case_dir, target_is_directory=True)
        else:
            os.mkfifo(source)
        revision = case_dir / "revision"
        revision.mkdir()

        skipped = workspace.seed_revision_source_evidence(paper, revision)

        assert skipped == [{"path": str(source), "reason": reasons[kind]}], kind
        assert not list(revision.iterdir()), kind


def test_source_link_swap_cannot_escape_pinned_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper = _paper(tmp_path)
    revision = tmp_path / "revision"
    revision.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("must not copy")
    original_open = os.open

    def swap_source(path, flags, mode=0o777, *, dir_fd=None):
        if path == "main.py" and not flags & os.O_WRONLY:
            source = paper / "source_evidence/winner/main.py"
            source.unlink()
            source.symlink_to(outside)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_source)
    with pytest.raises(OSError):
        workspace.seed_revision_source_evidence(paper, revision)

    assert not (revision / "source_evidence").exists()
    assert outside.read_text() == "must not copy"


# --- posthoc inventory preservation -----------------------------------------------------


def _source_data() -> dict:
    return {
        "posthoc_analyses": [
            {"analysis_id": "family", "rank": 1, "raw_data": {"gain": 0.1}},
            {"analysis_id": "audit", "rank": 2, "summary_text": "Observed limits"},
        ],
        "totals": {"posthoc_analyses_total": 2, "posthoc_analyses_ranked": 2},
        "grounding_sources": {"posthoc.family.data.gain": {"value": 0.1}},
    }


def _write_data(directory: Path, data: object) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "paper_data.json").write_text(json.dumps(data))


def _rerank(data: dict, ids: tuple[str, ...] = ("audit", "family")) -> dict:
    ranking = {
        "ranking_rationale": "Revised emphasis",
        "ranked_analyses": [
            {
                "analysis_id": analysis_id,
                "rank": rank,
                "relevance_to_thesis": "high",
                "recommended_usage": "supporting",
                "relevance_justification": "Relevant evidence",
                "key_finding_for_paper": "Observed result",
            }
            for rank, analysis_id in enumerate(ids, 1)
        ],
    }
    return augment_paper_data_with_posthoc(
        data, ranking, [{"analysis_id": analysis_id} for analysis_id in ids]
    )


def test_paper_edit_preparation_preserves_sources_context_and_posthoc_inventory(
    tmp_path: Path,
) -> None:
    # sources are seeded, the previous paper snapshotted, and the context records it
    sources = tmp_path / "sources"
    paper = _paper(sources)
    revision = sources / "revision"
    revision.mkdir()

    result = _run_paper_edit_preparation(paper, revision)

    assert result.returncode == 0, result.stderr
    relative = (revision / "paper.tex").read_text().strip()
    assert (revision / relative).read_bytes() == (paper / relative).read_bytes()
    assert (revision / "previous_paper" / relative).read_bytes() == (paper / relative).read_bytes()
    context = json.loads((revision / "paper_edit_context.json").read_text())
    assert "source_evidence" in context["seeded_workspace_artifacts"]

    # the posthoc inventory is preserved before augmentation
    posthoc = tmp_path / "posthoc"
    source = posthoc / "source"
    candidate = posthoc / "candidate"
    original = _source_data()
    _write_data(source, original)
    _write_data(candidate, {"totals": {}, "grounding_sources": {}})

    process = _run_paper_edit_preparation(source, candidate)

    assert process.returncode == 0, process.stderr
    prepared = json.loads((candidate / "paper_data.json").read_text())
    _write_data(candidate, _rerank(prepared))
    _validate_semantic_evidence(source, candidate)
    assert json.loads((source / "paper_data.json").read_text()) == original


def test_shell_preparation_rejects_invalid_original_data(tmp_path: Path) -> None:
    for kind in ("symlink", "pipe", "directory", "json"):
        case_dir = tmp_path / kind
        source = case_dir / "source"
        source.mkdir(parents=True)
        candidate = case_dir / "candidate"
        _write_data(candidate, {"totals": {}, "grounding_sources": {}})
        data_path = source / "paper_data.json"
        if kind == "symlink":
            data_path.symlink_to(candidate / "paper_data.json")
        elif kind == "pipe":
            os.mkfifo(data_path)
        elif kind == "directory":
            data_path.mkdir()
        else:
            data_path.write_text("{broken")

        process = _run_paper_edit_preparation(source, candidate)

        assert process.returncode != 0, kind
        assert "revision posthoc" in process.stderr, kind
        assert json.loads((candidate / "paper_data.json").read_text()) == {
            "totals": {},
            "grounding_sources": {},
        }, kind


def test_preserve_revision_posthoc_retains_inventory_metadata_and_new_analyses(
    tmp_path: Path,
) -> None:
    # the inventory survives a reversed ranking and the rebuilt data is not mutated
    reversed_dir = tmp_path / "reversed"
    source = reversed_dir / "source"
    candidate = reversed_dir / "candidate"
    original = _source_data()
    _write_data(source, original)
    rebuilt: dict = {"totals": {}, "grounding_sources": {}}

    restored = workspace.preserve_revision_posthoc(source, rebuilt)
    result = _rerank(restored)
    _write_data(candidate, result)

    assert result["posthoc_analyses"] == original["posthoc_analyses"]
    _validate_semantic_evidence(source, candidate)
    assert rebuilt == {"totals": {}, "grounding_sources": {}}

    # original metadata wins while unrelated rebuilt data is kept
    metadata = tmp_path / "metadata"
    original = _source_data()
    _write_data(metadata, original)
    rebuilt = {
        "totals": {"evaluated_programs": 4, "posthoc_analyses_total": 0},
        "grounding_sources": {
            "result.best": {"value": 0.7},
            "posthoc.family.data.gain": {"value": 0.9},
        },
        "final_evaluation": {"score": 0.7},
        "posthoc_analyses": [
            {"analysis_id": "family", "rank": 2, "raw_data": {"gain": 9.0}},
            {"analysis_id": "extra", "rank": 3},
        ],
    }

    result = workspace.preserve_revision_posthoc(metadata, rebuilt)

    assert result["totals"] == {"evaluated_programs": 4, **original["totals"]}
    assert result["grounding_sources"] == {
        "result.best": {"value": 0.7},
        **original["grounding_sources"],
    }
    assert result["final_evaluation"] == rebuilt["final_evaluation"]
    assert result["posthoc_analyses"] == original["posthoc_analyses"] + [
        {"analysis_id": "extra", "rank": 3}
    ]

    # a newly discovered analysis is allowed after the preserved ones
    discovered = tmp_path / "discovered"
    original = _source_data()
    _write_data(discovered, original)

    result = _rerank(workspace.preserve_revision_posthoc(discovered, {}), ("audit", "family", "new"))

    assert result["posthoc_analyses"][:2] == original["posthoc_analyses"]
    assert result["posthoc_analyses"][2]["analysis_id"] == "new"

    # a preserved inventory still rejects a later scientific change
    later = tmp_path / "later"
    source = later / "source"
    candidate = later / "candidate"
    _write_data(source, _source_data())
    data = deepcopy(workspace.preserve_revision_posthoc(source, {}))
    data["posthoc_analyses"][0]["raw_data"]["gain"] = 9.0
    _write_data(candidate, data)

    with pytest.raises(ValueError, match="posthoc inventory"):
        _validate_semantic_evidence(source, candidate)


LEGACY_SOURCES: list[dict[str, Any] | None] = [None, {}, {"posthoc_analyses": []}]
MALFORMED_SOURCES: list[Any] = [
    [],
    {"posthoc_analyses": None},
    {"posthoc_analyses": ["bad"]},
    {"posthoc_analyses": [{}]},
    {"posthoc_analyses": [{"analysis_id": "a"}, {"analysis_id": "a"}]},
    {"posthoc_analyses": [{"analysis_id": "a", "rank": True}]},
    {"posthoc_analyses": [], "totals": []},
    {"posthoc_analyses": [], "grounding_sources": []},
]


def test_preservation_accepts_legacy_paper_and_rejects_malformed_source(tmp_path: Path) -> None:
    # legacy papers without an inventory are accepted
    for index, source in enumerate(LEGACY_SOURCES):
        case_dir = tmp_path / f"legacy{index}"
        case_dir.mkdir()
        if source is not None:
            _write_data(case_dir, source)
        rebuilt = {"totals": {"evaluated_programs": 1}}

        result = workspace.preserve_revision_posthoc(case_dir, rebuilt)

        assert result["totals"] == rebuilt["totals"], source
        assert not result.get("posthoc_analyses"), source

    # malformed sources are rejected
    for index, source in enumerate(MALFORMED_SOURCES):
        case_dir = tmp_path / f"malformed{index}"
        _write_data(case_dir, source)

        with pytest.raises(ValueError, match="revision posthoc"):
            workspace.preserve_revision_posthoc(case_dir, {})


def test_preservation_rejects_unsafe_source(tmp_path: Path, monkeypatch) -> None:
    for kind in ("symlink", "pipe", "directory", "json", "oversize"):
        case_dir = tmp_path / kind
        case_dir.mkdir()
        source = case_dir / "paper_data.json"
        with monkeypatch.context() as m:
            if kind == "symlink":
                source.symlink_to(case_dir / "missing")
            elif kind == "pipe":
                os.mkfifo(source)
            elif kind == "directory":
                source.mkdir()
            elif kind == "json":
                source.write_text("{broken")
            else:
                source.write_text(json.dumps(_source_data()))
                m.setattr(workspace, "MAX_PAPER_DATA_BYTES", 1)

            with pytest.raises(ValueError, match="revision posthoc"):
                workspace.preserve_revision_posthoc(case_dir, {})
