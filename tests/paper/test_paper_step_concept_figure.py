#!/usr/bin/env python3
"""Concept-figure request resolution, artifact publication/provenance, and manifest validation."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.orchestrator import paper_step
from src.orchestrator.figure_manifest_schema import normalize_figure_manifest
from tests.paper_step_support import (
    diagram_row,
    png_chunk,
    read_json,
    write_json,
    write_manifest,
    write_png,
    write_strategy,
    write_valid_png,
)

CONCEPT = "generate-concept-figure"
PANEL_PLAN = '{"panels": [{"id": "overview"}]}'
PROMPT_LOG = "figures/overview.prompts.json"


def _sub(root: Path, name: str) -> Path:
    """Fresh isolated workspace for one section of a merged test."""
    path = root / name
    path.mkdir()
    return path


def _concept_spec(**fields: Any) -> dict[str, Any]:
    return {"figure_id": "concept_overview", "preferred_generator": CONCEPT, **fields}


def _validate(root: Path, tex_text: str = r"\includegraphics{figures/overview}") -> dict[str, Any]:
    return paper_step.validate_figure_manifest(root, tex_text=tex_text, allowed_source_ids=set())


def _publish(paper_dir: Path, render_dir: Path) -> None:
    paper_step.publish_concept_figure_artifacts(paper_dir, render_dir, "overview")


def _manifest_row(root: Path) -> dict[str, Any]:
    return read_json(root / "figure_manifest.json")["figures"][0]


def _update_json(path: Path, mutate: Callable[[Any], None]) -> None:
    payload = read_json(path)
    mutate(payload)
    write_json(path, payload)


# --------------------------------------------------------------------------------------------
# resolve_concept_figure_request
# --------------------------------------------------------------------------------------------


def _strategy(root: Path, specs: list[dict[str, Any]]) -> Path:
    write_strategy(root, specs, status="ready")
    return root


def _resolve(root: Path) -> dict[str, Any] | None:
    return paper_step.resolve_concept_figure_request(root)


def test_resolve_concept_figure_request_returns_none_without_active_spec(tmp_path: Path) -> None:
    # returns None when strategy missing
    assert _resolve(_sub(tmp_path, "missing")) is None
    # returns None without an active concept spec
    cases: list[tuple[str, list[dict[str, Any]]]] = [
        (
            "no_concept_figure_spec",
            [
                {
                    "figure_id": "main_comparison",
                    "preferred_generator": "matplotlib",
                    "preferred_span": "full_width",
                    "purpose": "bar chart",
                }
            ],
        ),
        (
            "concept_figure_explicitly_removed",
            [_concept_spec(required=False, evidence_audit_disposition="Removed from the manuscript.")],
        ),
        (
            "blank_figure_id",
            [{"figure_id": "", "preferred_generator": CONCEPT, "preferred_span": "full_width",
              "purpose": "no id"}],
        ),
    ]
    for name, specs in cases:
        assert _resolve(_strategy(_sub(tmp_path, name), specs)) is None, name


def test_resolve_concept_figure_request_resolves_active_spec(tmp_path: Path) -> None:
    # resolves full-width concept figure
    root = _strategy(
        _sub(tmp_path, "full_width"),
        [
            {"figure_id": "main_comparison", "preferred_generator": "matplotlib"},
            _concept_spec(preferred_span="full_width", purpose="Show the dual-stream architecture."),
        ],
    )
    assert _resolve(root) == {
        "figure_id": "concept_overview",
        "output_rel": "figures/concept_overview.png",
        "instructions": "Show the dual-stream architecture.",
        "placement": "full-width",
    }
    # request fields: non-full span maps to half-width; blank purpose gets default instruction
    field_cases: list[tuple[str, dict[str, Any], str, str]] = [
        (
            "non_full_span_maps_to_half_width",
            {"preferred_span": "column", "purpose": "Compact schematic."},
            "placement",
            "half-width",
        ),
        (
            "default_instruction_when_purpose_blank",
            {"preferred_span": "full_width", "purpose": "   "},
            "instructions",
            "Overview diagram for the figure 'concept_overview'.",
        ),
    ]
    for name, spec_fields, key, expected in field_cases:
        request = _resolve(_strategy(_sub(tmp_path, name), [_concept_spec(**spec_fields)]))
        assert request is not None, name
        assert request[key] == expected, name
    # allows removed concept spec with active replacement
    root = _strategy(
        _sub(tmp_path, "removed_with_replacement"),
        [
            _concept_spec(figure_id="removed_overview", required=False),
            _concept_spec(figure_id="active_overview", required=True),
        ],
    )
    write_manifest(root, [])
    request = _resolve(root)
    report = _validate(root, tex_text="")
    assert request is not None
    assert request["figure_id"] == "active_overview"
    assert (
        "paper_strategy.json may contain at most one generate-concept-figure spec"
        not in report["errors"]
    )
    # accepts slug figure id at maximum length
    figure_id = "a" + "b" * 125 + "-1"
    root = _strategy(
        _sub(tmp_path, "max_length"),
        [_concept_spec(figure_id=figure_id, preferred_span="full_width", purpose="Safe identifier.")],
    )
    request = _resolve(root)
    assert request is not None
    assert request["output_rel"] == f"figures/{figure_id}.png"


def test_resolve_concept_figure_request_rejects_invalid_specs(tmp_path: Path) -> None:
    # rejects multiple concept figure specs
    root = _strategy(_sub(tmp_path, "multiple"), [_concept_spec(), _concept_spec(figure_id="concept_details")])
    with pytest.raises(RuntimeError, match="at most one generate-concept-figure spec"):
        _resolve(root)
    # rejects concept spec with unsafe figure id
    unsafe_ids = [
        "../outside", "overview; touch owned", "overview\tother", "overview\nother", "-leading-hyphen",
        "x" * 129,
    ]
    for index, figure_id in enumerate(unsafe_ids):
        root = _strategy(
            _sub(tmp_path, f"unsafe_{index}"),
            [
                _concept_spec(
                    figure_id=figure_id,
                    preferred_span="full_width",
                    purpose="Unsafe identifier must never reach the shell.",
                )
            ],
        )
        with pytest.raises(RuntimeError, match="invalid figure_id"):
            _resolve(root)


# --------------------------------------------------------------------------------------------
# Audit-repair fallback promotion
# --------------------------------------------------------------------------------------------


def _concept_dirs(root: Path) -> tuple[Path, Path]:
    paper_dir = root / "paper"
    render_dir = root / "render"
    (paper_dir / "figures").mkdir(parents=True)
    (render_dir / "figures").mkdir(parents=True)
    (render_dir / "figures" / "overview.prompts.json").write_text(PANEL_PLAN, encoding="utf-8")
    return paper_dir, render_dir


def _repair_dir(root: Path) -> Path:
    paper_dir, _render_dir = _concept_dirs(root)
    figures = paper_dir / "figures"
    write_valid_png(figures / "overview.png", width=2, height=2)
    (figures / "overview.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (figures / "overview.prompts.json").write_text(PANEL_PLAN, encoding="utf-8")
    (paper_dir / "paper_evidence_audit_feedback.json").write_text(
        '{"verdict": "fail", "ok": false, "findings": '
        '[{"id": "method-figure-equation", "evidence": "figures/overview.png"}]}',
        encoding="utf-8",
    )
    write_manifest(
        paper_dir,
        [
            diagram_row(
                generator="manual",
                attempted_generator=CONCEPT,
                prompt_log_path=PROMPT_LOG,
                fallback_reason="Corrected equation after evidence audit.",
            )
        ],
    )
    return paper_dir


def _authorize(paper_dir: Path) -> tuple[str, list[str]]:
    return paper_step.authorize_concept_figure_repair_feedback(paper_dir, "overview")


def _promote(paper_dir: Path, fingerprint: str, feedback_sha256: str, finding_ids: Any) -> None:
    paper_step.promote_concept_figure_repair_fallback(
        paper_dir, "overview", fingerprint, feedback_sha256, finding_ids
    )


def _matches(paper_dir: Path, fingerprint: str, feedback_sha256: str, finding_ids: Any) -> bool:
    return paper_step.concept_figure_repair_fallback_matches(
        paper_dir, "overview", fingerprint, feedback_sha256, finding_ids
    )


def _mutate_fallback_reason(manifest: dict[str, Any]) -> None:
    manifest["figures"][0]["fallback_reason"] = "Changed reason."


def test_audit_repair_fallback_promotion_binds_request_provenance(tmp_path: Path) -> None:
    # promotes audit repair fallback with request-bound provenance
    repair_dir = _repair_dir(_sub(tmp_path, "promote"))
    feedback_sha256, finding_ids = _authorize(repair_dir)
    _promote(repair_dir, "a" * 64, feedback_sha256, finding_ids)
    assert _matches(repair_dir, "a" * 64, feedback_sha256, finding_ids)
    assert not _matches(repair_dir, "b" * 64, feedback_sha256, finding_ids)
    # promoted fallback is bound to the pdf, prompt log and manifest
    mutations: list[tuple[str, Callable[[Path], object]]] = [
        ("pdf", lambda d: (d / "figures" / "overview.pdf").write_bytes(b"%PDF-1.4\nchanged\n%%EOF\n")),
        (
            "prompt",
            lambda d: (d / "figures" / "overview.prompts.json").write_text(
                '{"panels": [{"id": "changed"}]}', encoding="utf-8"
            ),
        ),
        ("manifest", lambda d: _update_json(d / "figure_manifest.json", _mutate_fallback_reason)),
    ]
    for name, mutation in mutations:
        repair_dir = _repair_dir(_sub(tmp_path, name))
        feedback_sha256, finding_ids = _authorize(repair_dir)
        _promote(repair_dir, "a" * 64, feedback_sha256, finding_ids)
        mutation(repair_dir)
        assert not _matches(repair_dir, "a" * 64, feedback_sha256, finding_ids), name


def test_audit_repair_fallback_rejects_incomplete_or_changed_authorization(tmp_path: Path) -> None:
    # rejects fallback without attempted_generator
    repair_dir = _repair_dir(_sub(tmp_path, "no_attempted_generator"))
    _update_json(
        repair_dir / "figure_manifest.json",
        lambda manifest: manifest["figures"][0].pop("attempted_generator"),
    )
    feedback_sha256, finding_ids = _authorize(repair_dir)
    with pytest.raises(RuntimeError, match="Incomplete concept figure fallback"):
        _promote(repair_dir, "a" * 64, feedback_sha256, finding_ids)
    assert not (repair_dir / "figures" / "overview.png.prov.json").exists()
    # rejects authorization for unrelated finding
    repair_dir = _repair_dir(_sub(tmp_path, "unrelated_finding"))
    (repair_dir / "paper_evidence_audit_feedback.json").write_text(
        '{"verdict": "fail", "ok": false, "findings": '
        '[{"id": "claim-value", "issue": "Wrong number in Table 2"}]}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="did not identify concept figure"):
        _authorize(repair_dir)
    # rejects changed audit feedback before promotion
    repair_dir = _repair_dir(_sub(tmp_path, "changed_feedback"))
    feedback_sha256, finding_ids = _authorize(repair_dir)
    _update_json(
        repair_dir / "paper_evidence_audit_feedback.json",
        lambda feedback: feedback.__setitem__("summary", "mutated"),
    )
    with pytest.raises(RuntimeError, match="authorization changed"):
        _promote(repair_dir, "a" * 64, feedback_sha256, finding_ids)


# --------------------------------------------------------------------------------------------
# Atomic writes, transaction journal, staged PNG validation
# --------------------------------------------------------------------------------------------


def test_durable_writes_fsync_file_before_replace_and_parent_after(tmp_path: Path) -> None:
    writes: list[tuple[str, Callable[[Path], None]]] = [
        (
            "atomic_json_write",
            lambda root: paper_step._write_json_atomically(root / "manifest.json", {"status": "ready"}),
        ),
        ("marker_write", paper_step.publish_concept_figure_marker),
    ]
    real_replace = os.replace
    for name, write in writes:
        root = _sub(tmp_path, name)
        events: list[str] = []

        def record_fsync(descriptor: int, events: list[str] = events) -> None:
            mode = os.fstat(descriptor).st_mode
            events.append("fsync-directory" if stat.S_ISDIR(mode) else "fsync-file")

        def record_replace(*args: Any, events: list[str] = events, **kwargs: Any) -> None:
            events.append("replace")
            real_replace(*args, **kwargs)

        with (
            patch.object(paper_step.os, "fsync", side_effect=record_fsync),
            patch.object(paper_step.os, "replace", side_effect=record_replace),
        ):
            write(root)

        assert events == ["fsync-file", "replace", "fsync-directory"], name


def test_accepts_legacy_two_artifact_transaction_journals() -> None:
    token = "0123456789abcdef"
    for phase in ("prepared", "backed_up", "published", "committed"):
        records = [
            {
                "destination": destination,
                "temp": f".{destination}.{token}.new",
                "backup": f".{destination}.{token}.old",
                "existed": False,
                "identity": None,
                "temp_identity": [1, 2, stat.S_IFREG],
                "digest": "0" * 64,
                "size": 1,
            }
            for destination in ("overview.png", "overview.prompts.json")
        ]
        payload = {"phase": phase, "token": token, "records": records}
        assert paper_step._validate_concept_transaction(payload) is payload, phase


def _oversized_dimensions(png_path: Path) -> None:
    write_valid_png(png_path, width=2, height=2)
    data = bytearray(png_path.read_bytes())
    data[16:24] = struct.pack(">II", 100_000, 100_000)
    data[29:33] = struct.pack(">I", zlib.crc32(data[12:29]) & 0xFFFFFFFF)
    png_path.write_bytes(data)


def _unknown_critical_chunk(png_path: Path) -> None:
    write_valid_png(png_path, width=2, height=2)
    data = png_path.read_bytes()
    png_path.write_bytes(data[:-12] + png_chunk(b"ABCD", b"unknown") + data[-12:])


def _bad_prompt_log(payload: str) -> Callable[[Path], None]:
    def stage(png_path: Path) -> None:
        write_valid_png(png_path, width=2, height=2)
        png_path.with_name("overview.prompts.json").write_text(payload, encoding="utf-8")

    return stage


UNSAFE_STAGED_CASES: list[tuple[str, Callable[[Path], None], str]] = [
    ("truncated_png", lambda p: write_png(p, width=1400, height=800), "staged concept figure PNG"),
    ("oversized_png", _oversized_dimensions, "Unsafe staged concept figure PNG"),
    (
        "invalid_scanline_filter",
        lambda p: write_valid_png(p, width=2, height=2, filter_byte=255),
        "PNG filter",
    ),
    ("unknown_critical_chunk", _unknown_critical_chunk, "Unknown critical"),
    ("no_panel_plan", _bad_prompt_log('{"anything": true}'), "prompt log"),
    (
        "duplicate_panel_ids",
        _bad_prompt_log('{"panels": [{"id": "same"}, {"id": "same"}]}'),
        "prompt log",
    ),
]


def test_rejects_unsafe_staged_concept_artifacts(tmp_path: Path) -> None:
    for name, stage, match in UNSAFE_STAGED_CASES:
        paper_dir, render_dir = _concept_dirs(_sub(tmp_path, name))
        stage(render_dir / "figures" / "overview.png")
        with pytest.raises(RuntimeError, match=match):
            _publish(paper_dir, render_dir)
        assert not (paper_dir / "figures" / "overview.png").exists(), name


# --------------------------------------------------------------------------------------------
# publish_concept_figure_artifacts / recovery / rollback
# --------------------------------------------------------------------------------------------


def _stage_valid(render_dir: Path, *, width: int = 2, height: int = 2) -> Path:
    png_path = render_dir / "figures" / "overview.png"
    write_valid_png(png_path, width=width, height=height)
    return png_path


def _write_live(paper_dir: Path, png: bytes = b"original-png", prompts: str = '{"original": true}'):
    (paper_dir / "figures" / "overview.png").write_bytes(png)
    (paper_dir / "figures" / "overview.prompts.json").write_text(prompts, encoding="utf-8")


def _truncate_journal(paper_dir: Path, *, phase: str | None = None) -> Path:
    journal_path = paper_dir / "figures" / ".concept-figure-transaction.json"
    journal = read_json(journal_path)
    journal["records"] = journal["records"][:2]
    if phase is not None:
        journal["phase"] = phase
    write_json(journal_path, journal)
    (paper_dir / "figures" / "overview.png.prov.json").unlink()
    return journal_path


def _assert_live_unchanged(paper_dir: Path) -> None:
    assert (paper_dir / "figures" / "overview.png").read_bytes() == b"original-png"
    assert (paper_dir / "figures" / "overview.prompts.json").read_text(encoding="utf-8") == (
        '{"original": true}'
    )


def test_publish_concept_figure_artifacts_records_provenance_and_guards_symlinks(
    tmp_path: Path,
) -> None:
    # publishes provenance for rendered concept figure
    paper_dir, render_dir = _concept_dirs(_sub(tmp_path, "provenance"))
    (paper_dir / "paper.tex").write_text("Rendered paper input.\n", encoding="utf-8")
    _stage_valid(render_dir)
    _publish(paper_dir, render_dir)
    png_path = paper_dir / "figures" / "overview.png"
    provenance = read_json(paper_dir / "figures" / "overview.png.prov.json")
    assert provenance["artifact"]["sha256"] == hashlib.sha256(png_path.read_bytes()).hexdigest()
    assert provenance["derived_from"] == [
        "@evidence[[paper.tex]]",
        "@evidence[[figures/overview.prompts.json]]",
    ]
    assert not provenance["named_outputs"]["placeholder"]
    assert paper_step.concept_figure_provenance_matches(paper_dir, "overview")
    # publishes staged artifacts without following destination symlink
    base = _sub(tmp_path, "dest_symlink")
    paper_dir, render_dir = _concept_dirs(base)
    _stage_valid(render_dir, width=1400, height=800)
    sentinel = base / "sentinel.png"
    sentinel.write_bytes(b"outside")
    (paper_dir / "figures" / "overview.png").symlink_to(sentinel)
    _publish(paper_dir, render_dir)
    assert sentinel.read_bytes() == b"outside"
    assert not (paper_dir / "figures" / "overview.png").is_symlink()
    assert paper_step.read_png_size(paper_dir / "figures" / "overview.png") == (1400, 800)
    assert read_json(paper_dir / "figures" / "overview.prompts.json") == json.loads(PANEL_PLAN)
    # rejects staged symlink without changing live artifacts
    base = _sub(tmp_path, "staged_symlink")
    paper_dir, render_dir = _concept_dirs(base)
    _write_live(paper_dir)
    sentinel = base / "sentinel.png"
    write_valid_png(sentinel, width=1400, height=800)
    (render_dir / "figures" / "overview.png").symlink_to(sentinel)
    with pytest.raises(RuntimeError, match="staged concept figure PNG"):
        _publish(paper_dir, render_dir)
    _assert_live_unchanged(paper_dir)


def test_recover_concept_figure_artifact_publication_finishes_legacy_transactions(
    tmp_path: Path,
) -> None:
    # recovers legacy published transaction by finishing provenance
    paper_dir, render_dir = _concept_dirs(_sub(tmp_path, "published"))
    live_png = paper_dir / "figures" / "overview.png"
    live_prompts = paper_dir / "figures" / "overview.prompts.json"
    _write_live(paper_dir, b"old-png", '{"old": true}')
    new_png = _stage_valid(render_dir).read_bytes()
    write_manifest(
        paper_dir,
        [
            {
                "figure_id": "overview",
                "generator": None,
                "prompt_log_path": PROMPT_LOG,
                "path_pdf": "figures/overview.pdf",
            }
        ],
    )
    (paper_dir / "figures" / "overview.pdf").write_bytes(b"stale-pdf")
    _publish(paper_dir, render_dir)
    assert live_png.read_bytes() != b"old-png"
    journal_path = _truncate_journal(paper_dir)
    paper_step.recover_concept_figure_artifact_publication(paper_dir)
    assert live_png.read_bytes() == new_png
    assert read_json(live_prompts) == {"panels": [{"id": "overview"}]}
    assert _manifest_row(paper_dir)["generator"] == CONCEPT
    assert (paper_dir / ".concept_figure_complete").is_file()
    assert paper_step.concept_figure_provenance_matches(paper_dir, "overview")
    assert not (paper_dir / "figures" / "overview.pdf").exists()
    assert not journal_path.exists()
    # recovers legacy committed transaction by finishing provenance
    paper_dir, render_dir = _concept_dirs(_sub(tmp_path, "committed"))
    _stage_valid(render_dir)
    _publish(paper_dir, render_dir)
    journal_path = _truncate_journal(paper_dir, phase="committed")
    paper_step.recover_concept_figure_artifact_publication(paper_dir)
    assert paper_step.concept_figure_provenance_matches(paper_dir, "overview")
    assert not journal_path.exists()


def test_recover_concept_figure_artifact_publication_rejects_modified_artifacts(
    tmp_path: Path,
) -> None:
    # legacy recovery does not backfill provenance before commit
    paper_dir, render_dir = _concept_dirs(_sub(tmp_path, "no_backfill"))
    _stage_valid(render_dir)
    _publish(paper_dir, render_dir)
    _truncate_journal(paper_dir)
    provenance_path = paper_dir / "figures" / "overview.png.prov.json"
    with (
        patch.object(
            paper_step, "record_concept_figure_success", side_effect=RuntimeError("manifest failed")
        ),
        pytest.raises(RuntimeError, match="manifest failed"),
    ):
        paper_step.recover_concept_figure_artifact_publication(paper_dir)
    assert not provenance_path.exists()
    paper_step.rollback_concept_figure_artifact_publication(paper_dir)
    assert not provenance_path.exists()
    # rejects modified png in legacy committed transaction
    paper_dir, render_dir = _concept_dirs(_sub(tmp_path, "modified_committed"))
    _stage_valid(render_dir)
    _publish(paper_dir, render_dir)
    journal_path = _truncate_journal(paper_dir, phase="committed")
    write_valid_png(paper_dir / "figures" / "overview.png", width=3, height=3)
    with pytest.raises(RuntimeError, match="transaction was modified"):
        paper_step.recover_concept_figure_artifact_publication(paper_dir)
    assert not (paper_dir / "figures" / "overview.png.prov.json").exists()
    assert journal_path.exists()
    # rejects in-place mutation of published artifact
    paper_dir, render_dir = _concept_dirs(_sub(tmp_path, "in_place_mutation"))
    _stage_valid(render_dir)
    _publish(paper_dir, render_dir)
    with (paper_dir / "figures" / "overview.png").open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        handle.write(b"X")
    with pytest.raises(RuntimeError, match="transaction was modified"):
        paper_step.recover_concept_figure_artifact_publication(paper_dir)
    assert (paper_dir / "figures" / ".concept-figure-transaction.json").is_file()
    # removes orphaned pre-journal temp files
    root = _sub(tmp_path, "orphans")
    figures = root / "figures"
    figures.mkdir()
    orphans = [
        figures / ".overview.png.0123456789abcdef.new",
        figures / ".overview.png.prov.json.0123456789abcdef.new",
    ]
    for orphan in orphans:
        orphan.write_bytes(b"orphan")
    paper_step.recover_concept_figure_artifact_publication(root)
    assert not any(orphan.exists() for orphan in orphans)


def test_publish_concept_figure_artifacts_rolls_back_on_injected_failures(tmp_path: Path) -> None:
    real_link = os.link
    real_replace = os.replace

    # rolls back both artifacts when second publish fails
    paper_dir, render_dir = _concept_dirs(_sub(tmp_path, "second_publish"))
    _stage_valid(render_dir, width=1400, height=800)
    _write_live(paper_dir)
    prompt_publish_failed = False

    def fail_prompt_publish(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal prompt_publish_failed
        if (
            not prompt_publish_failed
            and destination == "overview.prompts.json"
            and str(source).endswith(".new")
        ):
            prompt_publish_failed = True
            raise OSError("injected prompt publish failure")
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    with (
        patch("src.orchestrator.paper_step.os.link", side_effect=fail_prompt_publish),
        pytest.raises(RuntimeError, match="publish concept figure artifacts"),
    ):
        _publish(paper_dir, render_dir)
    _assert_live_unchanged(paper_dir)

    # preserves concurrent destination replacement
    paper_dir, render_dir = _concept_dirs(_sub(tmp_path, "concurrent_replacement"))
    _stage_valid(render_dir)
    _write_live(paper_dir)
    (paper_dir / "figures" / ".concurrent.png").write_bytes(b"concurrent-png")
    replace_injected = False

    def replace_after_concurrent_write(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal replace_injected
        if not replace_injected and source == "overview.png" and str(destination).endswith(".old"):
            replace_injected = True
            real_replace(".concurrent.png", "overview.png", src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
        real_replace(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    with (
        patch("src.orchestrator.paper_step.os.replace", side_effect=replace_after_concurrent_write),
        pytest.raises(RuntimeError, match="publish concept figure artifacts"),
    ):
        _publish(paper_dir, render_dir)
    assert (paper_dir / "figures" / "overview.png").read_bytes() == b"concurrent-png"
    assert (paper_dir / "figures" / "overview.prompts.json").read_text(encoding="utf-8") == (
        '{"original": true}'
    )
    assert not (paper_dir / "figures" / ".concept-figure-transaction.json").exists()

    # does not overwrite concurrently created destination
    paper_dir, render_dir = _concept_dirs(_sub(tmp_path, "concurrent_creation"))
    _stage_valid(render_dir)
    figures = paper_dir / "figures"
    (figures / ".concurrent.png").write_bytes(b"concurrent-png")
    link_injected = False

    def create_destination_before_publish(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal link_injected
        if not link_injected and destination == "overview.png" and str(source).endswith(".new"):
            link_injected = True
            real_link(
                ".concurrent.png",
                "overview.png",
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
                follow_symlinks=False,
            )
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    with (
        patch("src.orchestrator.paper_step.os.link", side_effect=create_destination_before_publish),
        pytest.raises(RuntimeError, match="roll back concept figure artifact publication"),
    ):
        _publish(paper_dir, render_dir)
    assert (figures / "overview.png").read_bytes() == b"concurrent-png"

    # cleans current temp when second temp write setup fails
    paper_dir, render_dir = _concept_dirs(_sub(tmp_path, "temp_setup"))
    _stage_valid(render_dir)
    real_fdopen = os.fdopen
    binary_writes = 0

    def fail_second_binary_writer(descriptor: int, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        nonlocal binary_writes
        if mode == "wb":
            binary_writes += 1
            if binary_writes == 2:
                os.close(descriptor)
                raise OSError("injected fdopen failure")
        return real_fdopen(descriptor, mode, *args, **kwargs)

    with (
        patch("src.orchestrator.paper_step.os.fdopen", side_effect=fail_second_binary_writer),
        pytest.raises(OSError, match="injected fdopen failure"),
    ):
        _publish(paper_dir, render_dir)
    assert list((paper_dir / "figures").glob("*.new")) == []


def test_replays_deferred_termination_signal() -> None:
    script = (
        "import os, signal\n"
        "from src.orchestrator.paper_step import _defer_publication_signals\n"
        "with _defer_publication_signals():\n"
        "    os.kill(os.getpid(), signal.SIGTERM)\n"
        "    print('critical section completed', flush=True)\n"
        "print('signal swallowed', flush=True)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == -signal.SIGTERM
    assert result.stdout.strip() == "critical section completed"


# --------------------------------------------------------------------------------------------
# record_concept_figure_failure / record_concept_figure_success
# --------------------------------------------------------------------------------------------


def _success_case(
    root: Path,
    *,
    extra_rows: list[dict[str, object]] | None = None,
    prompt_log: str = '{"title": "Overview", "panels": [{"id": "main"}]}',
    **row_fields: object,
) -> Path:
    figures = root / "figures"
    figures.mkdir()
    (figures / "overview.prompts.json").write_text(prompt_log, encoding="utf-8")
    row = {"figure_id": "overview", "generator": CONCEPT, "prompt_log_path": PROMPT_LOG, **row_fields}
    write_manifest(root, [row, *(extra_rows or [])])
    return figures


def _record_success(root: Path) -> None:
    paper_step.record_concept_figure_success(root, "overview")


def test_record_concept_figure_failure_and_prompt_log_guards(tmp_path: Path) -> None:
    # records failed render in prompt log and manifest
    root = _sub(tmp_path, "failed_render")
    figures = _success_case(root, prompt_log="{}")
    reason = "generate-concept-figure exited non-zero (rc=125)"
    paper_step.record_concept_figure_failure(root, "overview", reason)
    assert read_json(figures / "overview.prompts.json") == {
        "attempted_generator": CONCEPT,
        "reason": reason,
        "run_log_path": "concept_figure_run.log",
        "status": "failed",
    }
    row = _manifest_row(root)
    assert row["generator"] is None
    assert row["attempted_generator"] == CONCEPT
    assert row["fallback_reason"] == reason
    # rejects empty placeholder prompt log as provenance
    root = _sub(tmp_path, "placeholder_log")
    (root / "figures").mkdir()
    (root / "figures" / "overview.prompts.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="empty placeholder"):
        paper_step.read_concept_figure_prompt_log(root, "overview", PROMPT_LOG)
    # rejects symlinked prompt log
    root = _sub(tmp_path, "symlinked_log")
    (root / "figures").mkdir()
    outside = root / "outside.json"
    write_json(outside, {"secret": "outside"})
    (root / "figures" / "overview.prompts.json").symlink_to(outside)
    with pytest.raises(RuntimeError, match="safely open JSON artifact"):
        paper_step.read_concept_figure_prompt_log(root, "overview", PROMPT_LOG)
    # rejects symlinked manifest without overwriting target
    root = _sub(tmp_path, "symlinked_manifest")
    (root / "figures").mkdir()
    outside = root / "outside.json"
    original = json.dumps({"figures": []})
    outside.write_text(original, encoding="utf-8")
    (root / "figure_manifest.json").symlink_to(outside)
    with pytest.raises(RuntimeError, match="safely open JSON artifact"):
        paper_step.record_concept_figure_failure(root, "overview", "renderer failed")
    assert outside.read_text(encoding="utf-8") == original


def test_record_concept_figure_success_cleans_stale_pdf_artifacts(tmp_path: Path) -> None:
    # restores manifest after a prior failed render
    root = _sub(tmp_path, "restore_manifest")
    _success_case(root, generator=None, fallback_reason="earlier failure")
    _record_success(root)
    row = _manifest_row(root)
    assert row["generator"] == CONCEPT
    assert "fallback_reason" not in row
    assert "attempted_generator" not in row
    # cleans orphaned stale pdf quarantine (legacy pdf row / manifest already cleared)
    quarantine_cases: list[tuple[str, dict[str, Any]]] = [
        ("legacy_pdf_row", {"generator": None, "path_pdf": "figures/overview.pdf"}),
        ("manifest_already_cleared", {"path_pdf": ""}),
    ]
    for name, row_fields in quarantine_cases:
        root = _sub(tmp_path, name)
        figures = _success_case(root, prompt_log=PANEL_PLAN, **row_fields)
        orphan = figures / ".overview.pdf.0123456789abcdef.stale"
        orphan.write_bytes(b"stale")
        _record_success(root)
        assert not orphan.exists(), name
    # removes stale pdf artifact (legacy path_pdf / rich files.pdf)
    stale_cases: list[tuple[str, dict[str, Any], tuple[str, ...]]] = [
        ("legacy", {"path_pdf": "figures/overview.pdf"}, ("path_pdf",)),
        (
            "rich",
            {"files": {"pdf": "figures/overview.pdf", "png": "figures/overview.png"}},
            ("files", "pdf"),
        ),
    ]
    for name, row_fields, pdf_key in stale_cases:
        root = _sub(tmp_path, name)
        figures = _success_case(root, **row_fields)
        stale_pdf = figures / "overview.pdf"
        stale_pdf.write_bytes(b"stale legacy render")
        _record_success(root)
        value: Any = _manifest_row(root)
        for key in pdf_key:
            value = value[key]
        assert value == "", name
        assert not stale_pdf.exists(), name
    # unlinks stale pdf without touching symlink target
    root = _sub(tmp_path, "symlinked_stale_pdf")
    figures = _success_case(root, path_pdf="figures/overview.pdf")
    outside = root / "outside.pdf"
    outside.write_bytes(b"must remain")
    stale_pdf = figures / "overview.pdf"
    stale_pdf.symlink_to(outside)
    _record_success(root)
    assert not stale_pdf.exists()
    assert outside.read_bytes() == b"must remain"


def test_record_concept_figure_success_rejects_unsafe_stale_pdf_paths(tmp_path: Path) -> None:
    # rejects pdf owned by another figure
    root = _sub(tmp_path, "owned_by_another")
    figures = _success_case(root, path_pdf="figures/results.pdf")
    unrelated_pdf = figures / "results.pdf"
    unrelated_pdf.write_bytes(b"results")
    with pytest.raises(RuntimeError, match="canonical PDF path"):
        _record_success(root)
    assert unrelated_pdf.read_bytes() == b"results"
    # restores pdf when manifest write fails
    root = _sub(tmp_path, "manifest_write_fails")
    figures = _success_case(root, path_pdf="figures/overview.pdf")
    stale_pdf = figures / "overview.pdf"
    stale_pdf.write_bytes(b"fallback")
    manifest_path = root / "figure_manifest.json"
    original_manifest = manifest_path.read_text(encoding="utf-8")
    with (
        patch.object(paper_step, "_write_json_atomically", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        _record_success(root)
    assert stale_pdf.read_bytes() == b"fallback"
    assert manifest_path.read_text(encoding="utf-8") == original_manifest
    # rejects pdf shared with another manifest row
    root = _sub(tmp_path, "shared_pdf")
    figures = _success_case(
        root,
        path_pdf="figures/overview.pdf",
        extra_rows=[
            {"figure_id": "results", "path_pdf": "figures/overview.pdf", "generator": "matplotlib"}
        ],
    )
    shared_pdf = figures / "overview.pdf"
    shared_pdf.write_bytes(b"shared")
    with pytest.raises(RuntimeError, match="shared by another manifest row"):
        _record_success(root)
    assert shared_pdf.read_bytes() == b"shared"
    # rejects directory at stale pdf path
    root = _sub(tmp_path, "directory_pdf")
    figures = _success_case(root, path_pdf="figures/overview.pdf")
    stale_pdf = figures / "overview.pdf"
    stale_pdf.mkdir()
    manifest_path = root / "figure_manifest.json"
    original_manifest = manifest_path.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="regular file or symlink"):
        _record_success(root)
    assert stale_pdf.is_dir()
    assert manifest_path.read_text(encoding="utf-8") == original_manifest


def test_preserve_snapshot_changes_when_declared_concept_pdf_changes(tmp_path: Path) -> None:
    figures = tmp_path / "figures"
    figures.mkdir()
    write_strategy(
        tmp_path,
        [{"figure_id": "overview", "preferred_generator": CONCEPT, "purpose": "Show the method."}],
    )
    (figures / "overview.png").write_bytes(b"trusted-png")
    pdf_path = figures / "overview.pdf"
    pdf_path.write_bytes(b"trusted-pdf")
    (figures / "overview.prompts.json").write_text(PANEL_PLAN, encoding="utf-8")
    write_manifest(tmp_path, [diagram_row(generator=CONCEPT, prompt_log_path=PROMPT_LOG)])

    trusted = paper_step.fingerprint_concept_figure_preserve_state(tmp_path)
    pdf_path.write_bytes(b"mutated-pdf")
    mutated = paper_step.fingerprint_concept_figure_preserve_state(tmp_path)

    assert trusted is not None
    assert mutated is not None
    assert mutated != trusted


# --------------------------------------------------------------------------------------------
# Manifest schema + validate_figure_manifest for failed/claimed concept renders
# --------------------------------------------------------------------------------------------


def test_schema_failed_render_rows() -> None:
    # accepts documented png-only failed render
    rows = normalize_figure_manifest(
        {
            "figures": [
                diagram_row(
                    pdf=False,
                    panels=[{"panel_id": "overview", "content": "Placeholder"}],
                    generator=None,
                    attempted_generator=CONCEPT,
                    prompt_log_path=PROMPT_LOG,
                    fallback_reason="renderer failed",
                )
            ]
        }
    )
    assert rows[0].generator is None
    assert rows[0].fallback_reason == "renderer failed"
    # rejects png-only null generator without concept attempt
    with pytest.raises(ValidationError, match="PDF artifact path is required"):
        normalize_figure_manifest(
            {
                "figures": [
                    {
                        "figure_id": "results",
                        "figure_type": "bar",
                        "files": {"png": "figures/results.png"},
                        "panel_count": 1,
                        "panels": [{"panel_id": "results", "category_count": 2}],
                        "uses_progression": False,
                        "min_font_size_pt": 7.0,
                        "generator": None,
                        "prompt_log_path": "figures/results.prompts.json",
                        "fallback_reason": "claimed failure",
                    }
                ]
            }
        )


def _failed_case(root: Path) -> Path:
    """Workspace whose concept figure render failed and was recorded as such."""
    figures = root / "figures"
    figures.mkdir()
    write_png(figures / "overview.png", width=1600, height=1000)
    (figures / "overview.prompts.json").write_text("{}", encoding="utf-8")
    write_strategy(
        root,
        [
            {
                "figure_id": "overview",
                "figure_type": "diagram",
                "preferred_generator": CONCEPT,
                "required": True,
            }
        ],
    )
    write_manifest(
        root,
        [
            diagram_row(
                pdf=False,
                panels=[{"panel_id": "overview", "content": "Placeholder"}],
                generator=CONCEPT,
                prompt_log_path=PROMPT_LOG,
            )
        ],
    )
    paper_step.record_concept_figure_failure(
        root, "overview", "generate-concept-figure exited non-zero (rc=125)"
    )
    return root


def _alias_strategy(root: Path, *, drop_generator: bool) -> None:
    def mutate(strategy: dict[str, Any]) -> None:
        strategy["figure_specs"][0]["figure_type"] = "flow_diagram"
        if drop_generator:
            strategy["figure_specs"][0].pop("preferred_generator")

    _update_json(root / "paper_strategy.json", mutate)


def test_validation_accepts_failed_render_and_strategy_alias(tmp_path: Path) -> None:
    root = _failed_case(_sub(tmp_path, "failed"))
    # accepts failed render provenance (direct path / graphicspath)
    for name, tex_text in (
        ("matching_failed_render_provenance", r"\includegraphics{figures/overview}"),
        ("graphicspath", r"\graphicspath{{figures/}}\includegraphics{overview}"),
    ):
        assert _validate(root, tex_text)["errors"] == [], name
    # accepts strategy figure_type alias
    _alias_strategy(root, drop_generator=False)
    assert _validate(root)["errors"] == []
    # applies required diagram policy to alias
    _alias_strategy(root, drop_generator=True)
    report = _validate(root)
    assert "figure_specs[0] missing preferred_generator for required diagram" in report["errors"]


def _declare_pdf(root: Path) -> None:
    (root / "figures" / "overview.pdf").write_bytes(b"%PDF-1.7\n")
    _update_json(
        root / "figure_manifest.json",
        lambda m: m["figures"][0]["files"].__setitem__("pdf", "figures/overview.pdf"),
    )


def test_validation_resolves_graphicspath_and_extension_order(tmp_path: Path) -> None:
    # rejects root asset shadowing graphicspath
    root = _failed_case(_sub(tmp_path, "shadowing"))
    write_png(root / "overview.png", width=1600, height=1000)
    report = _validate(root, r"\graphicspath{{figures/}}\includegraphics{overview}")
    assert any("shadows graphicspath" in error for error in report["errors"])
    # uses first existing graphicspath asset
    root = _failed_case(_sub(tmp_path, "first_existing"))
    for name in ("first", "second"):
        (root / "figures" / name).mkdir()
        write_png(root / "figures" / name / "overview.png", width=1600, height=1000)
    _update_json(
        root / "figure_manifest.json",
        lambda m: m["figures"][0]["files"].__setitem__("png", "figures/second/overview.png"),
    )
    report = _validate(
        root, r"\graphicspath{{figures/first/}{figures/second/}}\includegraphics{overview.png}"
    )
    assert (
        "paper.tex references figure missing from figure_manifest.json: overview.png"
        in report["errors"]
    )
    # bounds graphicspath expansion
    root = _failed_case(_sub(tmp_path, "bounds"))
    search_paths = "".join(f"{{figures/path_{index}/}}" for index in range(33))
    report = _validate(root, rf"\graphicspath{{{search_paths}}}\includegraphics{{overview}}")
    assert any("more than 32 graphics search paths" in error for error in report["errors"])
    # rejects custom graphics extension order
    for extension_override in (
        r"\DeclareGraphicsExtensions{.png,.pdf}",
        r"\csname DeclareGraphicsExtensions\endcsname{.png,.pdf}",
        r"\makeatletter\def\Gin@extensions{.png,.pdf}\makeatother",
        r"\expandafter\def\csname Gin@extensions\endcsname{.png,.pdf}",
    ):
        report = _validate(root, extension_override + r"\includegraphics{figures/overview}")
        assert any(
            "custom graphics extension order" in error for error in report["errors"]
        ), extension_override
    # does not treat extensionless reference as explicit png
    root = _failed_case(_sub(tmp_path, "declared_pdf"))
    _declare_pdf(root)
    report = _validate(root)
    assert not any("references raster PNG" in error for error in report["errors"])
    # prefers pdf for extensionless pdflatex reference
    (root / "figures" / "overview.eps").write_bytes(b"%!PS-Adobe-3.0 EPSF-3.0\n")
    report = _validate(root)
    assert (
        "paper.tex references figure missing from figure_manifest.json: figures/overview"
        not in report["errors"]
    )


def test_bounded_tex_reader_rejects_oversized_input_and_symlink(tmp_path: Path) -> None:
    from src.orchestrator import tex_graphics

    # rejects oversized input
    tex_path = tmp_path / "paper.tex"
    tex_path.write_text("oversized", encoding="utf-8")
    with (
        patch.object(tex_graphics, "MAX_GRAPHICS_TEX_CHARS", 4),
        pytest.raises(ValueError, match="exceeds the graphics resolution limit"),
    ):
        tex_graphics.read_tex_file_bounded(tex_path)
    # rejects symlink
    target = tmp_path / "target.tex"
    target.write_text("content", encoding="utf-8")
    linked_path = tmp_path / "linked.tex"
    linked_path.symlink_to(target)
    with pytest.raises(ValueError, match="Unable to open regular TeX file"):
        tex_graphics.read_tex_file_bounded(linked_path)


def test_validation_rejects_unsupported_concept_claims(tmp_path: Path) -> None:
    # rejects failure log without attempted_generator
    root = _failed_case(_sub(tmp_path, "no_attempted_generator"))
    _update_json(
        root / "figures" / "overview.prompts.json",
        lambda log: log.pop("attempted_generator"),
    )
    report = _validate(root)
    assert (
        "figures[0] failure prompt log must record attempted_generator generate-concept-figure"
        in report["errors"]
    )
    # rejects concept output without matching strategy spec (matplotlib spec / no spec)
    for name, preferred_generator in (("matplotlib", "matplotlib"), ("no_spec", None)):
        root = _sub(tmp_path, name)
        figures = root / "figures"
        figures.mkdir()
        write_png(figures / "overview.png", width=1600, height=1000)
        (figures / "overview.prompts.json").write_text(PANEL_PLAN, encoding="utf-8")
        specs = []
        if preferred_generator is not None:
            specs.append(
                {
                    "figure_id": "overview",
                    "figure_type": "diagram",
                    "preferred_generator": preferred_generator,
                }
            )
        write_strategy(root, specs)
        write_manifest(
            root,
            [
                diagram_row(
                    pdf=False,
                    panels=[{"panel_id": "overview", "content": "Claimed concept output"}],
                    generator=CONCEPT,
                    prompt_log_path=PROMPT_LOG,
                )
            ],
        )
        report = _validate(root)
        assert (
            "figures[0] claims generate-concept-figure output without a matching strategy spec"
            in report["errors"]
        ), name
