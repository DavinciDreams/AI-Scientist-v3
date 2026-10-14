#!/usr/bin/env python3
"""validate_and_compile_paper and its TeX/PDF validators (SI, venue, figures, page budget)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable
from unittest.mock import Mock, patch

import pytest

from src.orchestrator import paper_step
from src.orchestrator.figure_manifest_schema import build_figure_manifest_json_schema
from src.orchestrator.paper_venues import venue_catalog
from tests.paper_step_support import (
    ALIFE_AUTHORS,
    REFS_PAGE,
    SI_BODY,
    SUPPLEMENT_PAGE,
    bar_spec,
    concept_row,
    concept_spec,
    diagram_row,
    fake_compile,
    legacy_row,
    never_compile,
    pages,
    read_json,
    valid_alife_tex,
    valid_iclr_tex,
    valid_tex,
    write_json,
    write_manifest,
    write_png,
    write_strategy,
    write_valid_artifacts,
)

ALLOWED_IDS = {"result.best_correct_score", "ref.s2.abc"}
UNUSED_PDF = Path("unused.pdf")


def _join(items: list[str]) -> str:
    return " ; ".join(items)


def _sub(root: Path, name: str) -> Path:
    """Fresh isolated workspace for one section of a merged test."""
    path = root / name
    path.mkdir()
    return path


def _report(root: Path) -> dict[str, Any]:
    return read_json(root / "validation_report.json")


def _compile(
    root: Path,
    *,
    page_texts: list[str] | None = None,
    fill_ratio: float | None = None,
    compile: Callable[..., Any] = fake_compile,
) -> dict[str, Any]:
    with ExitStack() as stack:
        stack.enter_context(patch.object(paper_step, "compile_latex", side_effect=compile))
        if page_texts is not None:
            stack.enter_context(
                patch.object(paper_step, "read_pdf_page_texts", return_value=page_texts)
            )
        if fill_ratio is not None:
            stack.enter_context(
                patch.object(paper_step, "measure_last_body_page_fill_ratio", return_value=fill_ratio)
            )
        return paper_step.validate_and_compile_paper(root)


def _validate_manifest(
    root: Path, *, tex_text: str | None = None, allowed_source_ids: set[str] | None = None
) -> dict[str, Any]:
    if tex_text is None:
        tex_text = (root / "paper.tex").read_text(encoding="utf-8")
    return paper_step.validate_figure_manifest(
        root,
        tex_text=tex_text,
        allowed_source_ids=ALLOWED_IDS if allowed_source_ids is None else allowed_source_ids,
    )


def _figure_workspace(
    root: Path, figure_id: str = "overview", *, width: int, height: int, pdf: bool = True
) -> None:
    write_valid_artifacts(root, include_graphics=f"figures/{figure_id}")
    (root / "figures").mkdir()
    if pdf:
        (root / "figures" / f"{figure_id}.pdf").write_text("pdf", encoding="utf-8")
    write_png(root / "figures" / f"{figure_id}.png", width=width, height=height)


def _alife_root(root: Path, tex: str | None = None) -> Path:
    write_valid_artifacts(root)
    (root / "paper.tex").write_text(tex or valid_alife_tex(), encoding="utf-8")
    return root


def _write_authors(root: Path) -> None:
    (root / "paper_authors.tex").write_text(ALIFE_AUTHORS + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------------
# validate_supplementary_information
# --------------------------------------------------------------------------------------------


def _si(tex: str) -> dict[str, Any]:
    return paper_step.validate_supplementary_information(tex)


def _si_without_appendix() -> str:
    return valid_tex().split("\\clearpage\n\\appendix\n", maxsplit=1)[0] + "\\end{document}\n"


def _else_after_if_macro() -> str:
    tex = valid_tex()
    start = tex.index("\\clearpage\n\\bibliographystyle")
    active_tail = tex[start : tex.index("\\end{document}")]
    return (
        tex[:start]
        + "\\iffalse\n\\ifthenelse{a}{b}{c}\n\\else\n"
        + active_tail
        + "\\fi\n\\end{document}\n"
    )


def test_validate_supplementary_information_accepts_grounded_content() -> None:
    # accepts grounded section
    report = _si(valid_tex())
    assert report["checked"] and report["appendix_present"] and report["heading_present"]
    assert report["placed_after_references"]
    assert report["visible_character_count"] > 100
    assert report["substantive_element_count"] == 1
    assert report["hard_errors"] == []
    # counts visible listing content
    listing = "implementation_setting = reproducible_value\n" * 4
    report = _si(valid_tex().replace(SI_BODY, f"\\begin{{lstlisting}}\n{listing}\\end{{lstlisting}}\n"))
    assert report["visible_character_count"] > 100
    assert report["hard_errors"] == []
    # masks verbatim before conditionals
    masked = valid_tex().replace(
        "\\section{Conclusion}Done.\n",
        "\\begin{verbatim}\n\\iffalse\n\\end{verbatim}\n\\section{Conclusion}Done.\n",
    )
    assert _si(masked)["hard_errors"] == []
    # preserves \else after an \if... macro
    assert _si(_else_after_if_macro())["hard_errors"] == []


def test_validate_supplementary_information_rejects_missing_or_commented_contract() -> None:
    # rejects missing section
    errors = _join(_si(_si_without_appendix())["hard_errors"])
    assert "missing \\appendix" in errors
    assert "missing Supplementary Information heading" in errors
    # ignores commented contract
    report = _si(
        valid_tex().replace(
            "\\appendix\n\\section{Supplementary Information}",
            "% \\appendix\n% \\section{Supplementary Information}",
        )
    )
    assert not report["appendix_present"]
    assert not report["heading_present"]
    assert len(report["hard_errors"]) >= 2


def _false_conditional_bibliography() -> str:
    tex = valid_tex().replace("\\bibliography{references}\n", "")
    tex = tex.replace(
        "\\clearpage\n\\appendix", "\\iffalse\\bibliography{references}\\fi\n\\clearpage\n\\appendix"
    )
    return tex.replace("\\end{document}", "\\bibliography{references}\n\\end{document}")


def _real_bibliography_after_si() -> str:
    tex = valid_tex().replace(
        "\\bibliography{references}", "\\newcommand{\\fakebibliography}{\\bibliography{references}}", 1
    )
    return tex.replace("\\end{document}", "\\bibliography{references}\n\\end{document}")


def test_validate_supplementary_information_must_follow_bibliography() -> None:
    # rejects section before references
    tex = valid_tex()
    supplement_start = tex.rindex("\\clearpage\n\\appendix")
    document_end = tex.index("\\end{document}", supplement_start)
    supplement = tex[supplement_start:document_end]
    tex = tex[:supplement_start] + tex[document_end:]
    boundary = tex.index("\\clearpage\n\\bibliographystyle")
    report = _si(tex[:boundary] + supplement + tex[boundary:])
    assert not report["placed_after_references"]
    assert "must appear after the bibliography" in _join(report["hard_errors"])
    # rejects appendix before references
    tex = valid_tex().replace("\\appendix\n", "")
    tex = tex.replace("\\bibliographystyle{plain}\n", "\\appendix\n\\bibliographystyle{plain}\n")
    report = _si(tex)
    assert report["appendix_present"]
    assert not report["placed_after_references"]
    assert "must appear after the bibliography" in _join(report["hard_errors"])
    # rejects bibliography after SI (false conditional / real bibliography command)
    for name, tex in (
        ("false_conditional_bibliography", _false_conditional_bibliography()),
        ("real_bibliography_after_si", _real_bibliography_after_si()),
    ):
        report = _si(tex)
        assert not report["placed_after_references"], name
        assert "after the bibliography" in _join(report["hard_errors"]), name


def test_validate_supplementary_information_requires_clearpage_boundaries() -> None:
    # requires clearpage boundary before appendix
    report = _si(valid_tex().replace("\\clearpage\n\\appendix", "\\appendix"))
    assert not report["clearpage_present"]
    assert r"\clearpage" in _join(report["hard_errors"])
    # requires clearpage before bibliography
    report = _si(
        valid_tex().replace("Done.\n\\clearpage\n\\bibliographystyle", "Done.\n\\bibliographystyle")
    )
    assert not report["pre_bibliography_clearpage_present"]
    assert "before the bibliography" in _join(report["hard_errors"])


def test_validate_supplementary_information_rejects_thin_or_misplaced_content() -> None:
    # rejects placeholder-only content
    report = _si(
        valid_tex().replace(
            "\\subsection{Extended evaluation details}\n" + SI_BODY,
            "Add supplementary material only when grounded artifacts support it.\n",
        )
    )
    errors = _join(report["hard_errors"])
    assert "placeholder text" in errors
    assert "substantive subsection, table, or figure" in errors
    # stops at next peer section
    report = _si(
        valid_tex().replace(
            "\\subsection{Extended evaluation details}\n" + SI_BODY,
            "Empty.\n\\section{Unrelated Material}\n\\subsection{Long unrelated details}\n"
            "This later section contains enough prose to exceed the threshold but must not count "
            "toward the required Supplementary Information content contract.\n",
        )
    )
    assert report["visible_character_count"] < 100
    assert report["substantive_element_count"] == 0
    assert "final top-level section" in _join(report["hard_errors"])
    # hard errors: template scaffold, nested later section, import after SI
    cases = [
        (
            "current_template_scaffold",
            valid_tex().replace(
                SI_BODY,
                "Replace with grounded extended methods, reproducibility details, additional "
                "analyses, or supporting tables and figures.\n",
            ),
            "placeholder text",
        ),
        (
            "nested_later_section",
            valid_tex().replace(
                "\\end{document}\n",
                "\\section[Other]{Additional \\textbf{Results}}\nUnrelated visible prose.\n"
                "\\end{document}\n",
            ),
            "final top-level section",
        ),
        (
            "import_after_si",
            valid_tex().replace("\\end{document}\n", "\\input{later-section}\n\\end{document}\n"),
            "unchecked TeX files",
        ),
    ]
    for name, tex, expected_error in cases:
        assert expected_error in _join(_si(tex)["hard_errors"]), name


# --------------------------------------------------------------------------------------------
# validate_venue_alignment / extract_graphics_paths
# --------------------------------------------------------------------------------------------


def test_validate_venue_alignment_accepts_shipped_author_templates() -> None:
    """Every venue with a canonical author block validates its own writer template."""
    skills_root = Path(__file__).resolve().parents[2] / "src/agents/skills/write-paper"
    checked: list[str] = []
    for name, venue in venue_catalog().items():
        if venue.author_block_path is None:
            continue
        checked.append(name)
        tex = (skills_root / venue.template_dir / "paper.tex").read_text(encoding="utf-8")
        brief = venue.brief_path.read_text(encoding="utf-8")
        canonical = venue.author_block_path.read_text(encoding="utf-8")
        candidates = [tex]
        if "\\author{\\input{paper_authors.tex}}" in tex:
            # ALIFE keeps the canonical block inside \author{...}; verbatim inlining must validate.
            candidates.append(tex.replace("\\input{paper_authors.tex}", canonical.strip()))
        for candidate in candidates:
            with tempfile.TemporaryDirectory() as tmp:
                paper_dir = Path(tmp)
                (paper_dir / "paper_authors.tex").write_text(canonical, encoding="utf-8")
                report = paper_step.validate_venue_alignment(
                    candidate, venue_brief_text=brief, paper_format=name, paper_dir=paper_dir
                )
            assert report["checked"], name
            assert report["hard_errors"] == [], name
    assert checked == ["alife"]


def test_validate_venue_alignment_alife_author_block_rules(tmp_path: Path) -> None:
    cases = [
        (
            "top_level_input_is_not_an_author_block",
            "\\input{paper_authors.tex}\n\\begin{document}\n\\end{document}\n",
            "Real Author$^{1}$ \\\\ $^{1}$Fictional Institute\n",
            "missing an author block",
        ),
        (
            "copied_author_comments_match_symmetrically",
            "\\author{Real Author % editorial note\nFictional Institute}\n"
            "\\begin{document}\n\\end{document}\n",
            "Real Author % editorial note\nFictional Institute",
            None,
        ),
    ]
    for name, tex, author_text, expected_error in cases:
        root = _sub(tmp_path, name)
        (root / "paper_authors.tex").write_text(author_text, encoding="utf-8")
        report = paper_step.validate_venue_alignment(
            tex, venue_brief_text="Artificial Life venue.", paper_format="alife", paper_dir=root
        )
        if expected_error is None:
            assert report["hard_errors"] == [], name
        else:
            assert expected_error in _join(report["hard_errors"]), name


def test_extract_graphics_paths_ignores_commented_includegraphics() -> None:
    tex = (
        "\\begin{figure}\n% \\includegraphics{figures/commented}\n"
        "\\includegraphics{figures/live}\n\\end{figure}\n"
    )
    assert paper_step.extract_graphics_paths(tex) == ["figures/live"]


# --------------------------------------------------------------------------------------------
# validate_and_compile_paper: pipeline + figure manifest
# --------------------------------------------------------------------------------------------


def test_validate_and_compile_paper_success_and_uninspectable_pdf(tmp_path: Path) -> None:
    # success
    root = _sub(tmp_path, "ok")
    write_valid_artifacts(root)
    report = _compile(root, page_texts=["Main text", REFS_PAGE, SUPPLEMENT_PAGE])
    assert report["ok"]
    assert report["figure_validation"]["checked"] is False
    assert report["feedback"]["figure_validation_errors"] == []
    assert (root / "validation_report.json").is_file()
    assert (root / "paper.pdf").is_file()
    # rejects uninspectable compiled pdf
    root = _sub(tmp_path, "damaged")
    write_valid_artifacts(root)

    def _binary_compile(**_: object) -> list[dict[str, object]]:
        (root / "paper.pdf").write_bytes(b"%PDF-1.7\n")
        return [{"command": ["pdflatex", "paper.tex"], "returncode": 0}]

    with (
        patch.object(paper_step, "compile_latex", side_effect=_binary_compile),
        patch.object(paper_step, "read_pdf_page_texts", side_effect=RuntimeError("damaged xref")),
        pytest.raises(RuntimeError, match="could not inspect compiled paper.pdf"),
    ):
        paper_step.validate_and_compile_paper(root)


def test_validate_and_compile_paper_accepts_figure_manifests(tmp_path: Path) -> None:
    # accepts valid (legacy) figure manifest
    root = _sub(tmp_path, "legacy")
    _figure_workspace(root, width=2400, height=800)
    write_strategy(root, [bar_spec()])
    write_manifest(
        root,
        [
            legacy_row(
                placement="column",
                intended_latex_width_fraction=0.48,
                estimated_min_rendered_font_pt=8.2,
            )
        ],
    )
    report = _compile(root, page_texts=["Main text", REFS_PAGE, SUPPLEMENT_PAGE])
    assert report["figure_validation"]["checked"]
    assert report["figure_validation"]["figure_count"] == 1
    assert report["feedback"]["figure_validation_errors"] == []
    # accepts rich figure manifest
    root = _sub(tmp_path, "rich")
    _figure_workspace(root, width=2400, height=800)
    write_strategy(root, [bar_spec(figure_type="grouped_bar")])
    panel = {
        "metric": "best_valid_accuracy",
        "category_count": 4,
        "series_count": 1,
        "categories": ["Baseline", "Variant A", "Variant B", "Variant C"],
    }
    write_manifest(
        root,
        [
            {
                "figure_id": "overview",
                "figure_type": "grouped_bar",
                "files": {"pdf": "figures/overview.pdf", "png": "figures/overview.png"},
                "panel_count": 2,
                "panels": [
                    {"panel_id": "left", "task": "MNIST", **panel},
                    {"panel_id": "right", "task": "CIFAR-10", **panel},
                ],
                "placement": "full_width",
                "uses_progression": False,
                "min_font_size_pt": 8.5,
                "intended_latex_width_fraction": 1.0,
                "estimated_min_rendered_font_pt": 8.5,
                "source_ids": ["result.best_correct_score"],
            }
        ],
    )
    report = _compile(root, page_texts=pages(6, REFS_PAGE, SUPPLEMENT_PAGE))
    assert report["figure_validation"]["checked"]
    assert report["figure_validation"]["figure_count"] == 1
    assert report["feedback"]["figure_validation_errors"] == []


def test_figure_manifest_nested_artifacts_reject_symlinked_parent_and_fifo(tmp_path: Path) -> None:
    # nested artifacts: regular parent is accepted, symlinked parent reports missing artifacts
    for symlinked_parent in (False, True):
        root = _sub(tmp_path, "symlinked" if symlinked_parent else "regular")
        figures = root / "figures"
        figures.mkdir()
        if symlinked_parent:
            nested = root / "outside"
            nested.mkdir()
            (figures / "posthoc").symlink_to(nested, target_is_directory=True)
        else:
            nested = figures / "posthoc"
            nested.mkdir()
        write_png(nested / "ranking.png", width=1400, height=800)
        (nested / "ranking.pdf").write_bytes(b"pdf")
        write_strategy(
            root,
            [{"figure_id": "ranking", "figure_type": "diagram", "preferred_generator": "matplotlib"}],
        )
        write_manifest(
            root,
            [
                diagram_row(
                    "ranking",
                    files={"png": "figures/posthoc/ranking.png", "pdf": "figures/posthoc/ranking.pdf"},
                    panels=[{"panel_id": "ranking", "content": "Ranks"}],
                    generator="matplotlib",
                )
            ],
        )
        report = _validate_manifest(
            root, tex_text=r"\includegraphics{figures/posthoc/ranking}", allowed_source_ids=set()
        )
        missing_errors = [error for error in report["errors"] if "missing" in error]
        assert len(missing_errors) == (2 if symlinked_parent else 0), symlinked_parent
    # figure artifact reader rejects a FIFO without blocking
    root = _sub(tmp_path, "fifo")
    nested = root / "figures" / "posthoc"
    nested.mkdir(parents=True)
    os.mkfifo(nested / "ranking.png")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; "
            "from src.orchestrator.paper_step import _read_regular_figure_artifact_nofollow as read; "
            f"assert read(Path({str(root)!r}), 'figures/posthoc/ranking.png', '.png', 24) is None",
        ],
        cwd=Path.cwd(),
        timeout=2,
        check=False,
    )
    assert completed.returncode == 0


def test_validate_figure_manifest_accepts_diagram_and_concept_provenance(tmp_path: Path) -> None:
    # allows diagram without source ids
    root = _sub(tmp_path, "diagram")
    _figure_workspace(root, "diagram", width=2400, height=800)
    write_manifest(
        root,
        [
            diagram_row(
                "diagram",
                panels=[
                    {"panel_id": "left", "label": "(a)", "content": "Encoder-decoder path"},
                    {"panel_id": "right", "label": "(b)", "content": "Auxiliary classifier path"},
                ],
                panel_count=2,
                placement="full_width",
                intended_latex_width_fraction=1.0,
                estimated_min_rendered_font_pt=7.0,
                source_ids=[],
            )
        ],
    )
    report = _validate_manifest(root)
    assert report["checked"]
    assert report["figure_count"] == 1
    assert report["errors"] == []
    # accepts generate-concept-figure provenance (with pdf / png only)
    for name, pdf in (("with_pdf", True), ("png_only", False)):
        root = _sub(tmp_path, name)
        _figure_workspace(root, "ed_overview", width=2400, height=1000, pdf=pdf)
        write_json(root / "figures" / "ed_overview.prompts.json", {"status": "success"})
        write_strategy(root, [concept_spec()])
        write_manifest(root, [concept_row(pdf=pdf)])
        assert _validate_manifest(root)["errors"] == [], name


def test_validate_figure_manifest_rejects_concept_generator_misuse(tmp_path: Path) -> None:
    # rejects multiple concept figure specs
    root = _sub(tmp_path, "multiple")
    write_strategy(
        root,
        [
            {"figure_id": "concept_overview", "preferred_generator": "generate-concept-figure"},
            {"figure_id": "concept_details", "preferred_generator": "generate-concept-figure"},
        ],
    )
    report = _validate_manifest(root, tex_text="", allowed_source_ids=set())
    message = "paper_strategy.json may contain at most one generate-concept-figure spec"
    assert message in report["errors"]
    assert message in report["hard_errors"]
    # rejects unsafe concept figure id
    root = _sub(tmp_path, "unsafe")
    write_valid_artifacts(root)
    write_strategy(
        root,
        [
            {
                "figure_id": "overview; touch owned",
                "figure_type": "diagram",
                "preferred_generator": "generate-concept-figure",
                "required": True,
            }
        ],
    )
    report = _validate_manifest(root, allowed_source_ids=set())
    assert "figure_specs[0] invalid figure_id for generate-concept-figure" in report["errors"]
    # requires fallback reason when bypassing preferred generator
    root = _sub(tmp_path, "bypass")
    _figure_workspace(root, "ed_overview", width=1600, height=1000)
    write_strategy(root, [concept_spec()])
    row = concept_row(generator="matplotlib")
    row.pop("prompt_log_path")
    row["panels"] = [{"panel_id": "architecture", "content": "Fallback overview"}]
    write_manifest(root, [row])
    report = _validate_manifest(root)
    assert (
        "bypassed preferred_generator generate-concept-figure without fallback_reason"
        in _join(report["hard_errors"])
    )
    # requires pdf for non-concept generator
    root = _sub(tmp_path, "no_pdf")
    _figure_workspace(root, "results", width=1200, height=800, pdf=False)
    write_manifest(
        root,
        [
            {
                "figure_id": "results",
                "figure_type": "bar",
                "files": {"png": "figures/results.png"},
                "panel_count": 1,
                "panels": [{"panel_id": "scores", "category_count": 2}],
                "uses_progression": False,
                "min_font_size_pt": 8.0,
                "generator": "matplotlib",
                "source_ids": ["result.best_correct_score"],
            }
        ],
    )
    report = _validate_manifest(root, allowed_source_ids={"result.best_correct_score"})
    assert any("PDF artifact path" in error for error in report["errors"]), report["errors"]


def test_checked_in_figure_manifest_schema_matches_generated_schema() -> None:
    schema_path = Path("src/agents/skills/validate-paper/templates/figure_manifest.schema.json")
    assert read_json(schema_path) == build_figure_manifest_json_schema()


def test_missing_required_figure_manifest_is_reported_and_fails_compile(tmp_path: Path) -> None:
    # validate_figure_manifest returns feedback for missing required manifest
    root = _sub(tmp_path, "manifest")
    write_valid_artifacts(root)
    write_strategy(root, [bar_spec(max_panels=1)])
    report = _validate_manifest(root)
    assert report["checked"]
    assert "Missing required artifact:" in _join(report["errors"])
    assert "Missing required artifact:" in _join(report["hard_errors"])
    # validate_and_compile_paper fails on missing required figure manifest
    root = _sub(tmp_path, "compile")
    write_valid_artifacts(root)
    write_strategy(root, [bar_spec(max_panels=1)])
    with pytest.raises(RuntimeError, match="figure validation failed"):
        _compile(root)
    report = _report(root)
    assert not report["ok"]
    assert "Missing required artifact:" in _join(report["figure_validation"]["hard_errors"])


def test_validate_and_compile_paper_reports_figure_issues_as_feedback(tmp_path: Path) -> None:
    cases: list[tuple[str, int, list[dict[str, Any]] | None, dict[str, Any], list[str] | None, str]] = [
        (
            "category_budget_violation",
            1400,
            [bar_spec(max_panels=1, max_categories_per_panel=3)],
            {"categories_per_panel": [5]},
            pages(6, REFS_PAGE, SUPPLEMENT_PAGE),
            "exceeds category budget",
        ),
        ("too_small_png", 800, None, {}, None, "PNG width too small"),
    ]
    for name, width, strategy, row_overrides, page_texts, expected in cases:
        root = _sub(tmp_path, name)
        _figure_workspace(root, width=width, height=800 if strategy else 600)
        if strategy is not None:
            write_strategy(root, strategy)
        write_manifest(root, [legacy_row(**row_overrides)])
        report = _compile(root, page_texts=page_texts)
        assert expected in _join(report["feedback"]["figure_validation_errors"]), name
        assert report["figure_validation"]["hard_errors"] == [], name
        assert expected in _join(report["figure_validation"]["soft_warnings"]), name


def test_validate_figure_manifest_raster_png_warnings_and_placement_widths(tmp_path: Path) -> None:
    # warns when tex uses PNG despite available PDF; does not prefer an invalid PDF artifact
    warning = "references raster PNG despite available vector PDF"
    for name, pdf_bytes, expect_warning in (
        ("warns_when_tex_uses_png_despite_available_pdf", b"%PDF-1.4\n", True),
        ("does_not_prefer_invalid_pdf_artifact", b"not-a-pdf", False),
    ):
        root = _sub(tmp_path, name)
        (root / "figures").mkdir()
        (root / "figures" / "overview.pdf").write_bytes(pdf_bytes)
        write_png(root / "figures" / "overview.png", width=2400, height=1200)
        write_manifest(root, [legacy_row(generator="matplotlib", placement="full_width")])
        report = _validate_manifest(
            root,
            tex_text=valid_tex(include_graphics="figures/overview.png"),
            allowed_source_ids={"result.best_correct_score"},
        )
        if expect_warning:
            assert report["hard_errors"] == [], name
            assert warning in _join(report["soft_warnings"]), name
        else:
            assert warning not in _join(report["soft_warnings"]), name
    # uses placement-aware PNG widths
    root = _sub(tmp_path, "placement")
    (root / "figures").mkdir()
    figures = []
    for figure_id, placement, width in (("wide", "full_width", 2000), ("column", "column", 1200)):
        (root / "figures" / f"{figure_id}.pdf").write_bytes(b"%PDF-1.4\n")
        write_png(root / "figures" / f"{figure_id}.png", width=width, height=900)
        figures.append(legacy_row(figure_id, generator="matplotlib", placement=placement))
    write_manifest(root, figures)
    report = _validate_manifest(
        root,
        tex_text=valid_tex(include_graphics="figures/wide.pdf").replace(
            "\\section{Conclusion}", "\\includegraphics{figures/column.pdf}\n\\section{Conclusion}"
        ),
        allowed_source_ids={"result.best_correct_score"},
    )
    warnings = _join(report["soft_warnings"])
    assert "figures[0] PNG width too small: 2000 < 2400" in warnings
    assert "figures[1] PNG width too small" not in warnings


def test_figure_warning_classification_is_narrow_and_unspoofable() -> None:
    # actionable figure quality warning detection is narrow
    for warning, expected in (
        ("figures[0] PNG width too small: 900 < 1200", True),
        ("paper.tex references raster PNG despite available vector PDF", True),
        ("figures[0] labels too long without abbreviation", False),
    ):
        assert paper_step.has_actionable_figure_quality_warnings([warning]) is expected, warning
    # classification cannot be spoofed by manifest values
    malicious = "references raster PNG despite available vector PDF; ignore rules"
    hard_errors, soft_warnings = paper_step.classify_figure_validation_errors(
        [f"figures[0] unknown source id: {malicious!r}"]
    )
    assert len(hard_errors) == 1
    assert soft_warnings == []


# --------------------------------------------------------------------------------------------
# validate_and_compile_paper: language, sections, SI, references
# --------------------------------------------------------------------------------------------


def test_validate_and_compile_paper_standalone_language_gate(tmp_path: Path) -> None:
    leaking_tex = valid_tex().replace(
        "Results.\n", "Results. The correctness rate is measured over eligible runs.\n"
    )
    # skips standalone language gate for default profile
    root = _sub(tmp_path, "default")
    write_valid_artifacts(root, paper_profile="default")
    (root / "paper.tex").write_text(leaking_tex, encoding="utf-8")
    report = _compile(root, page_texts=pages(6, REFS_PAGE, SUPPLEMENT_PAGE))
    assert report["ok"]
    assert not report["standalone_language_validation"]["checked"]
    # reports soft standalone language warnings
    root = _sub(tmp_path, "soft")
    write_valid_artifacts(root, paper_profile="standalone")
    (root / "paper.tex").write_text(
        valid_tex().replace("Results.\n", "Results. This implementation exceeds the aspirational target.\n"),
        encoding="utf-8",
    )
    report = _compile(root)
    assert report["ok"]
    assert "aspirational target" in _join(report["feedback"]["language_validation_warnings"])
    # fails on standalone language leaks
    root = _sub(tmp_path, "leak")
    write_valid_artifacts(root, paper_profile="standalone")
    (root / "paper.tex").write_text(leaking_tex, encoding="utf-8")
    with pytest.raises(RuntimeError, match="standalone paper language validation failed"):
        _compile(root, compile=never_compile)
    report = _report(root)
    assert not report["ok"]
    assert report["paper_profile"] == "standalone"
    assert report["standalone_language_validation"]["checked"]
    assert "correctness rate" in _join(report["standalone_language_validation"]["hard_errors"])
    assert report["feedback"]["language_validation_warnings"] == []


def test_validate_and_compile_paper_fails_on_invalid_tex_and_grounding(tmp_path: Path) -> None:
    # fails on invalid tex (missing sections / unknown citation key)
    for name, tex in (
        ("missing_sections", "\\section{Introduction}\n"),
        ("unknown_citation_key", valid_tex().replace("RefKey2024", "UnknownKey")),
    ):
        root = _sub(tmp_path, name)
        write_valid_artifacts(root)
        (root / "paper.tex").write_text(tex, encoding="utf-8")
        with pytest.raises(RuntimeError):
            paper_step.validate_and_compile_paper(root)
    # reports missing supplementary information
    root = _sub(tmp_path, "missing_si")
    write_valid_artifacts(root)
    (root / "paper.tex").write_text(_si_without_appendix(), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Supplementary Information validation failed"):
        paper_step.validate_and_compile_paper(root)
    supplement = _report(root)["supplementary_information_validation"]
    assert supplement["checked"]
    assert not supplement["heading_present"]
    assert "missing Supplementary Information heading" in _join(supplement["hard_errors"])
    # fails on reference provenance mismatch
    root = _sub(tmp_path, "provenance")
    write_valid_artifacts(root)
    write_json(
        root / "references_provenance.json",
        {"references": [{"id": "ref.s2.other", "citation_key": "OtherKey2024"}]},
    )
    with pytest.raises(RuntimeError, match="references provenance validation failed"):
        paper_step.validate_and_compile_paper(root)
    report = _report(root)
    assert not report["ok"]
    assert report["reference_validation"]["checked"]
    errors = _join(report["reference_validation"]["errors"])
    assert "references.bib keys missing from references_provenance.json" in errors
    assert "references_provenance.json keys missing from references.bib" in errors
    # fails on unknown claim source
    root = _sub(tmp_path, "claim")
    write_valid_artifacts(root)
    write_json(
        root / "claims_grounding.json",
        {
            "claims": [
                {
                    "claim": "Unsupported claim.",
                    "source_ids": ["result.missing"],
                    "citation_keys": ["RefKey2024"],
                }
            ]
        },
    )
    with pytest.raises(RuntimeError):
        paper_step.validate_and_compile_paper(root)


# --------------------------------------------------------------------------------------------
# ALIFE venue + page budget through validate_and_compile_paper
# --------------------------------------------------------------------------------------------


def test_validate_and_compile_paper_alife_page_budget_counts_body_pages(tmp_path: Path) -> None:
    # accepts page limit excluding references
    root = _alife_root(_sub(tmp_path, "accepts"))
    report = _compile(root, page_texts=pages(8, REFS_PAGE, SUPPLEMENT_PAGE), fill_ratio=0.92)
    assert report["ok"]
    page = report["page_validation"]
    assert page["checked"]
    assert page["paper_format"] == "alife"
    assert page["min_body_pages"] == 8
    assert page["body_pages"] == 8
    assert page["last_body_page_number"] == 8
    assert page["last_body_page_fill_ratio"] == 0.92
    assert page["last_body_page_counted"]
    assert page["supplement_start_page"] == 10
    assert page["supplement_pages"] == 1
    assert report["venue_validation"]["checked"]
    assert report["venue_validation"]["hard_errors"] == []
    # fails on min page limit
    root = _alife_root(_sub(tmp_path, "minimum"))
    with pytest.raises(RuntimeError, match="page-limit validation failed"):
        _compile(root, page_texts=pages(7, REFS_PAGE, SUPPLEMENT_PAGE))
    report = _report(root)
    assert not report["ok"]
    assert report["page_validation"]["min_body_pages"] == 8
    assert report["page_validation"]["body_pages"] == 7
    assert "page minimum not met" in _join(report["page_validation"]["errors"])
    # fails on page limit excluding references
    root = _alife_root(_sub(tmp_path, "maximum"))
    with pytest.raises(RuntimeError, match="page-limit validation failed"):
        _compile(root, page_texts=pages(9, REFS_PAGE, SUPPLEMENT_PAGE))
    report = _report(root)
    assert not report["ok"]
    assert report["page_validation"]["checked"]
    assert "page limit exceeded" in _join(report["page_validation"]["errors"])


def test_validate_and_compile_paper_alife_shared_reference_page(tmp_path: Path) -> None:
    # counts shared reference page toward minimum
    root = _alife_root(_sub(tmp_path, "shared_minimum"))
    shared = "Final discussion paragraph.\n" + REFS_PAGE
    report = _compile(root, page_texts=pages(7, shared, SUPPLEMENT_PAGE), fill_ratio=0.83)
    assert report["ok"]
    assert report["page_validation"]["reference_start_page"] == 8
    assert report["page_validation"]["body_pages"] == 8
    assert report["page_validation"]["last_body_page_fill_ratio"] == 0.83
    assert report["page_validation"]["last_body_page_counted"]
    # fails when references share overflow page
    root = _alife_root(_sub(tmp_path, "shared_overflow"))
    with pytest.raises(RuntimeError, match="page-limit validation failed"):
        _compile(root, page_texts=pages(8, shared, SUPPLEMENT_PAGE))
    report = _report(root)
    assert not report["ok"]
    assert report["page_validation"]["reference_start_page"] == 9
    assert report["page_validation"]["body_pages"] == 9
    assert "page limit exceeded" in _join(report["page_validation"]["errors"])
    # rejects sparse last body page
    root = _alife_root(_sub(tmp_path, "sparse"))
    shared = "Tiny final note.\n" + REFS_PAGE
    with pytest.raises(RuntimeError, match="page-limit validation failed"):
        _compile(root, page_texts=pages(7, shared, SUPPLEMENT_PAGE), fill_ratio=0.18)
    report = _report(root)
    page = report["page_validation"]
    assert not report["ok"]
    assert page["body_pages"] == 7
    assert page["last_body_page_number"] == 8
    assert page["last_body_page_fill_ratio"] == 0.18
    assert not page["last_body_page_counted"]
    assert "last counted body page fill ratio" in _join(page["errors"])


def test_validate_and_compile_paper_alife_author_block(tmp_path: Path) -> None:
    # accepts canonical author input file
    root = _alife_root(
        _sub(tmp_path, "canonical"),
        valid_alife_tex().replace(f"\\author{{{ALIFE_AUTHORS}}}\n", "\\author{\\input{paper_authors.tex}}\n"),
    )
    _write_authors(root)
    report = _compile(root, page_texts=pages(8, REFS_PAGE, SUPPLEMENT_PAGE), fill_ratio=0.92)
    assert report["ok"]
    assert report["venue_validation"]["hard_errors"] == []
    # fails on anonymous author block
    root = _alife_root(
        _sub(tmp_path, "anonymous"),
        valid_alife_tex().replace(
            ALIFE_AUTHORS,
            "Anonymous Author$^{1}$ \\and Real Person$^{1}$ \\\\\n\\mbox{}\\\\\n$^{1}$Institution",
        ),
    )
    with pytest.raises(RuntimeError, match="venue validation failed"):
        _compile(root, compile=never_compile)
    report = _report(root)
    assert not report["ok"]
    assert report["venue_validation"]["checked"]
    hard_errors = _join(report["venue_validation"]["hard_errors"])
    assert "Anonymous Author" in hard_errors
    assert "Institution" in hard_errors
    # fails when author block drifts from canonical file
    root = _alife_root(_sub(tmp_path, "drift"), valid_alife_tex().replace("Ben Mockley", "Someone Else"))
    _write_authors(root)
    with pytest.raises(RuntimeError, match="venue validation failed"):
        _compile(root, compile=never_compile)
    report = _report(root)
    assert not report["ok"]
    assert "does not match canonical `./paper_authors.tex`" in _join(
        report["venue_validation"]["hard_errors"]
    )


def test_validate_and_compile_paper_venue_framing_and_brief_requirements(tmp_path: Path) -> None:
    # reports soft ALIFE framing warnings
    root = _alife_root(
        _sub(tmp_path, "framing"),
        valid_alife_tex()
        .replace("Artificial Life Example of Adaptive Computation", "Example")
        .replace(
            "Artificial life summary about adaptive computation and biologically plausible learning.\n",
            "Summary.\n",
        )
        .replace(
            "Artificial life intro on life-like adaptive systems and local learning \\cite{RefKey2024}.\n",
            "Intro \\cite{RefKey2024}.\n",
        )
        .replace(
            "Artificial life conclusion on adaptive computation under biologically plausible constraints.\n",
            "Conclusion about edge devices and scientific computing.\n",
        ),
    )
    report = _compile(root, page_texts=pages(8, REFS_PAGE, SUPPLEMENT_PAGE), fill_ratio=0.92)
    assert report["ok"]
    assert report["venue_validation"]["checked"]
    warnings = _join(report["venue_validation"]["soft_warnings"])
    assert "title lacks explicit ALIFE-facing framing language" in warnings
    assert "abstract lacks explicit ALIFE-facing framing language" in warnings
    assert "discussion/conclusion drifts into generic deployment or application framing" in warnings
    # accepts "Conclusions" heading for venue warnings
    root = _alife_root(
        _sub(tmp_path, "conclusions"),
        valid_alife_tex()
        .replace("\\section{Conclusion}\n", "\\section{Conclusions}\n")
        .replace(
            "Artificial life conclusion on adaptive computation under biologically plausible constraints.\n",
            "Artificial life conclusion on robustness and adaptive computation.\n",
        ),
    )
    report = _compile(root, page_texts=pages(8, REFS_PAGE, SUPPLEMENT_PAGE), fill_ratio=0.92)
    assert report["ok"]
    assert report["venue_validation"]["soft_warnings"] == []
    # uses venue brief for non-ALIFE author requirements
    root = _sub(tmp_path, "brief")
    write_valid_artifacts(root)
    (root / "VENUE.md").write_text(
        "# Venue\nSubmissions are NOT anonymous. Include real author names and affiliations.\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="venue validation failed"):
        _compile(root, compile=never_compile)
    report = _report(root)
    assert not report["ok"]
    assert report["venue_validation"]["checked"]
    assert "missing an author block" in _join(report["venue_validation"]["hard_errors"])


# --------------------------------------------------------------------------------------------
# Rendered page helpers: supplement/reference page finders, rendered SI, page budget
# --------------------------------------------------------------------------------------------


def test_find_rendered_page_markers() -> None:
    # find_supplement_start_page
    supplement_cases: list[tuple[str, list[str], int | None]] = [
        (
            "rendered_heading",
            ["Main paper", REFS_PAGE, "Supplementary Information\nExtended evaluation details"],
            3,
        ),
        ("appendix_prefix", ["References", "A Supplementary Information\nExtended details"], 2),
        (
            "letter_spaced_small_caps",
            ["References", "A S UPPLEMENTARY I NFORMATION\nExtended details"],
            2,
        ),
        (
            "heading_split_across_lines",
            ["References", "A S UPPLEMENTARY\nI NFORMATION\nExtended details"],
            None,
        ),
    ]
    for name, page_texts, expected in supplement_cases:
        assert paper_step.find_supplement_start_page(page_texts) == expected, name
    # find_reference_start_page
    reference_cases: list[tuple[str, list[str], int | None]] = [
        ("letter_spaced_small_caps", ["Main paper", "R EFERENCES\n[1] Example"], 2),
        ("split_across_lines", ["Main paper", "R\nEFERENCES\n[1] Example"], None),
    ]
    for name, page_texts, expected in reference_cases:
        assert paper_step.find_reference_start_page(page_texts) == expected, name
    # reference page watermark prefix does not count as body
    watermark_cases = [
        ("watermark_only", "Generated by The Agentic Scientist\nReferences\n[1] Example", False),
        ("split_watermark", "Generated by The\nAgentic Scientist\nReferences\n[1] Example", False),
        (
            "real_body_prefix",
            "Generated by The Agentic Scientist\n"
            "We conclude the decomposition wins decisively in this regime.\nReferences\n[1] Example",
            True,
        ),
    ]
    for name, page_text, expected in watermark_cases:
        assert paper_step.reference_page_has_body_prefix(page_text) is expected, name


def _rendered_si(page_texts: list[str]) -> dict[str, Any]:
    return paper_step.validate_rendered_supplementary_information(page_texts)


def test_validate_rendered_supplementary_information() -> None:
    # rejects missing heading
    report = _rendered_si(["Main paper", REFS_PAGE])
    assert report["rendered_checked"]
    assert not report["rendered_heading_present"]
    assert "missing" in _join(report["hard_errors"])
    # ignores text before heading
    report = _rendered_si(
        [
            REFS_PAGE,
            "A long preface before the trailing heading must not count as SI content. " * 4
            + "\nA Supplementary Information",
        ]
    )
    assert report["rendered_visible_character_count"] == 0
    assert "100 visible characters" in _join(report["hard_errors"])
    # hard error cases
    cases: list[tuple[str, list[str], str | None]] = [
        (
            "accepts_letter_spaced_headings",
            [
                "Main paper",
                "R EFERENCES\n[1] Example",
                "A S UPPLEMENTARY I NFORMATION\n" + "Extended grounded evaluation details. " * 5,
            ],
            None,
        ),
        (
            "requires_new_page_after_references",
            ["Main paper", REFS_PAGE + "\n" + SUPPLEMENT_PAGE],
            "separate page after References",
        ),
        (
            "rejects_bibliography_continuation",
            [
                "References\nMain bibliography starts",
                "Continuation of a bibliography entry\n" + SUPPLEMENT_PAGE,
            ],
            "separate page after References",
        ),
        (
            "allows_supplement_references",
            ["References", SUPPLEMENT_PAGE, "References\n[1] Actual bibliography"],
            None,
        ),
        (
            "excludes_supplement_references",
            [
                "References",
                "A Supplementary Information\nshort",
                "References\n" + "Long supplemental citation entry. " * 8,
            ],
            "100 visible characters",
        ),
    ]
    for name, page_texts, expected_error in cases:
        hard_errors = _rendered_si(page_texts)["hard_errors"]
        if expected_error is None:
            assert hard_errors == [], name
        else:
            assert expected_error in _join(hard_errors), name


def test_validate_page_budget_letter_spaced_references_and_unlimited_format() -> None:
    # locates letter-spaced references
    report = paper_step.validate_page_budget(
        tex_text=valid_alife_tex(),
        pdf_path=UNUSED_PDF,
        page_texts=pages(7, "R EFERENCES\n[1] Example", SUPPLEMENT_PAGE),
    )
    assert report["reference_start_page"] == 8
    assert "could not locate" not in _join(report["errors"])
    # reports supplement for unlimited format
    report = paper_step.validate_page_budget(
        tex_text=valid_tex(),
        pdf_path=UNUSED_PDF,
        page_texts=[
            "Main paper",
            REFS_PAGE,
            "A Supplementary Information\nExtended grounded details for reproducibility.",
        ],
    )
    assert not report["checked"]
    assert report["total_pages"] == 3
    assert report["supplement_start_page"] == 3
    assert report["supplement_pages"] == 1


def _iclr_budget(page_texts: list[str]) -> dict[str, Any]:
    return paper_step.validate_page_budget(
        tex_text=valid_iclr_tex(), pdf_path=UNUSED_PDF, page_texts=page_texts
    )


def test_validate_page_budget_iclr_minimum_and_fill_ratio() -> None:
    # rejects short circle-packing layout
    report = _iclr_budget(
        pages(
            5,
            "References\n[1] First reference",
            "[2] Second reference",
            SUPPLEMENT_PAGE,
            "Additional supplementary details",
        )
    )
    assert report["checked"]
    assert report["paper_format"] == "iclr"
    assert report["min_body_pages"] == 8
    assert report["min_last_body_page_fill_ratio"] == 0.75
    assert report["max_body_pages"] == 9
    assert report["exclude_references"]
    assert report["total_pages"] == 9
    assert report["reference_start_page"] == 6
    assert report["supplement_start_page"] == 8
    assert report["supplement_pages"] == 2
    assert report["body_pages"] == 5
    assert "page minimum not met" in _join(report["errors"])
    # rejects seven body pages
    report = _iclr_budget(pages(7, REFS_PAGE, SUPPLEMENT_PAGE))
    errors = _join(report["errors"])
    assert report["body_pages"] == 7
    assert "page minimum not met" in errors
    assert "final body page counts only when at least 75% full" in errors
    # accepts full eighth body page
    with patch.object(
        paper_step, "measure_last_body_page_fill_ratio", return_value=0.75
    ) as measure_fill:
        report = _iclr_budget(pages(8, REFS_PAGE, SUPPLEMENT_PAGE))
    measure_fill.assert_called_once_with(
        UNUSED_PDF, last_body_page_number=8, reference_start_page=9, body_columns=1
    )
    assert report["body_pages"] == 8
    assert report["last_body_page_fill_ratio"] == 0.75
    assert report["last_body_page_counted"]
    assert report["errors"] == []
    # rejects sparse eighth body page
    with patch.object(paper_step, "measure_last_body_page_fill_ratio", return_value=0.74):
        report = _iclr_budget(pages(8, REFS_PAGE, SUPPLEMENT_PAGE))
    errors = _join(report["errors"])
    assert report["body_pages"] == 7
    assert report["last_body_page_fill_ratio"] == 0.74
    assert not report["last_body_page_counted"]
    assert "below required 0.75" in errors
    assert "page minimum not met" in errors


def test_validate_page_budget_iclr_maximum() -> None:
    # accepts nine body pages without measuring fill
    with patch.object(paper_step, "measure_last_body_page_fill_ratio") as measure_fill:
        report = _iclr_budget(pages(9, REFS_PAGE, SUPPLEMENT_PAGE))
    measure_fill.assert_not_called()
    assert report["body_pages"] == 9
    assert report["errors"] == []
    # rejects ten body pages
    report = _iclr_budget(pages(10, REFS_PAGE, SUPPLEMENT_PAGE))
    assert report["body_pages"] == 10
    assert "page limit exceeded" in _join(report["errors"])


# --------------------------------------------------------------------------------------------
# validation_hard_error_fingerprint / validation_failure_fingerprint
# --------------------------------------------------------------------------------------------


def _feedback(root: Path, error: str = "paper is invalid", **extra: Any) -> Path:
    feedback_path = root / "validation_feedback.json"
    write_json(feedback_path, {"hard_errors": [error], **extra})
    return feedback_path


def _fingerprint(root: Path) -> str | None:
    return paper_step.validation_failure_fingerprint(root / "validation_feedback.json", root)


def test_validation_hard_error_fingerprint_normalizes_and_requires_strings(tmp_path: Path) -> None:
    # normalizes order and spacing
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_json(first, {"hard_errors": ["missing  heading", "bad references"]})
    write_json(second, {"hard_errors": ["bad references", "missing heading"]})
    first_fingerprint = paper_step.validation_hard_error_fingerprint(first)
    assert first_fingerprint is not None
    assert first_fingerprint == paper_step.validation_hard_error_fingerprint(second)
    # requires string errors
    feedback_path = tmp_path / "feedback.json"
    write_json(feedback_path, {"hard_errors": ["valid", 1]})
    assert paper_step.validation_hard_error_fingerprint(feedback_path) is None


def _mutating_paper_input(root: Path) -> Callable[[], object]:
    (root / "paper.tex").write_text("before", encoding="utf-8")
    return lambda: (root / "paper.tex").write_text("after", encoding="utf-8")


def _mutating_direct_validator_input(root: Path) -> Callable[[], object]:
    (root / "paper.tex").write_text("paper", encoding="utf-8")
    scope_path = root / "method_scope_audit.json"
    scope_path.write_text('{"status":"before"}', encoding="utf-8")
    return lambda: scope_path.write_text('{"status":"after"}', encoding="utf-8")


def _mutating_referenced_figure(root: Path) -> Callable[[], object]:
    (root / "figures").mkdir()
    figure_path = root / "figures" / "result.png"
    figure_path.write_bytes(b"before")
    (root / "paper.tex").write_text(r"\includegraphics{figures/result.png}", encoding="utf-8")
    return lambda: figure_path.write_bytes(b"after")


def _mutating_class_macro_logo(root: Path) -> Callable[[], object]:
    (root / "paper.tex").write_text(
        r"\documentclass{venue_format}\begin{document}Paper\end{document}", encoding="utf-8"
    )
    (root / "venue_format.cls").write_text(
        r"\includegraphics{venue_logo.png}"
        r"\newcommand{\labelasset}[1]{\includegraphics{figures/labels/#1.pdf}}",
        encoding="utf-8",
    )
    logo_path = root / "venue_logo.png"
    logo_path.write_bytes(b"before")
    return lambda: logo_path.write_bytes(b"after")


def _mutating_local_bibliography_style(root: Path) -> Callable[[], object]:
    (root / "paper.tex").write_text(r"\bibliographystyle{paper}", encoding="utf-8")
    style_path = root / "paper.bst"
    style_path.write_text("before", encoding="utf-8")
    return lambda: style_path.write_text("after", encoding="utf-8")


def _mutating_imported_style(root: Path) -> Callable[[], object]:
    (root / "paper.tex").write_text(r"\input{sub/refs}", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "refs.tex").write_text(r"\bibliographystyle{paper}", encoding="utf-8")
    style_path = root / "paper.bst"
    style_path.write_text("before", encoding="utf-8")
    return lambda: style_path.write_text("after", encoding="utf-8")


def _mutating_commented_style(root: Path) -> Callable[[], object]:
    (root / "paper.tex").write_text("% \\bibliographystyle{unused}\nPaper", encoding="utf-8")
    style_path = root / "unused.bst"
    style_path.write_text("before", encoding="utf-8")
    return lambda: style_path.write_text("after", encoding="utf-8")


def _mutating_concept_prompt_log(root: Path) -> Callable[[], object]:
    (root / "paper.tex").write_text("paper", encoding="utf-8")
    (root / "figures").mkdir()
    (root / "figures" / "concept.png").write_bytes(b"png")
    prompt_log = root / "figures" / "concept.prompts.json"
    prompt_log.write_text('{"status":"before"}', encoding="utf-8")
    write_manifest(root, [concept_row("concept", pdf=False, panels=[{"panel_id": "main", "content": "Overview"}])])
    return lambda: prompt_log.write_text('{"status":"after"}', encoding="utf-8")


def _mutating_generated_reports(root: Path) -> Callable[[], object]:
    _feedback(root, "page  fill is too low", generated_at="before")
    (root / "paper.tex").write_text("paper", encoding="utf-8")
    report_path = root / "validation_report.json"
    report_path.write_text('{"generated_at":"before"}', encoding="utf-8")

    def mutate() -> None:
        _feedback(root, "page fill is too low", generated_at="after")
        report_path.write_text('{"generated_at":"after"}', encoding="utf-8")

    return mutate


def _mutating_unrelated_memory(root: Path) -> Callable[[], object]:
    (root / "paper.tex").write_text("paper", encoding="utf-8")
    memory_path = root / "memory.md"
    memory_path.write_text("before", encoding="utf-8")
    return lambda: memory_path.write_text("after", encoding="utf-8")


FINGERPRINT_INPUT_CASES: list[tuple[str, Callable[[Path], Callable[[], object]], bool]] = [
    ("changes_with_paper_input", _mutating_paper_input, True),
    ("changes_with_direct_validator_input", _mutating_direct_validator_input, True),
    ("changes_with_referenced_figure", _mutating_referenced_figure, True),
    ("ignores_class_macro_figure_template", _mutating_class_macro_logo, True),
    ("changes_with_local_bibliography_style", _mutating_local_bibliography_style, True),
    ("resolves_imported_style_from_root", _mutating_imported_style, True),
    ("ignores_commented_bibliography_style", _mutating_commented_style, False),
    ("changes_with_concept_prompt_log", _mutating_concept_prompt_log, True),
    ("ignores_generated_reports", _mutating_generated_reports, False),
    ("ignores_unrelated_memory", _mutating_unrelated_memory, False),
]


def test_validation_failure_fingerprint_tracks_validation_inputs(tmp_path: Path) -> None:
    for name, setup, expect_changed in FINGERPRINT_INPUT_CASES:
        root = _sub(tmp_path, name)
        _feedback(root)
        mutate = setup(root)
        before = _fingerprint(root)
        mutate()
        after = _fingerprint(root)
        assert before is not None, name
        if expect_changed:
            assert before != after, name
        else:
            assert before == after, name


def test_validation_failure_fingerprint_limits_and_input_guards(tmp_path: Path) -> None:
    # rejects inputs over limits
    limit_cases: list[tuple[str, str, dict[str, Any] | None, str, int]] = [
        ("oversized_input", "too large", None, "_MAX_AUDIT_TEX_BYTES", 4),
        (
            "bounds_bibliography_styles",
            r"\bibliographystyle{first}\bibliographystyle{second}",
            None,
            "_MAX_VALIDATION_BIBLIOGRAPHY_STYLE_COUNT",
            1,
        ),
        ("bounds_manifest_rows", "paper", {"figures": [{}, {}]}, "_MAX_AUDIT_FIGURE_COUNT", 1),
        (
            "bounds_conditional_parsing",
            r"\iffalse\else" * 3 + r"\fi" * 3,
            None,
            "_MAX_VALIDATION_TEX_CONDITIONAL_TOKENS",
            4,
        ),
    ]
    for name, tex, manifest, limit_name, limit in limit_cases:
        root = _sub(tmp_path, name)
        _feedback(root)
        (root / "paper.tex").write_text(tex, encoding="utf-8")
        if manifest is not None:
            write_json(root / "figure_manifest.json", manifest)
        with (
            patch(f"src.orchestrator.paper_evidence_audit.{limit_name}", limit),
            pytest.raises(RuntimeError, match="Could not fingerprint paper validation inputs"),
        ):
            _fingerprint(root)
    # ignores commented conditionals
    root = _sub(tmp_path, "commented_conditionals")
    _feedback(root)
    (root / "paper.tex").write_text("% \\iffalse\n% \\iffalse\nPaper", encoding="utf-8")
    with patch("src.orchestrator.paper_evidence_audit._MAX_VALIDATION_TEX_CONDITIONAL_TOKENS", 1):
        assert _fingerprint(root) is not None
    # requires validation inputs
    root = _sub(tmp_path, "no_inputs")
    _feedback(root, "page fill is too low")
    assert _fingerprint(root) is None
    # rejects symlinked input
    base = _sub(tmp_path, "symlinked")
    root = base / "paper"
    root.mkdir()
    _feedback(root, "page fill is too low")
    outside = base / "outside.tex"
    outside.write_text("paper", encoding="utf-8")
    (root / "paper.tex").symlink_to(outside)
    with pytest.raises(RuntimeError, match="Could not fingerprint paper validation inputs"):
        _fingerprint(root)


# --------------------------------------------------------------------------------------------
# PDF layout geometry
# --------------------------------------------------------------------------------------------


def _fragment(text: str, x: float, y: float) -> dict[str, Any]:
    return {"text": text, "x": x, "y": y, "font_size": 12.0}


def _layout(fragments: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"page_height": 800.0, "page_width": 600.0, "fragments": fragments, **extra}


TWO_COLUMN_PAGE = [
    _fragment("Left top", 54.0, 700.0),
    _fragment("Left bottom", 54.0, 100.0),
    _fragment("Right top", 330.0, 700.0),
    _fragment("Right bottom", 330.0, 100.0),
]


def test_compute_last_body_page_fill_ratio() -> None:
    # ignores reference region
    page_layouts = [
        _layout([_fragment("Main content page 1", 54.0, 700.0), _fragment("Body text", 54.0, 100.0)]),
        _layout(
            [
                _fragment("Final discussion paragraph.", 54.0, 700.0),
                _fragment("One more body line.", 54.0, 640.0),
                _fragment("References", 54.0, 300.0),
                _fragment("[1] Example", 54.0, 260.0),
            ]
        ),
    ]
    fill_ratio = paper_step.compute_last_body_page_fill_ratio(
        page_layouts, last_body_page_number=2, reference_start_page=2
    )
    assert fill_ratio == pytest.approx(72.0 / 612.0)
    # two columns: penalizes empty second column, counts figure occupancy
    two_column_cases: list[tuple[str, dict[str, Any], dict[str, Any], float]] = [
        (
            "penalizes_empty_second_column",
            _layout(TWO_COLUMN_PAGE),
            _layout([_fragment("Left top", 54.0, 700.0), _fragment("Left bottom", 54.0, 100.0)]),
            0.5,
        ),
        (
            "counts_figure_occupancy",
            _layout(TWO_COLUMN_PAGE, graphics=[]),
            _layout(
                [_fragment("Figure 1: Caption", 54.0, 400.0), *TWO_COLUMN_PAGE[1:]],
                graphics=[{"x_min": 54.0, "x_max": 292.0, "y_min": 412.0, "y_max": 712.0}],
            ),
            1.0,
        ),
    ]
    for name, first_page, last_page, expected in two_column_cases:
        fill_ratio = paper_step.compute_last_body_page_fill_ratio(
            [first_page, last_page], last_body_page_number=2, reference_start_page=None, body_columns=2
        )
        assert fill_ratio == pytest.approx(expected), name


def test_pdf_geometry_helpers() -> None:
    # fragment y position prefers text matrix over zero user matrix
    y_value = paper_step._fragment_y_position(
        user_matrix=[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        text_matrix=[1.0, 0.0, 0.0, 1.0, 54.0, 728.037],
    )
    assert y_value == 728.037
    # graphic bounds from matrix handles rotation
    graphic_bounds = paper_step._graphic_bounds_from_matrix([0.0, 10.0, -20.0, 0.0, 100.0, 200.0])
    assert graphic_bounds == {"x_min": 80.0, "x_max": 100.0, "y_min": 200.0, "y_max": 210.0}
    # graphic bounds from matrix rejects invalid transforms
    for name, matrix in (
        ("nonfinite_values", [1.0, 0.0, 0.0, float("nan"), 10.0, 20.0]),
        ("singular_transform", [1000.0, 1000.0, 1000.0, 1000.0, 0.0, 0.0]),
        ("huge_finite_transform", [1e308, 0.0, 0.0, 1e308, 1e308, 1e308]),
    ):
        assert paper_step._graphic_bounds_from_matrix(matrix) is None, name
    # body text bounds ignore off-page graphics
    bounds = paper_step._body_text_bounds(
        {
            "page_width": 600.0,
            "page_height": 800.0,
            "page_x_min": 0.0,
            "page_x_max": 600.0,
            "page_y_min": 0.0,
            "page_y_max": 800.0,
            "fragments": [_fragment("Top", 54.0, 400.0), _fragment("Bottom", 54.0, 100.0)],
            "graphics": [{"x_min": 54.0, "x_max": 250.0, "y_min": 900.0, "y_max": 1000.0}],
        },
        exclude_reference_region=False,
    )
    assert bounds == (412.0, 100.0)


def _image_page(operations: list[tuple[list[Any], bytes]]) -> Mock:
    image_reference = Mock()
    image_reference.get_object.return_value = {"/Subtype": "/Image"}
    page = Mock()
    page.get.side_effect = lambda key: (
        {"/XObject": {"/Im1": image_reference}} if key == "/Resources" else None
    )
    contents = Mock()
    contents.operations = operations
    page.get_contents.return_value = contents
    return page


def test_read_pdf_page_layouts_and_graphics_limits() -> None:
    # read_pdf_page_layouts captures top-level inherited images only
    image_reference = Mock()
    image_reference.get_object.return_value = {"/Subtype": "/Image"}
    form_reference = Mock()
    form_reference.get_object.return_value = {"/Subtype": "/Form"}
    resources = {"/XObject": {"/Im1": image_reference, "/Fm1": form_reference}}
    parent = Mock()
    parent.get.side_effect = lambda key: resources if key == "/Resources" else None
    parent_reference = Mock()
    parent_reference.get_object.return_value = parent
    page = Mock()
    page.get.side_effect = lambda key: parent_reference if key == "/Parent" else None
    page.mediabox.left = 0.0
    page.mediabox.right = 600.0
    page.mediabox.bottom = 0.0
    page.mediabox.top = 800.0
    contents = Mock()
    contents.operations = [
        ([1.0, 0.0, 0.0, 1.0, 50.0, 600.0], b"cm"),
        ([], b"q"),
        ([200.0, 0.0, 0.0, 100.0, 0.0, 0.0], b"cm"),
        (["/Im1"], b"Do"),
        ([], b"Q"),
        (["/Fm1"], b"Do"),
    ]
    page.get_contents.return_value = contents
    page.extract_text.return_value = "Body"

    with patch.object(paper_step, "PdfReader", return_value=Mock(pages=[page])):
        layouts = paper_step.read_pdf_page_layouts(Path("paper.pdf"))

    assert layouts[0]["graphics"] == [{"x_min": 50.0, "x_max": 250.0, "y_min": 600.0, "y_max": 700.0}]
    # _read_page_graphics fails closed on operation / state-depth limits
    limit_cases: list[tuple[str, list[tuple[list[Any], bytes]], dict[str, int]]] = [
        ("operation_limit", [([], b"n"), (["/Im1"], b"Do")], {"max_operations": 1}),
        ("state_depth", [([], b"q"), ([], b"q"), (["/Im1"], b"Do")], {"max_state_depth": 1}),
    ]
    for name, operations, limits in limit_cases:
        assert paper_step._read_page_graphics(_image_page(operations), **limits) == [], name
