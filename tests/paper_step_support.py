"""Shared factories for the paper_step test files."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

SI_BODY = (
    "We report additional grounded evaluation details, implementation settings, and "
    "failure-analysis context that support reproducibility without carrying the central claim.\n"
)
ALIFE_AUTHORS = (
    "Ada Testwright$^{1}$ \\and Ben Mockley$^{1}$ \\\\\n\\mbox{}\\\\\n$^{1}$Fictional Institute"
)
REFS_PAGE = "References\n[1] Example"
SUPPLEMENT_PAGE = (
    "A Supplementary Information\nExtended grounded evaluation and implementation details "
    "support reproducibility, including additional controls, robustness checks, limitations, "
    "and failure analysis."
)
SKILL_CLAIM = ".codex/skills/payload/claim.tex"


def valid_tex(*, include_graphics: str | None = None) -> str:
    graphics_package = "\\usepackage{graphicx}\n" if include_graphics else ""
    graphics_block = (
        f"\\begin{{figure}}\n\\includegraphics{{{include_graphics}}}\n"
        "\\caption{Example figure.}\n\\end{figure}\n"
        if include_graphics
        else ""
    )
    return (
        "\\documentclass{article}\n"
        f"{graphics_package}"
        "\\begin{document}\n"
        "\\begin{abstract}\nSummary.\n\\end{abstract}\n"
        "\\section{Introduction}\nIntro \\cite{RefKey2024}.\n"
        "\\section{Related Work}\nPrior art.\n"
        "\\section{Method}\nMethod.\n"
        "\\section{Results}\n"
        f"{graphics_block}"
        "Results.\n"
        "\\section{Conclusion}\nDone.\n"
        "\\clearpage\n"
        "\\bibliographystyle{plain}\n"
        "\\bibliography{references}\n"
        "\\clearpage\n\\appendix\n"
        "\\section{Supplementary Information}\n"
        "\\subsection{Extended evaluation details}\n"
        f"{SI_BODY}"
        "\\end{document}\n"
    )


def valid_alife_tex() -> str:
    return (
        "\\documentclass{article}\n"
        "\\usepackage{natbib,alifeconf}\n"
        "\\title{Artificial Life Example of Adaptive Computation}\n"
        f"\\author{{{ALIFE_AUTHORS}}}\n"
        "\\begin{document}\n\\maketitle\n"
        "\\begin{abstract}\n"
        "Artificial life summary about adaptive computation and biologically plausible learning.\n"
        "\\end{abstract}\n"
        "\\section{Introduction}\n"
        "Artificial life intro on life-like adaptive systems and local learning \\cite{RefKey2024}.\n"
        "\\section{Related Work}\nPrior art.\n"
        "\\section{Method}\nMethod.\n"
        "\\section{Results}\nResults.\n"
        "\\section{Conclusion}\n"
        "Artificial life conclusion on adaptive computation under biologically plausible constraints.\n"
        "\\clearpage\n"
        "\\bibliographystyle{apalike}\n"
        "\\bibliography{references}\n"
        "\\clearpage\n\\appendix\n"
        "\\section{Supplementary Information}\n"
        "\\subsection{Extended evaluation details}\n"
        f"{SI_BODY}"
        "\\end{document}\n"
    )


def valid_iclr_tex() -> str:
    return valid_tex().replace(
        "\\documentclass{article}\n",
        "\\documentclass{article}\n\\usepackage{paper,times}\n\\iclrfinalcopy\n",
        1,
    )


def pages(body: int, *tail: str) -> list[str]:
    return [f"Main content page {idx}" for idx in range(1, body + 1)] + list(tail)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_png(path: Path, *, width: int, height: int) -> None:
    """Header-only PNG (IHDR without checksum) sufficient for size probing."""
    ihdr_payload = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + struct.pack(">I", len(ihdr_payload)) + b"IHDR" + ihdr_payload
    )


def png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(payload, zlib.crc32(chunk_type)) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", checksum)


def write_valid_png(path: Path, *, width: int, height: int, filter_byte: int = 0) -> None:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = b"".join(bytes([filter_byte]) + b"\x00\x00\x00" * width for _ in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(rows))
        + png_chunk(b"IEND", b"")
    )


def write_valid_artifacts(
    root: Path, *, include_graphics: str | None = None, paper_profile: str = "default"
) -> None:
    (root / "paper.tex").write_text(valid_tex(include_graphics=include_graphics), encoding="utf-8")
    (root / "references.bib").write_text(
        "@Article{RefKey2024,\n title={Example Ref}\n}\n", encoding="utf-8"
    )
    write_json(
        root / "paper_data.json",
        {
            "paper_profile": paper_profile,
            "presentation_contract": {
                "forbidden_visible_terms": [
                    "eligible",
                    "correct and eligible",
                    "correctness rate",
                    "worker_01",
                    "worker_02",
                    "combined_score",
                    "rubric",
                ]
            },
            "grounding_sources": {"result.best_correct_score": {"type": "metric", "value": 2.71}},
        },
    )
    write_json(
        root / "references_provenance.json",
        {"references": [{"id": "ref.s2.abc", "citation_key": "RefKey2024"}]},
    )
    write_json(
        root / "claims_grounding.json",
        {
            "claims": [
                {
                    "claim": "Best score reached 2.71.",
                    "source_ids": ["result.best_correct_score"],
                    "citation_keys": ["RefKey2024"],
                }
            ]
        },
    )


def write_strategy(root: Path, specs: list[dict[str, Any]], **extra: Any) -> None:
    write_json(root / "paper_strategy.json", {"figure_specs": specs, **extra})


def write_manifest(root: Path, rows: list[dict[str, Any]]) -> None:
    write_json(root / "figure_manifest.json", {"figures": rows})


def bar_spec(figure_id: str = "overview", **overrides: Any) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "figure_type": "bar",
        "max_panels": 2,
        "max_categories_per_panel": 6,
        "max_series_per_panel": 4,
        "allow_progression": False,
        "required": True,
        **overrides,
    }


def concept_spec(figure_id: str = "ed_overview", **overrides: Any) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "figure_type": "diagram",
        "preferred_generator": "generate-concept-figure",
        "max_panels": 1,
        "max_categories_per_panel": 1,
        "max_series_per_panel": 1,
        "allow_progression": False,
        "required": True,
        **overrides,
    }


def legacy_row(figure_id: str = "overview", **overrides: Any) -> dict[str, Any]:
    """Legacy flat manifest row (path_pdf/path_png, n_panels, font_sizes)."""
    return {
        "figure_id": figure_id,
        "path_pdf": f"figures/{figure_id}.pdf",
        "path_png": f"figures/{figure_id}.png",
        "figure_type": "bar",
        "n_panels": 1,
        "panels_per_row": 1,
        "categories_per_panel": [4],
        "series_per_panel": [2],
        "max_label_chars": 12,
        "uses_abbrev_labels": False,
        "uses_progression": False,
        "font_sizes": {"title": 10, "axis": 9, "ticks": 8, "legend": 8},
        "source_ids": ["result.best_correct_score"],
        **overrides,
    }


def diagram_row(
    figure_id: str = "overview",
    *,
    pdf: bool = True,
    panels: list[dict[str, Any]] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Rich manifest row for a single-panel diagram."""
    files = {"png": f"figures/{figure_id}.png"}
    if pdf:
        files["pdf"] = f"figures/{figure_id}.pdf"
    row = {
        "figure_id": figure_id,
        "figure_type": "diagram",
        "files": files,
        "panel_count": 1,
        "panels": panels or [{"panel_id": "overview", "content": "Method"}],
        "uses_progression": False,
        "min_font_size_pt": 7.0,
    }
    row.update(overrides)
    return row


def concept_row(figure_id: str = "ed_overview", *, pdf: bool = True, **overrides: Any) -> dict:
    row = diagram_row(
        figure_id,
        pdf=pdf,
        panels=[{"panel_id": "architecture", "content": "Conceptual overview"}],
        placement="full_width",
        intended_latex_width_fraction=1.0,
        estimated_min_rendered_font_pt=7.0,
        generator="generate-concept-figure",
        prompt_log_path=f"figures/{figure_id}.prompts.json",
        source_ids=[],
    )
    row.update(overrides)
    return row


def fake_compile(*, paper_dir: Path, tex_name: str = "paper.tex", **_: object) -> list[dict]:
    (paper_dir / "paper.pdf").write_text("pdf", encoding="utf-8")
    return [{"command": ["pdflatex", tex_name], "returncode": 0}]


def never_compile(**_: object) -> list[dict]:
    raise AssertionError("compile_latex should not run when pre-compile validation fails")
