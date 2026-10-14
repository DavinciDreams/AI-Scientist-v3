"""Typed figure-manifest contract and generated JSON schema helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

FigureType = Literal[
    "grouped_bar",
    "bar",
    "line",
    "box",
    "violin",
    "scatter",
    "heatmap",
    "table_matrix",
    "table",
    "composite",
    "diagram",
]
SUPPORTED_FIGURE_TYPES = frozenset(get_args(FigureType))

FigurePlacement = Literal["column", "full_width"]
FigureGenerator = Literal["generate-concept-figure", "matplotlib", "tikz", "native_latex", "manual"]
SUPPORTED_FIGURE_GENERATORS = frozenset(get_args(FigureGenerator))


def is_documented_failed_render(
    generator: str | None,
    attempted_generator: str | None,
    prompt_log_path: str | None,
    fallback_reason: str | None,
) -> bool:
    """Return whether a PNG-only row explicitly records a failed render."""
    return (
        generator is None
        and attempted_generator == "generate-concept-figure"
        and isinstance(prompt_log_path, str)
        and bool(prompt_log_path.strip())
        and isinstance(fallback_reason, str)
        and bool(fallback_reason.strip())
    )


@dataclass(frozen=True)
class NormalizedFigureManifestFigure:
    """Internal figure-manifest row used by validation logic."""

    figure_id: str
    figure_type: str
    path_pdf: str
    path_png: str
    n_panels: int
    panels_per_row: int
    categories_per_panel: list[int]
    series_per_panel: list[int]
    max_label_chars: int | None
    uses_abbrev_labels: bool | None
    uses_progression: bool
    source_ids: list[str]
    font_sizes: dict[str, float] | None
    min_font_size_pt: float | None
    placement: str | None
    intended_latex_width_fraction: float | None
    estimated_min_rendered_font_pt: float | None
    generator: str | None
    attempted_generator: str | None
    prompt_log_path: str | None
    fallback_reason: str | None


class FigureFontSizes(BaseModel):
    """Explicit font sizes for the legacy manifest contract."""

    model_config = ConfigDict(extra="forbid")

    title: float
    axis: float
    ticks: float
    legend: float


class LegacyFigureManifestFigure(BaseModel):
    """Original flat figure-manifest entry shape."""

    model_config = ConfigDict(extra="forbid")

    figure_id: str = Field(min_length=1)
    path_pdf: str = ""
    path_png: str = Field(min_length=1)
    figure_type: FigureType
    n_panels: int = Field(gt=0)
    panels_per_row: int = Field(gt=0)
    categories_per_panel: list[int]
    series_per_panel: list[int]
    max_label_chars: int = Field(gt=0)
    font_sizes: FigureFontSizes
    uses_abbrev_labels: bool
    uses_progression: bool
    source_ids: list[str] = Field(default_factory=list)
    generator: FigureGenerator | None = None
    attempted_generator: FigureGenerator | None = None
    prompt_log_path: str | None = None
    fallback_reason: str | None = None
    placement: FigurePlacement | None = None
    intended_latex_width_fraction: float | None = Field(default=None, gt=0, le=1)
    estimated_min_rendered_font_pt: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_panel_lengths(self) -> "LegacyFigureManifestFigure":
        if len(self.categories_per_panel) != self.n_panels:
            raise ValueError("categories_per_panel must have length n_panels")
        if len(self.series_per_panel) != self.n_panels:
            raise ValueError("series_per_panel must have length n_panels")
        if (
            self.generator != "generate-concept-figure"
            and not self.path_pdf.strip()
            and not is_documented_failed_render(
                self.generator,
                self.attempted_generator,
                self.prompt_log_path,
                self.fallback_reason,
            )
        ):
            raise ValueError(
                "PDF artifact path is required unless this is a concept figure "
                "or documented failed render"
            )
        return self

    def normalize(self) -> NormalizedFigureManifestFigure:
        return NormalizedFigureManifestFigure(
            figure_id=self.figure_id,
            figure_type=self.figure_type,
            path_pdf=self.path_pdf,
            path_png=self.path_png,
            n_panels=self.n_panels,
            panels_per_row=self.panels_per_row,
            categories_per_panel=list(self.categories_per_panel),
            series_per_panel=list(self.series_per_panel),
            max_label_chars=self.max_label_chars,
            uses_abbrev_labels=self.uses_abbrev_labels,
            uses_progression=self.uses_progression,
            source_ids=list(self.source_ids),
            font_sizes={
                "title": float(self.font_sizes.title),
                "axis": float(self.font_sizes.axis),
                "ticks": float(self.font_sizes.ticks),
                "legend": float(self.font_sizes.legend),
            },
            min_font_size_pt=None,
            placement=self.placement,
            intended_latex_width_fraction=self.intended_latex_width_fraction,
            estimated_min_rendered_font_pt=self.estimated_min_rendered_font_pt,
            generator=self.generator,
            attempted_generator=self.attempted_generator,
            prompt_log_path=self.prompt_log_path,
            fallback_reason=self.fallback_reason,
        )


class RichFigureFiles(BaseModel):
    """Nested artifact paths used by the richer manifest shape."""

    model_config = ConfigDict(extra="forbid")

    pdf: str = ""
    png: str = Field(min_length=1)


class RichFigurePanel(BaseModel):
    """Panel-level metadata for the richer manifest shape."""

    model_config = ConfigDict(extra="forbid")

    panel_id: str | None = None
    label: str | None = None
    task: str | None = None
    metric: str | None = None
    threshold: str | None = None
    category_count: int | None = Field(default=None, gt=0)
    series_count: int | None = Field(default=None, gt=0)
    categories: list[str] | None = None
    content: str | None = None
    y_range: list[float] | None = None


class RichFigureManifestFigure(BaseModel):
    """Richer manifest shape used by newer standalone paper runs."""

    model_config = ConfigDict(extra="forbid")

    figure_id: str = Field(min_length=1)
    figure_type: FigureType
    files: RichFigureFiles
    panel_count: int = Field(gt=0)
    panels: list[RichFigurePanel]
    placement: FigurePlacement | None = None
    uses_progression: bool
    source_ids: list[str] = Field(default_factory=list)
    label: str | None = None
    caption_short: str | None = None
    required: bool | None = None
    note: str | None = None
    generator: FigureGenerator | None = None
    attempted_generator: FigureGenerator | None = None
    prompt_log_path: str | None = None
    fallback_reason: str | None = None
    min_font_size_pt: float | None = Field(default=None, gt=0)
    font_sizes: FigureFontSizes | None = None
    panels_per_row: int | None = Field(default=None, gt=0)
    max_label_chars: int | None = Field(default=None, gt=0)
    uses_abbrev_labels: bool | None = None
    intended_latex_width_fraction: float | None = Field(default=None, gt=0, le=1)
    estimated_min_rendered_font_pt: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_panel_lengths(self) -> "RichFigureManifestFigure":
        if len(self.panels) != self.panel_count:
            raise ValueError("panels must have length panel_count")
        if self.font_sizes is None and self.min_font_size_pt is None:
            raise ValueError("either font_sizes or min_font_size_pt must be provided")
        if (
            self.generator != "generate-concept-figure"
            and not self.files.pdf.strip()
            and not is_documented_failed_render(
                self.generator,
                self.attempted_generator,
                self.prompt_log_path,
                self.fallback_reason,
            )
        ):
            raise ValueError(
                "PDF artifact path is required unless this is a concept figure "
                "or documented failed render"
            )
        return self

    def normalize(self) -> NormalizedFigureManifestFigure:
        categories_per_panel: list[int] = []
        series_per_panel: list[int] = []
        inferred_max_label_chars = 0
        for panel in self.panels:
            if panel.category_count is not None:
                categories_per_panel.append(panel.category_count)
            elif isinstance(panel.categories, list) and panel.categories:
                categories_per_panel.append(len(panel.categories))
            else:
                categories_per_panel.append(1)

            if panel.series_count is not None:
                series_per_panel.append(panel.series_count)
            else:
                series_per_panel.append(1)

            if isinstance(panel.categories, list):
                inferred_max_label_chars = max(
                    inferred_max_label_chars,
                    max((len(str(item)) for item in panel.categories), default=0),
                )

        font_sizes: dict[str, float] | None = None
        if self.font_sizes is not None:
            font_sizes = {
                "title": float(self.font_sizes.title),
                "axis": float(self.font_sizes.axis),
                "ticks": float(self.font_sizes.ticks),
                "legend": float(self.font_sizes.legend),
            }

        return NormalizedFigureManifestFigure(
            figure_id=self.figure_id,
            figure_type=self.figure_type,
            path_pdf=self.files.pdf,
            path_png=self.files.png,
            n_panels=self.panel_count,
            panels_per_row=self.panels_per_row or min(self.panel_count, 3),
            categories_per_panel=categories_per_panel,
            series_per_panel=series_per_panel,
            max_label_chars=self.max_label_chars or (inferred_max_label_chars or None),
            uses_abbrev_labels=self.uses_abbrev_labels,
            uses_progression=self.uses_progression,
            source_ids=list(self.source_ids),
            font_sizes=font_sizes,
            min_font_size_pt=float(self.min_font_size_pt) if self.min_font_size_pt is not None else None,
            placement=self.placement,
            intended_latex_width_fraction=float(self.intended_latex_width_fraction) if self.intended_latex_width_fraction is not None else None,
            estimated_min_rendered_font_pt=float(self.estimated_min_rendered_font_pt) if self.estimated_min_rendered_font_pt is not None else None,
            generator=self.generator,
            attempted_generator=self.attempted_generator,
            prompt_log_path=self.prompt_log_path,
            fallback_reason=self.fallback_reason,
        )


FigureManifestRow = LegacyFigureManifestFigure | RichFigureManifestFigure


class FigureManifestDocument(BaseModel):
    """Top-level figure-manifest document."""

    model_config = ConfigDict(extra="forbid")

    figures: list[FigureManifestRow] = Field(min_length=1)


def build_figure_manifest_json_schema() -> dict[str, Any]:
    """Generate the canonical JSON schema for figure_manifest.json."""
    schema = FigureManifestDocument.model_json_schema()
    concept_condition = {
        "properties": {"generator": {"const": "generate-concept-figure"}},
        "required": ["generator"],
    }
    failed_render_condition = {
        "properties": {
            "generator": {"type": "null"},
            "attempted_generator": {"const": "generate-concept-figure"},
            "prompt_log_path": {"type": "string", "minLength": 1},
            "fallback_reason": {"type": "string", "minLength": 1},
        },
        "required": [
            "generator",
            "attempted_generator",
            "prompt_log_path",
            "fallback_reason",
        ],
    }
    png_only_condition = {"anyOf": [concept_condition, failed_render_condition]}
    schema["$defs"]["LegacyFigureManifestFigure"].setdefault("allOf", []).append(
        {
            "if": png_only_condition,
            "else": {
                "properties": {"path_pdf": {"minLength": 1}},
                "required": ["path_pdf"],
            },
        }
    )
    schema["$defs"]["RichFigureManifestFigure"].setdefault("allOf", []).append(
        {
            "if": png_only_condition,
            "else": {
                "properties": {
                    "files": {
                        "properties": {"pdf": {"minLength": 1}},
                        "required": ["pdf"],
                    }
                }
            },
        }
    )
    return schema


def normalize_figure_manifest(payload: dict[str, Any]) -> list[NormalizedFigureManifestFigure]:
    """Parse and normalize figure-manifest rows from either supported schema shape."""
    document = FigureManifestDocument.model_validate(payload)
    return [row.normalize() for row in document.figures]


def write_figure_manifest_json_schema(path: Path) -> None:
    """Write the generated JSON schema to disk."""
    schema = build_figure_manifest_json_schema()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def format_figure_manifest_validation_error(exc: ValidationError) -> str:
    """Render pydantic validation errors into one compact string."""
    messages: list[str] = []
    for item in exc.errors():
        loc = ".".join(str(part) for part in item.get("loc", ()))
        msg = str(item.get("msg", "invalid value"))
        if loc:
            messages.append(f"{loc}: {msg}")
        else:
            messages.append(msg)
    return "; ".join(messages)


if __name__ == "__main__":
    default_path = Path("src/agents/skills/validate-paper/templates/figure_manifest.schema.json")
    write_figure_manifest_json_schema(default_path)
    print(default_path)
