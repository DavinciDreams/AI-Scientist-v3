"""Catalog of paper venues declared under ``configs/venue/<name>.yaml``.

A venue bundles everything the paper stages need to know about one paper
format: the writer brief, the canonical author block, the LaTeX template
directory, the markers that identify a compiled manuscript, the enforced page
budget, and the venue-specific checks it opts into. Adding a venue is adding a
YAML file plus the referenced brief and template files; no Python changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENUE_DIR = REPO_ROOT / "configs" / "venue"
VENUE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

#: Behavioural checks implemented in ``paper_step``; a venue opts in by name.
KNOWN_CHECKS = frozenset({"alife_framing"})
_SPEC_KEYS = frozenset(
    {
        "name",
        "brief",
        "author_block",
        "template_dir",
        "latex_markers",
        "page_budget",
        "checks",
    }
)
_PAGE_BUDGET_KEYS = frozenset(
    {
        "min_body_pages",
        "max_body_pages",
        "min_last_body_page_fill_ratio",
        "body_columns",
        "exclude_references",
    }
)


class VenueSpecError(ValueError):
    """A venue file is missing, malformed, or references missing assets."""


@dataclass(frozen=True)
class PageBudget:
    """Page budget enforced by ``validate_page_budget``."""

    max_body_pages: int
    min_body_pages: int | None = None
    min_last_body_page_fill_ratio: float | None = None
    body_columns: int = 1
    exclude_references: bool = False

    def as_limits(self) -> dict[str, int | float | bool]:
        """Return the mapping shape ``validate_page_budget`` consumes."""
        limits: dict[str, int | float | bool] = {
            "max_body_pages": self.max_body_pages,
            "body_columns": self.body_columns,
            "exclude_references": self.exclude_references,
        }
        if self.min_body_pages is not None:
            limits["min_body_pages"] = self.min_body_pages
        if self.min_last_body_page_fill_ratio is not None:
            limits["min_last_body_page_fill_ratio"] = self.min_last_body_page_fill_ratio
        return limits


@dataclass(frozen=True)
class VenueSpec:
    """One paper venue as declared in ``configs/venue/<name>.yaml``."""

    name: str
    brief: str
    author_block: str | None
    template_dir: str
    latex_markers: tuple[str, ...]
    page_budget: PageBudget | None
    checks: frozenset[str]

    @property
    def brief_path(self) -> Path:
        return REPO_ROOT / self.brief

    @property
    def author_block_path(self) -> Path | None:
        return REPO_ROOT / self.author_block if self.author_block is not None else None

    def matches_tex(self, tex_text: str) -> bool:
        """Report whether *tex_text* carries one of this venue's LaTeX markers."""
        return any(re.search(marker, tex_text, re.IGNORECASE) for marker in self.latex_markers)


def venue_spec_path(name: str) -> Path:
    """Return the YAML path that declares venue *name*."""
    return VENUE_DIR / f"{name}.yaml"


def _require(condition: bool, name: str, message: str) -> None:
    if not condition:
        raise VenueSpecError(f"venue {name!r}: {message}")


def _parse_page_budget(name: str, raw: Any) -> PageBudget | None:
    if raw is None:
        return None
    _require(isinstance(raw, dict), name, "page_budget must be a mapping or null")
    unknown = sorted(set(raw) - _PAGE_BUDGET_KEYS)
    _require(not unknown, name, f"page_budget has unknown keys {unknown}")
    _require("max_body_pages" in raw, name, "page_budget.max_body_pages is required")
    for key in ("max_body_pages", "min_body_pages", "body_columns"):
        value = raw.get(key)
        if value is not None:
            _require(
                type(value) is int and value > 0, name, f"page_budget.{key} must be a positive int"
            )
    ratio = raw.get("min_last_body_page_fill_ratio")
    if ratio is not None:
        _require(
            isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and 0 < ratio <= 1,
            name,
            "page_budget.min_last_body_page_fill_ratio must lie in (0, 1]",
        )
    exclude = raw.get("exclude_references", False)
    _require(isinstance(exclude, bool), name, "page_budget.exclude_references must be a bool")
    minimum = raw.get("min_body_pages")
    _require(
        minimum is None or minimum <= raw["max_body_pages"],
        name,
        "page_budget.min_body_pages must not exceed max_body_pages",
    )
    return PageBudget(
        max_body_pages=int(raw["max_body_pages"]),
        min_body_pages=minimum,
        min_last_body_page_fill_ratio=float(ratio) if ratio is not None else None,
        body_columns=int(raw.get("body_columns", 1)),
        exclude_references=exclude,
    )


def _parse_repo_relative(name: str, key: str, raw: Any, *, must_exist: bool) -> str:
    _require(isinstance(raw, str) and bool(raw.strip()), name, f"{key} must be a non-empty string")
    relative = Path(raw)
    _require(
        not relative.is_absolute() and ".." not in relative.parts,
        name,
        f"{key} must be a repo-relative path without '..'",
    )
    if must_exist:
        _require((REPO_ROOT / relative).is_file(), name, f"{key} {raw!r} is not a file")
    return relative.as_posix()


def parse_venue_spec(name: str, raw: Any) -> VenueSpec:
    """Validate a decoded venue mapping and return its :class:`VenueSpec`."""
    _require(isinstance(raw, dict), name, "venue file must contain a mapping")
    unknown = sorted(set(raw) - _SPEC_KEYS)
    _require(not unknown, name, f"unknown keys {unknown}")
    missing = sorted(_SPEC_KEYS - set(raw))
    _require(not missing, name, f"missing keys {missing}")
    _require(raw["name"] == name, name, f"name {raw['name']!r} must match the file stem")

    brief = _parse_repo_relative(name, "brief", raw["brief"], must_exist=True)
    author_block = (
        _parse_repo_relative(name, "author_block", raw["author_block"], must_exist=True)
        if raw["author_block"] is not None
        else None
    )
    template_dir = _parse_repo_relative(name, "template_dir", raw["template_dir"], must_exist=False)

    markers = raw["latex_markers"]
    _require(
        isinstance(markers, list)
        and bool(markers)
        and all(isinstance(m, str) and bool(m) for m in markers),
        name,
        "latex_markers must be a non-empty list of regex strings",
    )
    for marker in markers:
        try:
            re.compile(marker)
        except re.error as exc:
            raise VenueSpecError(f"venue {name!r}: latex_markers {marker!r}: {exc}") from exc

    checks = raw["checks"]
    _require(
        isinstance(checks, list) and all(isinstance(check, str) for check in checks),
        name,
        "checks must be a list of check names",
    )
    unknown_checks = sorted(set(checks) - KNOWN_CHECKS)
    _require(not unknown_checks, name, f"unknown checks {unknown_checks}")

    return VenueSpec(
        name=name,
        brief=brief,
        author_block=author_block,
        template_dir=template_dir,
        latex_markers=tuple(markers),
        page_budget=_parse_page_budget(name, raw["page_budget"]),
        checks=frozenset(checks),
    )


def load_venue(name: str) -> VenueSpec:
    """Load and validate venue *name* from :data:`VENUE_DIR`."""
    if not VENUE_NAME_RE.fullmatch(name):
        raise VenueSpecError(f"venue name {name!r} must match {VENUE_NAME_RE.pattern}")
    path = venue_spec_path(name)
    if not path.is_file():
        raise VenueSpecError(f"venue {name!r}: {path} does not exist")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise VenueSpecError(f"venue {name!r}: {path} is not valid YAML: {exc}") from exc
    return parse_venue_spec(name, raw)


@lru_cache(maxsize=1)
def venue_catalog() -> dict[str, VenueSpec]:
    """Return every declared venue keyed by name, in file-name order."""
    names = sorted(
        path.stem for path in VENUE_DIR.glob("*.yaml") if VENUE_NAME_RE.fullmatch(path.stem)
    )
    return {name: load_venue(name) for name in names}


def venue_names() -> tuple[str, ...]:
    """Return the declared venue names, which are the valid paper formats."""
    return tuple(venue_catalog())


def find_venue(name: str | None) -> VenueSpec | None:
    """Return the declared venue called *name*, or ``None``."""
    return venue_catalog().get(name) if name is not None else None


def venue_has_check(name: str | None, check: str) -> bool:
    """Report whether venue *name* opts into behavioural *check*."""
    if check not in KNOWN_CHECKS:
        raise ValueError(f"unknown venue check {check!r}")
    venue = find_venue(name)
    return venue is not None and check in venue.checks


def detect_venue(tex_text: str) -> VenueSpec | None:
    """Return the first declared venue whose LaTeX markers match *tex_text*."""
    for venue in venue_catalog().values():
        if venue.matches_tex(tex_text):
            return venue
    return None


def page_budget_sentence(name: str | None) -> str:
    """Render the brief's ``{{PAPER_PAGE_BUDGET}}`` text from the enforced budget."""
    venue = find_venue(name)
    budget = venue.page_budget if venue is not None else None
    if budget is None:
        return "no enforced page budget; follow the venue guidance below."
    if budget.min_body_pages is None:
        text = f"up to {budget.max_body_pages} pages"
    elif budget.min_body_pages == budget.max_body_pages:
        text = f"exactly {budget.max_body_pages} pages"
    else:
        text = f"{budget.min_body_pages} to {budget.max_body_pages} pages"
    if budget.exclude_references:
        text += ", excluding references"
    text += "."
    if budget.min_body_pages is not None and budget.min_last_body_page_fill_ratio is not None:
        text += (
            f" A body of {budget.min_body_pages} pages counts only when its final page is at"
            f" least {budget.min_last_body_page_fill_ratio:.0%} full; a shorter or sparser body"
            " fails validation."
        )
    return text


def page_limits() -> dict[str, dict[str, int | float | bool]]:
    """Return enforced page budgets keyed by venue name (advisory venues omitted)."""
    return {
        name: venue.page_budget.as_limits()
        for name, venue in venue_catalog().items()
        if venue.page_budget is not None
    }
