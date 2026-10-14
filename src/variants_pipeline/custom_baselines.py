"""Parse custom baseline markdown files and folders into VariantCandidate objects.

Custom baselines can be specified as either:

1. **Markdown files** with YAML frontmatter for structured metadata and a
   markdown body for the detailed description::

       ---
       variant_id: baseline_dfa
       title: Direct Feedback Alignment
       hypothesis: DFA uses random fixed feedback weights...
       ---

       ## Description
       Direct Feedback Alignment is a biologically plausible learning rule...

2. **Folders** containing a working reference implementation.  Metadata is
   inferred from the folder name; an optional ``baseline.md`` inside the
   folder can supply overrides.  The folder contents are copied as
   ``program_0`` and the agent adapts them to integrate with the evaluation
   harness.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

from .models import VariantCandidate

logger = logging.getLogger(__name__)

_VARIANT_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def load_custom_baselines(paths: list[Path]) -> list[VariantCandidate]:
    """Load and validate custom baseline paths (markdown files or folders).

    Parameters
    ----------
    paths:
        Paths to ``.md`` files with YAML frontmatter, or directories
        containing reference implementations.

    Returns
    -------
    list[VariantCandidate]
        One candidate per path, with ``description_body`` populated and
        ``reference_dir`` set for folder-based baselines.

    Raises
    ------
    FileNotFoundError
        If a path does not exist.
    ValueError
        On parse errors, missing ``variant_id``, or duplicate IDs across paths.
    """
    candidates: list[VariantCandidate] = []
    seen_ids: dict[str, Path] = {}

    for path in paths:
        path = path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Custom baseline path not found: {path}")

        if path.is_dir():
            candidate = _load_folder_baseline(path)
        elif path.is_file():
            candidate = _parse_baseline_markdown(path)
        else:
            raise ValueError(
                f"Custom baseline path must be a file or directory: {path}"
            )

        if candidate.variant_id in seen_ids:
            raise ValueError(
                f"Duplicate variant_id '{candidate.variant_id}' in "
                f"{path} and {seen_ids[candidate.variant_id]}"
            )
        seen_ids[candidate.variant_id] = path
        candidates.append(candidate)

    folder_count = sum(1 for c in candidates if c.reference_dir is not None)
    logger.info(
        "Loaded %d custom baseline(s) (%d folder, %d markdown): %s",
        len(candidates),
        folder_count,
        len(candidates) - folder_count,
        [c.variant_id for c in candidates],
    )
    return candidates


def _parse_baseline_markdown(path: Path) -> VariantCandidate:
    """Parse a single markdown file into a ``VariantCandidate``."""
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text, path)

    meta = yaml.safe_load(frontmatter)
    if not isinstance(meta, dict):
        raise ValueError(f"YAML frontmatter in {path} is not a mapping")

    # --- variant_id (required) ---
    variant_id = meta.get("variant_id")
    if not variant_id:
        raise ValueError(f"Missing required 'variant_id' in frontmatter of {path}")
    variant_id = str(variant_id)

    if not _VARIANT_ID_RE.match(variant_id):
        raise ValueError(
            f"Invalid variant_id '{variant_id}' in {path}: "
            f"must match {_VARIANT_ID_RE.pattern}"
        )

    if not variant_id.startswith("baseline_"):
        old_id = variant_id
        variant_id = f"baseline_{variant_id}"
        logger.warning(
            "variant_id '%s' in %s does not start with 'baseline_'; "
            "auto-renamed to '%s'",
            old_id,
            path,
            variant_id,
        )

    # --- optional fields with defaults ---
    title = str(meta.get("title", variant_id))
    kind = str(meta.get("kind", "baseline"))
    base_program = str(meta.get("base_program", "best"))
    hypothesis = str(meta.get("hypothesis", ""))
    expected_effect = str(meta.get("expected_effect", ""))
    relevance_reason = str(meta.get("relevance_reason", ""))
    feasibility_reason = str(meta.get("feasibility_reason", ""))

    target_components = [str(c) for c in meta.get("target_components", [])]
    implementation_plan = [str(s) for s in meta.get("implementation_plan", [])]

    return VariantCandidate(
        variant_id=variant_id,
        title=title,
        kind=kind,
        base_program=base_program,
        hypothesis=hypothesis,
        relevance_reason=relevance_reason,
        feasibility_reason=feasibility_reason,
        target_components=target_components,
        implementation_plan=implementation_plan,
        expected_effect=expected_effect,
        description_body=body.strip(),
    )


def _load_folder_baseline(folder: Path) -> VariantCandidate:
    """Create a :class:`VariantCandidate` from a folder with reference code.

    Metadata is inferred from the folder name.  If the folder contains a
    ``baseline.md`` with YAML frontmatter, those values override the
    inferred defaults.
    """
    # --- infer defaults from folder name ---
    raw_id = _sanitize_folder_name(folder.name)
    if not raw_id:
        raise ValueError(
            f"Cannot derive variant_id from folder name '{folder.name}'"
        )

    if not raw_id.startswith("baseline_"):
        raw_id = f"baseline_{raw_id}"

    variant_id = raw_id
    title = _prettify_name(folder.name)
    kind = "baseline"
    base_program = "reference"
    hypothesis = ""
    expected_effect = ""
    relevance_reason = ""
    feasibility_reason = ""
    target_components: list[str] = []
    implementation_plan: list[str] = []
    description_body = ""

    # --- read README.md as description_body if present ---
    readme_path = folder / "README.md"
    if readme_path.is_file():
        description_body = readme_path.read_text(encoding="utf-8").strip()

    # --- optional baseline.md overrides ---
    baseline_md = folder / "baseline.md"
    if baseline_md.is_file():
        try:
            override = _parse_baseline_markdown(baseline_md)
            variant_id = override.variant_id
            title = override.title
            kind = override.kind
            base_program = override.base_program if override.base_program != "best" else "reference"
            if override.hypothesis:
                hypothesis = override.hypothesis
            if override.expected_effect:
                expected_effect = override.expected_effect
            if override.relevance_reason:
                relevance_reason = override.relevance_reason
            if override.feasibility_reason:
                feasibility_reason = override.feasibility_reason
            if override.target_components:
                target_components = override.target_components
            if override.implementation_plan:
                implementation_plan = override.implementation_plan
            if override.description_body:
                description_body = override.description_body
        except ValueError:
            logger.warning(
                "baseline.md in %s could not be parsed; using inferred defaults",
                folder,
            )

    return VariantCandidate(
        variant_id=variant_id,
        title=title,
        kind=kind,
        base_program=base_program,
        hypothesis=hypothesis,
        relevance_reason=relevance_reason,
        feasibility_reason=feasibility_reason,
        target_components=target_components,
        implementation_plan=implementation_plan,
        expected_effect=expected_effect,
        description_body=description_body,
        reference_dir=folder,
    )


def _sanitize_folder_name(name: str) -> str:
    """Convert a folder name to a valid variant_id component.

    Lowercases, replaces hyphens and spaces with underscores, strips other
    invalid characters.
    """
    result = name.lower().replace("-", "_").replace(" ", "_")
    # Keep only [a-z0-9_]
    result = re.sub(r"[^a-z0-9_]", "", result)
    # Collapse multiple underscores
    result = re.sub(r"_+", "_", result).strip("_")
    return result


def _prettify_name(name: str) -> str:
    """Convert a folder name to a human-readable title."""
    return name.replace("_", " ").replace("-", " ").title()


def _split_frontmatter(text: str, path: Path) -> tuple[str, str]:
    """Split ``---``-delimited YAML frontmatter from the markdown body.

    Returns ``(frontmatter_yaml, body_markdown)``.
    """
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        raise ValueError(
            f"File {path} does not contain YAML frontmatter "
            f"(expected '---' delimiter at the start)"
        )

    # Find the closing ---
    first_newline = stripped.index("\n")
    rest = stripped[first_newline + 1 :]
    closing = rest.find("\n---")
    if closing == -1:
        raise ValueError(
            f"File {path} has an opening '---' but no closing '---' delimiter"
        )

    frontmatter = rest[:closing]
    body = rest[closing + 4 :]  # skip \n---
    # Strip the rest of the closing --- line (e.g. trailing whitespace/newline)
    if body and body[0] == "\n":
        body = body[1:]

    return frontmatter, body
