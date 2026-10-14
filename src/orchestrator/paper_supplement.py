"""Structural and rendered Supplementary Information validation."""

from __future__ import annotations

import re
from typing import Any


APPENDIX_RE = re.compile(r"\\appendix\b", re.IGNORECASE)
BIBLIOGRAPHY_RE = re.compile(
    r"\\bibliography\s*\{[^}]+\}|\\printbibliography\b(?:\[[^\]]*\])?"
    r"|\\begin\s*\{thebibliography\}",
    re.IGNORECASE,
)
CLEARPAGE_RE = re.compile(r"\\clearpage\b", re.IGNORECASE)
SUPPLEMENT_SECTION_RE = re.compile(
    r"\\section\s*\{\s*supplementary\s+information\s*\}",
    re.IGNORECASE,
)
NEXT_SECTION_RE = re.compile(r"\\section\*?\s*(?:\[[^\[\]]*\]\s*)?\{")
LET_SECTION_ALIAS_RE = re.compile(
    r"\\let\s*\\[a-zA-Z@]+\s*=?\s*\\section\b", re.IGNORECASE
)
NEWCOMMAND_DEFINITION_RE = re.compile(
    r"\\(?:newcommand|renewcommand)\*?\s*\{", re.IGNORECASE
)
DEF_DEFINITION_RE = re.compile(
    r"\\(?:def|gdef|edef|xdef)\s*\\[a-zA-Z@]+", re.IGNORECASE
)
TEX_IMPORT_RE = re.compile(r"\\(?:input|include)\b", re.IGNORECASE)
SUBSTANTIVE_ELEMENT_RE = re.compile(
    r"\\subsection\*?\s*\{|\\begin\s*\{(?:table|figure)\*?\}",
    re.IGNORECASE,
)


def _letter_spaced_word_pattern(word: str) -> str:
    return r"[ \t]*".join(re.escape(character) for character in word)


RENDERED_SUPPLEMENT_HEADING_RE = re.compile(
    rf"(?mi)^[ \t]*(?:(?:appendix[ \t]+)?[A-Z0-9]+(?:\.\d+)*[.:]?[ \t]+)?"
    rf"{_letter_spaced_word_pattern('supplementary')}[ \t]+"
    rf"{_letter_spaced_word_pattern('information')}[ \t]*$",
)
RENDERED_REFERENCE_HEADING_RE = re.compile(
    rf"(?mi)^[ \t]*(?:{_letter_spaced_word_pattern('references')}|"
    rf"{_letter_spaced_word_pattern('bibliography')})[ \t]*$"
)
CONDITIONAL_TOKEN_RE = re.compile(r"\\[a-zA-Z@]+")
NEWIF_RE = re.compile(r"\\newif\s*(?P<name>\\if[a-zA-Z@]+)")
CONDITIONAL_OPENERS = frozenset(
    {
        r"\if",
        r"\ifcase",
        r"\ifcat",
        r"\ifcondition",
        r"\ifcsname",
        r"\ifdefined",
        r"\ifdim",
        r"\ifeof",
        r"\iffalse",
        r"\iffontchar",
        r"\ifhbox",
        r"\ifhmode",
        r"\ifincsname",
        r"\ifinner",
        r"\ifluatex",
        r"\ifmmode",
        r"\ifnum",
        r"\ifodd",
        r"\ifpdf",
        r"\ifpdfabsdim",
        r"\ifpdfabsnum",
        r"\ifpdfprimitive",
        r"\iftrue",
        r"\ifvbox",
        r"\ifvmode",
        r"\ifvoid",
        r"\ifx",
        r"\ifxetex",
    }
)
NON_CONDITIONAL_IF_MACROS = frozenset(
    {
        r"\ifblank",
        r"\ifbool",
        r"\ifboolexpr",
        r"\ifcsdef",
        r"\ifcsempty",
        r"\ifcsequal",
        r"\ifcsmacro",
        r"\ifcsparam",
        r"\ifcsprotected",
        r"\ifcsstring",
        r"\ifcsundef",
        r"\ifcsvoid",
        r"\ifdimcomp",
        r"\ifdimequal",
        r"\ifdimgreater",
        r"\ifdimless",
        r"\ifdef",
        r"\ifdefempty",
        r"\ifdefequal",
        r"\ifdefltxprotect",
        r"\ifdefmacro",
        r"\ifdefparam",
        r"\ifdefprotected",
        r"\ifdefstring",
        r"\ifdefvoid",
        r"\iflanguage",
        r"\ifltxcount",
        r"\ifltxcounter",
        r"\ifnumcomp",
        r"\ifnumequal",
        r"\ifnumgreater",
        r"\ifnumless",
        r"\ifstrequal",
        r"\ifstrempty",
        r"\ifthenelse",
        r"\iftoggle",
        r"\ifundef",
    }
)
VERBATIM_ENV_START_RE = re.compile(
    r"\\begin\s*\{(?P<env>verbatim\*?|lstlisting|minted)\}"
)
VERBATIM_ENV_END_RES = {
    environment: re.compile(rf"\\end\s*\{{{re.escape(environment)}\}}")
    for environment in ("verbatim", "verbatim*", "lstlisting", "minted")
}
PLACEHOLDER_TEXTS = (
    "add supplementary material only when grounded artifacts support it",
    "replace with grounded extended methods, reproducibility details, additional analyses, or supporting tables and figures",
)


def strip_tex_comments(tex_text: str) -> str:
    """Remove LaTeX comments while preserving escaped percent signs."""
    cleaned_lines: list[str] = []
    for line in tex_text.splitlines():
        kept: list[str] = []
        index = 0
        while index < len(line):
            character = line[index]
            if line.startswith(r"\verb", index):
                delimiter_index = index + len(r"\verb")
                if delimiter_index < len(line) and line[delimiter_index] == "*":
                    delimiter_index += 1
                if (
                    delimiter_index < len(line)
                    and not line[delimiter_index].isspace()
                    and not line[delimiter_index].isalpha()
                ):
                    closing_index = line.find(line[delimiter_index], delimiter_index + 1)
                    if closing_index >= 0:
                        kept.append(line[index : closing_index + 1])
                        index = closing_index + 1
                        continue
            if character == "%" and not _is_escaped(line, index):
                break
            kept.append(line[index])
            index += 1
        cleaned_lines.append("".join(kept))
    return "\n".join(cleaned_lines)


def mask_tex_verbatim(tex_text: str) -> str:
    """Mask inline and environment verbatim text while preserving line structure."""
    output: list[str] = []
    for line in tex_text.splitlines(keepends=True):
        chars = list(line)
        index = 0
        while index < len(line):
            if not line.startswith(r"\verb", index):
                index += 1
                continue
            delimiter_index = index + len(r"\verb")
            if delimiter_index < len(line) and line[delimiter_index] == "*":
                delimiter_index += 1
            if (
                delimiter_index >= len(line)
                or line[delimiter_index].isspace()
                or line[delimiter_index].isalpha()
            ):
                index += 1
                continue
            closing_index = line.find(line[delimiter_index], delimiter_index + 1)
            if closing_index < 0:
                index += 1
                continue
            chars[index : closing_index + 1] = " " * (closing_index + 1 - index)
            index = closing_index + 1
        output.append("".join(chars))
    inline_masked = "".join(output)
    return _mask_verbatim_environments(inline_masked)


def _mask_verbatim_environments(tex_text: str) -> str:
    output: list[str] = []
    cursor = 0
    while opening := VERBATIM_ENV_START_RE.search(tex_text, cursor):
        closing = VERBATIM_ENV_END_RES[opening.group("env")].search(tex_text, opening.end())
        if closing is None:
            break
        output.append(tex_text[cursor : opening.start()])
        output.append(_mask_text(tex_text[opening.start() : closing.end()]))
        cursor = closing.end()
    output.append(tex_text[cursor:])
    return "".join(output)


def _mask_text(text: str) -> str:
    return "".join("\n" if character == "\n" else " " for character in text)


def strip_false_conditionals(
    tex_text: str, conditional_openers: set[str] | None = None
) -> str:
    r"""Remove literal ``\iffalse ... \fi`` regions before structural parsing."""
    if conditional_openers is None:
        conditional_openers = _conditional_openers(tex_text)
    output: list[str] = []
    cursor = 0
    while (false_match := re.search(r"\\iffalse\b", tex_text[cursor:])) is not None:
        false_start = cursor + false_match.start()
        output.append(tex_text[cursor:false_start])
        depth = 1
        false_end = len(tex_text)
        else_start: int | None = None
        closing_start = len(tex_text)
        for token in CONDITIONAL_TOKEN_RE.finditer(tex_text, false_start + len(r"\iffalse")):
            token_text = token.group(0).lower()
            if token_text == r"\fi":
                depth -= 1
                if depth == 0:
                    closing_start = token.start()
                    false_end = token.end()
                    break
            elif token_text == r"\else":
                if depth == 1 and else_start is None:
                    else_start = token.end()
            elif _is_conditional_opener(token, token_text, conditional_openers):
                depth += 1
        if else_start is not None and closing_start < len(tex_text):
            output.append(
                strip_false_conditionals(
                    tex_text[else_start:closing_start], conditional_openers
                )
            )
        cursor = false_end
    output.append(tex_text[cursor:])
    return "".join(output)


def mask_false_conditionals(tex_text: str) -> str:
    r"""Mask literal ``\iffalse`` branches while preserving source offsets."""
    conditional_openers = _conditional_openers(tex_text)
    chars = list(tex_text)
    cursor = 0
    while (false_match := re.search(r"\\iffalse\b", tex_text[cursor:])) is not None:
        false_start = cursor + false_match.start()
        depth = 1
        else_end: int | None = None
        closing_start = len(tex_text)
        false_end = len(tex_text)
        for token in CONDITIONAL_TOKEN_RE.finditer(tex_text, false_start + len(r"\iffalse")):
            token_text = token.group(0).lower()
            if token_text == r"\fi":
                depth -= 1
                if depth == 0:
                    closing_start = token.start()
                    false_end = token.end()
                    break
            elif token_text == r"\else":
                if depth == 1 and else_end is None:
                    else_end = token.end()
            elif _is_conditional_opener(token, token_text, conditional_openers):
                depth += 1
        mask_end = else_end if else_end is not None else false_end
        chars[false_start:mask_end] = _mask_text(tex_text[false_start:mask_end])
        if else_end is not None and closing_start < len(tex_text):
            chars[closing_start:false_end] = _mask_text(tex_text[closing_start:false_end])
            cursor = else_end
        else:
            cursor = false_end
    return "".join(chars)


def _conditional_openers(tex_text: str) -> set[str]:
    openers = set(CONDITIONAL_OPENERS)
    openers.update(match.group("name").lower() for match in NEWIF_RE.finditer(tex_text))
    return openers


def _is_conditional_opener(
    token: re.Match[str], normalized_token: str, conditional_openers: set[str]
) -> bool:
    if normalized_token in conditional_openers:
        return True
    return token.group(0).startswith(r"\if") and (
        normalized_token not in NON_CONDITIONAL_IF_MACROS
    )


def extract_supplementary_information_excerpt(tex_text: str) -> str | None:
    """Return the uncommented SI source, beginning at its appendix command."""
    _, supplement = split_manuscript_and_supplement(tex_text)
    return supplement


def split_manuscript_and_supplement(tex_text: str) -> tuple[str, str | None]:
    """Split manuscript source so a separately packed SI is not duplicated."""
    source = _supplement_source_parts(tex_text)
    if source is None:
        return tex_text, None
    uncommented_tex, appendix_match, heading_match, _ = source
    start = appendix_match.start() if appendix_match is not None else heading_match.start()
    document_end = uncommented_tex.find(r"\end{document}", heading_match.end())
    supplement_end = document_end if document_end >= 0 else len(uncommented_tex)
    return uncommented_tex[:start].rstrip(), uncommented_tex[start:supplement_end].strip()


def validate_supplementary_information(tex_text: str) -> dict[str, Any]:
    """Validate placement and substantive source content for embedded SI."""
    uncommented_tex = strip_tex_comments(tex_text)
    structural_tex = mask_false_conditionals(mask_tex_verbatim(uncommented_tex))
    heading_match = SUPPLEMENT_SECTION_RE.search(structural_tex)
    appendix_matches = list(APPENDIX_RE.finditer(structural_tex))
    appendix_before_heading = _last_match_before(appendix_matches, heading_match)
    appendix_match = appendix_before_heading or (appendix_matches[0] if appendix_matches else None)
    bibliography_matches = list(BIBLIOGRAPHY_RE.finditer(structural_tex))
    bibliography_match = bibliography_matches[-1] if bibliography_matches else None
    section_before_bibliography = _last_match_before(
        list(NEXT_SECTION_RE.finditer(structural_tex)), bibliography_match
    )
    pre_bibliography_clearpage_match = None
    if bibliography_match is not None and section_before_bibliography is not None:
        pre_bibliography_clearpage_match = CLEARPAGE_RE.search(
            structural_tex, section_before_bibliography.end(), bibliography_match.start()
        )
    clearpage_match = None
    if bibliography_match is not None and appendix_before_heading is not None:
        clearpage_match = CLEARPAGE_RE.search(
            structural_tex, bibliography_match.end(), appendix_before_heading.start()
        )
    appendix_present = bool(appendix_matches)
    heading_present = heading_match is not None
    pre_bibliography_clearpage_present = pre_bibliography_clearpage_match is not None
    clearpage_present = clearpage_match is not None
    placed_after_references = bool(
        heading_match is not None
        and bibliography_match is not None
        and appendix_before_heading is not None
        and bibliography_match.start() < appendix_before_heading.start() < heading_match.start()
    )

    supplement_source = ""
    structural_supplement_source = ""
    if heading_match is not None:
        supplement_end = _supplement_end(structural_tex, heading_match.end())
        supplement_source = uncommented_tex[heading_match.end():supplement_end]
        structural_supplement_source = structural_tex[heading_match.end():supplement_end]
    visible_content = _visible_tex_text(supplement_source)
    substantive_element_count = len(
        SUBSTANTIVE_ELEMENT_RE.findall(structural_supplement_source)
    )

    hard_errors: list[str] = []
    if not appendix_present:
        hard_errors.append(r"paper.tex missing \appendix for Supplementary Information")
    if not heading_present:
        hard_errors.append("paper.tex missing Supplementary Information heading")
    if appendix_match is not None and heading_match is not None and appendix_match.start() > heading_match.start():
        hard_errors.append(r"\appendix must appear before the Supplementary Information heading")
    if heading_present and not placed_after_references:
        hard_errors.append("Supplementary Information must appear after the bibliography")
    if heading_present and placed_after_references and not pre_bibliography_clearpage_present:
        hard_errors.append(r"paper.tex requires \clearpage before the bibliography")
    if heading_present and placed_after_references and not clearpage_present:
        hard_errors.append(
            r"Supplementary Information requires \clearpage between the bibliography and \appendix"
        )
    if heading_present and any(
        placeholder in visible_content.lower() for placeholder in PLACEHOLDER_TEXTS
    ):
        hard_errors.append("Supplementary Information still contains template placeholder text")
    if heading_present and len(visible_content) < 100:
        hard_errors.append("Supplementary Information must contain at least 100 visible characters")
    if heading_present and substantive_element_count == 0:
        hard_errors.append(
            "Supplementary Information must contain at least one substantive subsection, table, or figure"
        )
    if heading_match is not None:
        later_section_titles = _section_titles_after(structural_tex, heading_match.end())
        allowed_terminal_section = (
            len(later_section_titles) == 1
            and later_section_titles[0] in {"references", "bibliography"}
        )
        if later_section_titles and not allowed_terminal_section:
            hard_errors.append("Supplementary Information must be the final top-level section")
        if r"\csname" in structural_tex[heading_match.end() :] or _has_section_alias(
            structural_tex
        ):
            hard_errors.append(
                "Supplementary Information must not use dynamic top-level sections"
            )
        if TEX_IMPORT_RE.search(structural_tex, heading_match.end()):
            hard_errors.append(
                "Supplementary Information must not import unchecked TeX files"
            )

    return {
        "checked": True,
        "appendix_present": appendix_present,
        "heading_present": heading_present,
        "placed_after_references": placed_after_references,
        "pre_bibliography_clearpage_present": pre_bibliography_clearpage_present,
        "clearpage_present": clearpage_present,
        "visible_character_count": len(visible_content),
        "substantive_element_count": substantive_element_count,
        "rendered_checked": False,
        "rendered_heading_present": False,
        "rendered_visible_character_count": 0,
        "hard_errors": hard_errors,
    }


def validate_rendered_supplementary_information(page_texts: list[str]) -> dict[str, Any]:
    """Validate SI visibility in text extracted from a compiled PDF."""
    start_page = find_supplement_start_page(page_texts)
    if start_page is None:
        return {
            "rendered_checked": True,
            "rendered_heading_present": False,
            "rendered_visible_character_count": 0,
            "supplement_start_page": None,
            "supplement_pages": 0,
            "hard_errors": ["compiled paper.pdf is missing the Supplementary Information heading"],
        }

    reference_start_page = _find_rendered_reference_start_page(
        page_texts[: start_page - 1]
    )
    heading_page_text = page_texts[start_page - 1]
    heading_match = RENDERED_SUPPLEMENT_HEADING_RE.search(heading_page_text)
    assert heading_match is not None
    references_before_heading_on_same_page = RENDERED_REFERENCE_HEADING_RE.search(
        heading_page_text[: heading_match.start()]
    )
    supplement_chunks = [heading_page_text[heading_match.end() :], *page_texts[start_page:]]
    substantive_chunks: list[str] = []
    for chunk in supplement_chunks:
        later_references = RENDERED_REFERENCE_HEADING_RE.search(chunk)
        if later_references is not None:
            substantive_chunks.append(chunk[: later_references.start()])
            break
        substantive_chunks.append(chunk)
    supplement_text = "\n".join(substantive_chunks)
    visible_character_count = len(re.sub(r"\s+", " ", supplement_text).strip())
    hard_errors = []
    heading_page_prefix = heading_page_text[: heading_match.start()]
    prefix_lines = [line.strip() for line in heading_page_prefix.splitlines() if line.strip()]
    earlier_header_lines = {
        line.strip()
        for page_text in page_texts[: start_page - 1]
        for line in page_text.splitlines()[:2]
        if line.strip()
    }
    substantive_prefix_lines = [
        line
        for line in prefix_lines
        if re.fullmatch(r"(?:page\s+)?\d+", line, re.IGNORECASE) is None
        and line not in earlier_header_lines
    ]
    if substantive_prefix_lines:
        hard_errors.append(
            "compiled Supplementary Information must start on a separate page after References"
        )
    if references_before_heading_on_same_page is not None:
        hard_errors.append(
            "compiled Supplementary Information must start on a separate page after References"
        )
    elif reference_start_page is None:
        hard_errors.append("compiled paper.pdf is missing the References heading before SI")
    elif start_page <= reference_start_page:
        hard_errors.append(
            "compiled Supplementary Information must start on a separate page after References"
        )
    if visible_character_count < 100:
        hard_errors.append(
            "compiled Supplementary Information must contain at least 100 visible characters"
        )
    return {
        "rendered_checked": True,
        "rendered_heading_present": True,
        "rendered_visible_character_count": visible_character_count,
        "supplement_start_page": start_page,
        "supplement_pages": len(page_texts) - start_page + 1,
        "hard_errors": hard_errors,
    }


def find_supplement_start_page(page_texts: list[str]) -> int | None:
    """Return the first 1-indexed PDF page containing the rendered SI heading."""
    for page_number, page_text in enumerate(page_texts, start=1):
        if RENDERED_SUPPLEMENT_HEADING_RE.search(page_text):
            return page_number
    return None


def _find_rendered_reference_start_page(page_texts: list[str]) -> int | None:
    last_reference_page = None
    for page_number, page_text in enumerate(page_texts, start=1):
        if RENDERED_REFERENCE_HEADING_RE.search(page_text):
            last_reference_page = page_number
    return last_reference_page


def _supplement_source_parts(
    tex_text: str,
) -> tuple[str, re.Match[str] | None, re.Match[str], int] | None:
    uncommented_tex = strip_tex_comments(tex_text)
    structural_tex = mask_false_conditionals(mask_tex_verbatim(uncommented_tex))
    heading_match = SUPPLEMENT_SECTION_RE.search(structural_tex)
    if heading_match is None:
        return None
    appendix_match = _last_match_before(list(APPENDIX_RE.finditer(structural_tex)), heading_match)
    return uncommented_tex, appendix_match, heading_match, _supplement_end(
        structural_tex, heading_match.end()
    )


def _last_match_before(
    matches: list[re.Match[str]], boundary: re.Match[str] | None
) -> re.Match[str] | None:
    if boundary is None:
        return None
    preceding = [match for match in matches if match.start() < boundary.start()]
    return preceding[-1] if preceding else None


def _section_titles_after(tex_text: str, start: int) -> list[str | None]:
    titles: list[str | None] = []
    for match in NEXT_SECTION_RE.finditer(tex_text, start):
        title = _balanced_braced_content(tex_text, match.end() - 1)
        if title is None:
            titles.append(None)
            continue
        normalized_title = _visible_tex_text(title).strip().lower()
        titles.append(normalized_title or None)
    return titles


def _balanced_braced_content(tex_text: str, open_brace_index: int) -> str | None:
    span = _balanced_braced_span(tex_text, open_brace_index)
    return span[0] if span is not None else None


def _balanced_braced_span(tex_text: str, open_brace_index: int) -> tuple[str, int] | None:
    depth = 0
    for index in range(open_brace_index, len(tex_text)):
        character = tex_text[index]
        if character == "{" and not _is_escaped(tex_text, index):
            depth += 1
        elif character == "}" and not _is_escaped(tex_text, index):
            depth -= 1
            if depth == 0:
                return tex_text[open_brace_index + 1 : index], index + 1
    return None


def _has_section_alias(tex_text: str) -> bool:
    if LET_SECTION_ALIAS_RE.search(tex_text):
        return True
    for match in NEWCOMMAND_DEFINITION_RE.finditer(tex_text):
        name_span = _balanced_braced_span(tex_text, match.end() - 1)
        if name_span is None:
            continue
        cursor = name_span[1]
        while True:
            optional_argument = re.match(r"\s*\[[^\[\]]*\]", tex_text[cursor:])
            if optional_argument is None:
                break
            cursor += optional_argument.end()
        cursor += len(tex_text[cursor:]) - len(tex_text[cursor:].lstrip())
        if cursor >= len(tex_text) or tex_text[cursor] != "{":
            continue
        body_span = _balanced_braced_span(tex_text, cursor)
        if body_span is not None and re.search(r"\\section\b", body_span[0]):
            return True
    for match in DEF_DEFINITION_RE.finditer(tex_text):
        body_start = tex_text.find("{", match.end(), min(len(tex_text), match.end() + 500))
        if body_start < 0:
            continue
        body_span = _balanced_braced_span(tex_text, body_start)
        if body_span is not None and re.search(r"\\section\b", body_span[0]):
            return True
    return False


def _supplement_end(uncommented_tex: str, content_start: int) -> int:
    next_section = NEXT_SECTION_RE.search(uncommented_tex, content_start)
    document_end = uncommented_tex.find(r"\end{document}", content_start)
    candidates = [position for position in (document_end, next_section.start() if next_section else -1) if position >= 0]
    return min(candidates) if candidates else len(uncommented_tex)


def _visible_tex_text(tex_text: str) -> str:
    visible = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]+\}", " ", tex_text)
    visible = re.sub(
        r"\\(?:bibliography|bibliographystyle|label|ref|eqref|pageref|cite[a-zA-Z*]*)"
        r"\*?(?:\[[^\]]*\])?\{[^}]*\}",
        " ",
        visible,
    )
    visible = re.sub(r"\\(?:input|include)\{[^}]*\}", " ", visible)
    visible = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?", " ", visible)
    visible = re.sub(r"[{}]", " ", visible)
    return re.sub(r"\s+", " ", visible).strip()


def _is_escaped(text: str, index: int) -> bool:
    backslash_count = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslash_count += 1
        cursor -= 1
    return backslash_count % 2 == 1
