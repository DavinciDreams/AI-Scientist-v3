"""Pure checks shared by harness rubric generation and evaluation."""

from __future__ import annotations

import ast
import io
import math
import operator
import re
import token
import tokenize
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeAlias

COMPARATORS: dict[str, Callable[[float, float], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}

COMPARATOR_ALIASES = {
    "exceeds": ">",
    "exceed": ">",
    "above": ">",
    "greater than": ">",
    "at least": ">=",
    "minimum": ">=",
    "min": ">=",
    "below": "<",
    "less than": "<",
    "at most": "<=",
    "maximum": "<=",
    "max": "<=",
}

MAX_HARNESS_RUBRICS = 32
NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
NUMBER_BOUNDARY = r"(?!\w|\.(?=[\w.]))"
METRIC_COMPARISON_RE = re.compile(
    rf"\b(?P<metric>[A-Za-z_][A-Za-z0-9_]*)\b\s*"
    rf"(?P<operator>>=|<=|>|<)\s*"
    rf"(?P<threshold>{NUMBER_PATTERN}){NUMBER_BOUNDARY}"
)

TokenValue: TypeAlias = str | int | float | complex | bytes | tuple[object, ...]
SourceToken: TypeAlias = tuple[int, TokenValue]
SourceTokens: TypeAlias = tuple[SourceToken, ...]
CandidateTokens: TypeAlias = tuple[SourceTokens, ...]
SourceSpan: TypeAlias = tuple[tuple[int, int], tuple[int, int]]

FSTRING_START = getattr(token, "FSTRING_START", -1)
FSTRING_MIDDLE = getattr(token, "FSTRING_MIDDLE", -1)
FSTRING_END = getattr(token, "FSTRING_END", -1)


def rubric_metric_check(rubric: dict[str, Any]) -> tuple[str, str, float] | None:
    """Return a numeric metric comparison declared by a rubric."""
    structured = _structured_metric_check(rubric)
    if structured is not None:
        return structured

    description = rubric.get("description")
    if isinstance(description, str):
        return _description_metric_check(description)
    return None


def candidate_source_tokens(candidate_sources: Sequence[str]) -> CandidateTokens:
    """Tokenize candidate files once, skipping invalid auxiliary Python files."""
    tokenized: list[SourceTokens] = []
    for source in candidate_sources:
        try:
            tokenized.append(_source_tokens(source))
        except (SyntaxError, ValueError, RecursionError, tokenize.TokenError):
            tokenized.append(())
    return tuple(tokenized)


def missing_source_patterns(candidate_tokens: CandidateTokens, patterns: Iterable[str]) -> list[str]:
    """Return patterns absent from executable Python tokens."""
    pattern_list = list(patterns)
    found: set[int] = set()
    bare_names: dict[TokenValue, set[int]] = {}
    transitions: list[dict[SourceToken, int]] = [{}]
    failures = [0]
    outputs: list[set[int]] = [set()]

    for index, pattern in enumerate(pattern_list):
        pattern_tokens = _pattern_tokens(pattern)
        if not pattern_tokens:
            continue
        if len(pattern_tokens) == 1 and pattern_tokens[0][0] == token.NAME:
            bare_names.setdefault(pattern_tokens[0][1], set()).add(index)
            continue
        state = 0
        for pattern_token in pattern_tokens:
            next_state = transitions[state].get(pattern_token)
            if next_state is None:
                next_state = len(transitions)
                transitions[state][pattern_token] = next_state
                transitions.append({})
                failures.append(0)
                outputs.append(set())
            state = next_state
        outputs[state].add(index)

    _populate_failure_transitions(transitions, failures, outputs)
    for source_tokens in candidate_tokens:
        state = 0
        for source_token in source_tokens:
            bare_name = _bare_name_value(source_token)
            if bare_name is not None:
                found.update(bare_names.get(bare_name, ()))
            while state and source_token not in transitions[state]:
                state = failures[state]
            state = transitions[state].get(source_token, 0)
            found.update(outputs[state])
    return [pattern for index, pattern in enumerate(pattern_list) if index not in found]


def _populate_failure_transitions(
    transitions: Sequence[dict[SourceToken, int]],
    failures: list[int],
    outputs: list[set[int]],
) -> None:
    pending = deque(transitions[0].values())
    while pending:
        state = pending.popleft()
        for source_token, next_state in transitions[state].items():
            pending.append(next_state)
            fallback = failures[state]
            while fallback and source_token not in transitions[fallback]:
                fallback = failures[fallback]
            failures[next_state] = transitions[fallback].get(source_token, 0)
            outputs[next_state].update(outputs[failures[next_state]])


def _numeric_value(raw: Any) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    try:
        value = float(raw)
    except (OverflowError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _structured_metric_check(rubric: dict[str, Any]) -> tuple[str, str, float] | None:
    metric_raw = rubric.get("metric") or rubric.get("metric_key") or rubric.get("metric_name")
    operator_raw = rubric.get("operator") or rubric.get("comparison")
    threshold_raw = rubric.get("threshold") if "threshold" in rubric else rubric.get("target")
    if metric_raw is None or operator_raw is None or threshold_raw is None:
        return None

    metric = str(metric_raw).strip()
    comparator = str(operator_raw).strip()
    threshold = _numeric_value(threshold_raw)
    if not metric or comparator not in COMPARATORS or threshold is None:
        return None
    return metric, comparator, threshold


def _description_metric_check(description: str) -> tuple[str, str, float] | None:
    match = METRIC_COMPARISON_RE.search(description)
    if match:
        threshold = _finite_float(match.group("threshold"))
        if threshold is not None:
            return match.group("metric"), match.group("operator"), threshold

    lowered = description.lower()
    for phrase, comparator in COMPARATOR_ALIASES.items():
        phrase_match = re.search(
            rf"\b{re.escape(phrase)}\s+(?P<threshold>{NUMBER_PATTERN}){NUMBER_BOUNDARY}",
            lowered,
        )
        if phrase_match:
            threshold = _finite_float(phrase_match.group("threshold"))
            if threshold is not None:
                return "combined_score", comparator, threshold
    return None


def _finite_float(raw: str) -> float | None:
    try:
        value = float(raw)
    except (OverflowError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _source_tokens(source: str) -> SourceTokens:
    tree = ast.parse(source)
    docstring_spans = _docstring_spans(tree, source)
    return _token_values(source, docstring_spans=docstring_spans, allow_incomplete=False)


def _pattern_tokens(pattern: str) -> SourceTokens:
    return _token_values(pattern, docstring_spans=(), allow_incomplete=True)


def _token_values(
    source: str,
    *,
    docstring_spans: Sequence[SourceSpan],
    allow_incomplete: bool,
) -> SourceTokens:
    values: list[SourceToken] = []
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    try:
        for source_token in tokens:
            value = _token_value(source_token, docstring_spans)
            if value is not None:
                values.append(value)
    except (IndentationError, tokenize.TokenError):
        if not allow_incomplete:
            raise
    return tuple(values)


def _token_value(
    source_token: tokenize.TokenInfo,
    docstring_spans: Sequence[SourceSpan],
) -> SourceToken | None:
    if source_token.type in {
        token.ENCODING,
        token.ENDMARKER,
        token.INDENT,
        token.DEDENT,
        token.NEWLINE,
        token.NL,
        token.COMMENT,
    }:
        return None
    if source_token.type == token.ERRORTOKEN:
        return None if source_token.string.isspace() else (token.ERRORTOKEN, source_token.string)
    if source_token.type == token.STRING:
        if any(_position_in_span(source_token.start, span) for span in docstring_spans):
            return None
        return token.STRING, _normalized_string(source_token.string)
    if source_token.type == token.NUMBER:
        try:
            value = ast.literal_eval(source_token.string)
        except (SyntaxError, ValueError):
            return token.NUMBER, ("invalid", source_token.string)
        if isinstance(value, (int, float, complex)):
            return token.NUMBER, (type(value).__name__, value)
        return token.NUMBER, ("invalid", source_token.string)
    if source_token.type == FSTRING_START:
        return FSTRING_START, _fstring_prefix(source_token.string)
    if source_token.type == FSTRING_END:
        return FSTRING_END, ""
    return source_token.type, source_token.string


def _normalized_string(raw: str) -> TokenValue:
    try:
        expression = ast.parse(raw, mode="eval").body
    except (SyntaxError, ValueError):
        return "invalid", raw
    if isinstance(expression, ast.Constant) and isinstance(expression.value, (str, bytes)):
        return "literal", type(expression.value).__name__, expression.value
    if isinstance(expression, ast.JoinedStr):
        return "fstring", ast.dump(expression, include_attributes=False)
    return "invalid", raw


def _fstring_prefix(raw: str) -> str:
    prefix = raw[: min(index for index, character in enumerate(raw) if character in "'\"")]
    return "".join(sorted(prefix.lower()))


def _docstring_spans(tree: ast.AST, source: str) -> tuple[SourceSpan, ...]:
    lines = source.splitlines(keepends=True)
    spans: list[SourceSpan] = []
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        expression = body[0]
        if not (
            isinstance(expression, ast.Expr)
            and isinstance(expression.value, ast.Constant)
            and isinstance(expression.value.value, str)
        ):
            continue
        start = (expression.lineno, _character_column(lines, expression.lineno, expression.col_offset))
        end = (
            expression.end_lineno or expression.lineno,
            _character_column(
                lines,
                expression.end_lineno or expression.lineno,
                expression.end_col_offset or expression.col_offset,
            ),
        )
        spans.append((start, end))
    return tuple(spans)


def _character_column(lines: Sequence[str], line_number: int, byte_column: int) -> int:
    line = lines[line_number - 1]
    return len(line.encode("utf-8")[:byte_column].decode("utf-8"))


def _position_in_span(position: tuple[int, int], span: SourceSpan) -> bool:
    start, end = span
    return start <= position < end


def _bare_name_value(source_token: SourceToken) -> TokenValue | None:
    token_type, value = source_token
    if token_type == token.NAME:
        return value
    if token_type == token.STRING and isinstance(value, tuple):
        if value[:2] == ("literal", "str") and isinstance(value[2], str):
            return value[2]
        return None
    return value if token_type == FSTRING_MIDDLE else None
