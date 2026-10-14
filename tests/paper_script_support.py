"""Parse ``src/scripts/paper.sh``'s flag grammar for tests.

Two consumers share this: the golden effective-config test, which needs the
variable state paper.sh would end up with after a given flag list, and the
paper config mirror test, which needs the flag list and script defaults.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_SCRIPT = REPO_ROOT / "src" / "scripts" / "paper.sh"

_ASSIGNMENT_RE = re.compile(r'^([A-Z_][A-Z0-9_]*)=(.*)$')
_ENV_DEFAULT_RE = re.compile(r'^"?\$\{[A-Z_][A-Z0-9_]*:-([^}]*)\}"?$')
_ENV_IS_SET_RE = re.compile(r'^"?\$\{[A-Z_][A-Z0-9_]*\+x\}"?$')
_ARM_RE = re.compile(r'^\s+((?:-{1,2}[a-z][a-z0-9-]*)(?:\|-{1,2}[a-z][a-z0-9-]*)*)\)\s*$')
_VALUE_ASSIGNMENT_RE = re.compile(r'\b([A-Z_][A-Z0-9_]*)="\$2"')
_LITERAL_ASSIGNMENT_RE = re.compile(r'(?<![\w$])([A-Z_][A-Z0-9_]*)=("([^"]*)"|[A-Za-z0-9_.-]+)')


@dataclass(frozen=True)
class FlagArm:
    """One ``case`` arm: the flags it matches and the variables it sets."""

    flags: tuple[str, ...]
    value_vars: tuple[str, ...]
    literal_assignments: tuple[tuple[str, str], ...]

    @property
    def takes_value(self) -> bool:
        return bool(self.value_vars)

    @property
    def long_flag(self) -> str:
        return next(flag for flag in self.flags if flag.startswith("--"))


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _split_script(text: str) -> tuple[list[str], list[str]]:
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("while [[ $# -gt 0 ]]"))
    end = next(index for index in range(start, len(lines)) if lines[index] == "done")
    return lines[:start], lines[start:end]


def script_defaults(text: str | None = None) -> dict[str, str]:
    """Variable defaults declared above the argument loop.

    ``${VAR:-x}`` is read as ``x`` and ``${VAR+x}`` as the empty string, so the
    result is the script's behaviour in a clean environment.
    """
    if text is None:
        text = PAPER_SCRIPT.read_text(encoding="utf-8")
    prelude, _ = _split_script(text)
    defaults: dict[str, str] = {}
    for line in prelude:
        match = _ASSIGNMENT_RE.match(line)
        if match is None:
            continue
        name, raw_value = match.groups()
        env_default = _ENV_DEFAULT_RE.match(raw_value)
        if env_default is not None:
            defaults[name] = env_default.group(1)
        elif _ENV_IS_SET_RE.match(raw_value):
            defaults[name] = ""
        else:
            defaults[name] = _strip_quotes(raw_value)
    return defaults


def environment_backed_vars(text: str | None = None) -> set[str]:
    """Prelude variables paper.sh initialises from the environment."""
    if text is None:
        text = PAPER_SCRIPT.read_text(encoding="utf-8")
    prelude, _ = _split_script(text)
    names: set[str] = set()
    for line in prelude:
        match = _ASSIGNMENT_RE.match(line)
        if match is None:
            continue
        name, raw_value = match.groups()
        if _ENV_DEFAULT_RE.match(raw_value) or _ENV_IS_SET_RE.match(raw_value):
            names.add(name)
    return names


def flag_arms(text: str | None = None) -> list[FlagArm]:
    """Every dashed ``case`` arm of the argument loop, in script order."""
    if text is None:
        text = PAPER_SCRIPT.read_text(encoding="utf-8")
    _, loop = _split_script(text)
    arms: list[FlagArm] = []
    index = 0
    while index < len(loop):
        match = _ARM_RE.match(loop[index])
        if match is None:
            index += 1
            continue
        flags = tuple(match.group(1).split("|"))
        body: list[str] = []
        index += 1
        while index < len(loop) and loop[index].strip() != ";;":
            body.append(loop[index])
            index += 1
        value_vars: list[str] = []
        literals: list[tuple[str, str]] = []
        for line in body:
            stripped = line.strip()
            if stripped.startswith(("echo ", "exit ", "shift", "if ", "fi", "case ", "esac", "*)")):
                continue
            for name in _VALUE_ASSIGNMENT_RE.findall(line):
                value_vars.append(name)
            for name, raw_value, quoted in _LITERAL_ASSIGNMENT_RE.findall(line):
                if raw_value == '"$2"':
                    continue
                literals.append((name, quoted if raw_value.startswith('"') else raw_value))
        arms.append(
            FlagArm(
                flags=flags,
                value_vars=tuple(dict.fromkeys(value_vars)),
                literal_assignments=tuple(literals),
            )
        )
    return arms


def apply_flags(argv: list[str], text: str | None = None) -> dict[str, str]:
    """Return the variable state paper.sh reaches after parsing *argv*."""
    if text is None:
        text = PAPER_SCRIPT.read_text(encoding="utf-8")
    state = script_defaults(text)
    arms_by_flag = {flag: arm for arm in flag_arms(text) for flag in arm.flags}
    index = 0
    while index < len(argv):
        token = argv[index]
        arm = arms_by_flag.get(token)
        if arm is None:
            raise ValueError(f"paper.sh does not accept {token!r}")
        if arm.takes_value:
            if index + 1 >= len(argv):
                raise ValueError(f"{token} requires a value")
            for name in arm.value_vars:
                state[name] = argv[index + 1]
            index += 2
        else:
            index += 1
        for name, literal in arm.literal_assignments:
            state[name] = literal
    return state
