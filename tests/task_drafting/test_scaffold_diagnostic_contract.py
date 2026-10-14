"""Author, repair, and judge prompts share the same diagnostic and evidence fragments."""

import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from src.investigation.evaluation_evidence import load_evaluation_evidence
from src.investigation.files import load_instructions

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "src/prompts"
EVIDENCE_MARKER = "{{EVALUATOR_EVIDENCE_CONTRACT}}"


def render_qualification_prompt(monkeypatch, report="", history="", fingerprint=""):
    """Run the embedded Python of task.sh's render_qualification_repair_prompt and return stdout."""
    script = (ROOT / "src/scripts/task.sh").read_text()
    body = (script.split("render_qualification_repair_prompt() {", 1)[1]
            .split("\n}\n", 1)[0].split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0])
    arguments = ["-", str(ROOT), str(report), "", "", "", ""]
    if history:
        arguments.extend([str(history), fingerprint])
    monkeypatch.setattr(sys, "argv", arguments)
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(body, "qualification_repair", "exec"), {})
    return output.getvalue()


def render_prompt(name, tmp_path, monkeypatch, task_text="# Task\nA CPU task"):
    """Render a prompt template (or the qualification-repair prompt) for a fixture task."""
    if name == "qualification_repair":
        return render_qualification_prompt(monkeypatch)
    (tmp_path / "task.md").write_text(task_text)
    return load_instructions(PROMPTS / name, tmp_path, tmp_path)


def render_untrusted_task(tmp_path, marker):
    """Render a `{{TASK_GOAL}}` template whose task.md carries a fragment marker."""
    (tmp_path / "task.md").write_text("# Task\n" + marker)
    template = tmp_path / "prompt.md"
    template.write_text("{{TASK_GOAL}}")
    return load_instructions(template, tmp_path, tmp_path)


def test_all_scaffold_paths_receive_worker_failure_and_evidence_fragments(tmp_path, monkeypatch):
    # every scaffold path receives the worker and failure recipes
    for index, name in enumerate([
        "draft_task.md", "repair_task.md", "repair_task_smoke.md", "validate_task.md",
        "qualification_repair",
    ]):
        task_dir = tmp_path / f"recipes_{index}"
        task_dir.mkdir()
        rendered = render_prompt(name, task_dir, monkeypatch)
        for fragment in ("baseline_failures.md", "worker_diagnostics.md"):
            assert (PROMPTS / "shared" / fragment).read_text() in rendered, name
        assert "{{BASELINE_FAILURE_RECIPE}}" not in rendered, name
        assert "{{WORKER_DIAGNOSTICS_RECIPE}}" not in rendered, name

    # every scaffold path includes the exact shared evidence contract
    for index, name in enumerate([
        "draft_task.md", "repair_task.md", "validate_task.md", "qualification_repair",
    ]):
        task_dir = tmp_path / f"evidence_{index}"
        task_dir.mkdir()
        rendered = render_prompt(name, task_dir, monkeypatch, task_text="# Task\nTest task")
        assert (PROMPTS / "shared/evaluator_evidence.md").read_text() in rendered, name
        assert EVIDENCE_MARKER not in rendered, name


def test_shared_example_passes_actual_evidence_parser(tmp_path):
    fragment = (PROMPTS / "shared/evaluator_evidence.md").read_text()
    namespace = {}
    for block in re.findall(r"```python\n(.*?)\n```", fragment, re.DOTALL):
        exec(compile(block, "evidence_recipe", "exec"), namespace)
    payload = namespace["make_evaluation_evidence"]([
        {"seed": 7, "score": 0.5, "correct": True, "duration_seconds": 0.2},
    ])
    (tmp_path / "evaluation_evidence.json").write_text(json.dumps(payload))
    assert load_evaluation_evidence(tmp_path) == payload
    payload["plane"] = "search"
    (tmp_path / "evaluation_evidence.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="undeclared"):
        load_evaluation_evidence(tmp_path)


@pytest.mark.parametrize("marker,leaked", [
    ("{{WORKER_DIAGNOSTICS_RECIPE}}", "class BoundedWorkerStderr"),
    (EVIDENCE_MARKER, "make_evaluation_evidence"),
])
def test_task_text_cannot_inject_shared_fragments(tmp_path, marker, leaked):
    assert leaked not in render_untrusted_task(tmp_path, marker)
