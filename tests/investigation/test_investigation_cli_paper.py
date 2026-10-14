#!/usr/bin/env python3
"""Investigation CLI paper command, paper shell scripts, and the concept-figure helper."""
from __future__ import annotations

import json
import os
import re
import shlex
import stat
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any
from unittest.mock import ANY, Mock, patch

import pytest

from src.investigation import cli as investigation_cli

CLI = "src.investigation.cli"
POPEN = f"{CLI}.subprocess.Popen"
PAPER_SH = "src/scripts/paper.sh"
PAPER_EDIT_SH = "src/scripts/paper_edit.sh"
CONCEPT_SH = "src/scripts/lib/concept_figure.sh"
CONTAINER_SH = "src/scripts/lib/container_agent.sh"
FIGURE_QUALITY_SH = "src/scripts/lib/paper_figure_quality.sh"
DOCKERFILE = "src/docker/agent-cli/Dockerfile"
PDF_BYTES = b"%PDF-1.4\n%%EOF\n"
PROMPT_PAYLOAD = {"panels": [{"id": "overview"}]}
RC125_REASON = "generate-concept-figure exited non-zero (rc=125)"
BUDGET_EXHAUSTED = "concept-figure workflow deadline budget exhausted"


def _text(rel: str) -> str:
    return Path(rel).read_text(encoding="utf-8")


def _fake_proc(output: str, returncode: int = 0) -> Mock:
    proc = Mock()
    proc.communicate.return_value = (output, "")
    proc.returncode = returncode
    return proc


def _write_valid_png(path: Path, *, width: int = 2, height: int = 2) -> None:
    def chunk(chunk_type: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(payload, zlib.crc32(chunk_type)) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", checksum)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _env(**extra: str) -> dict[str, str]:
    return os.environ | {
        "CONCEPT_HELPER": str(Path(CONCEPT_SH).resolve()),
        "CONTAINER_HELPER": str(Path(CONTAINER_SH).resolve()),
        "TEST_REPO_ROOT": str(Path.cwd()),
        "TEST_PYTHON": sys.executable,
    } | extra


def _bash(script: str, env: dict[str, str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    kwargs.setdefault("check", True)
    return subprocess.run(["bash", "-c", script], env=env, **kwargs)


def _bash_capture(script: str, env: dict[str, str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return _bash(script, env, capture_output=True, text=True, **kwargs)


def _paper_dirs(root: Path) -> tuple[Path, Path, Path]:
    output_dir = root / "paper"
    figures = output_dir / "figures"
    state_dir = root / "state"
    figures.mkdir(parents=True)
    state_dir.mkdir()
    return output_dir, figures, state_dir


def _spec(figure_id: str = "overview", purpose: str | None = "Show the method.", **extra: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {"figure_id": figure_id, "preferred_generator": "generate-concept-figure"}
    if purpose is not None:
        spec["purpose"] = purpose
    return spec | extra


def _write_strategy(output_dir: Path, *specs: dict[str, Any]) -> None:
    _write_json(output_dir / "paper_strategy.json", {"figure_specs": list(specs)})


def _write_manifest(output_dir: Path, *rows: dict[str, Any]) -> None:
    _write_json(output_dir / "figure_manifest.json", {"figures": list(rows)})


def _manifest_row(figure_id: str = "overview", **extra: Any) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "generator": "generate-concept-figure",
        "prompt_log_path": f"figures/{figure_id}.prompts.json",
        **extra,
    }


def _diagram_row(*, pdf: bool = False, content: str = "Method", **extra: Any) -> dict[str, Any]:
    files = {"png": "figures/overview.png"}
    if pdf:
        files["pdf"] = "figures/overview.pdf"
    return {
        "figure_id": "overview",
        "figure_type": "diagram",
        "files": files,
        "panel_count": 1,
        "panels": [{"panel_id": "overview", "content": content}],
        "uses_progression": False,
        "min_font_size_pt": 7.0,
        "generator": "generate-concept-figure",
        "prompt_log_path": "figures/overview.prompts.json",
        **extra,
    }


def _seed_trusted_overview(output_dir: Path, figures: Path) -> None:
    _write_strategy(output_dir, _spec())
    (figures / "overview.png").write_bytes(b"trusted")
    _write_json(figures / "overview.prompts.json", PROMPT_PAYLOAD)
    _write_manifest(output_dir, _diagram_row())


# Shared bash preludes for driving concept_figure.sh functions directly.
SRC = 'source "$CONCEPT_HELPER"; OUTPUT_DIR="$TEST_OUTPUT_DIR"; '
REPO = 'REPO_ROOT="$TEST_REPO_ROOT"; PYTHON_BIN="$TEST_PYTHON"; '
STATE = (
    'CONCEPT_FIGURE_RENDERED_DIR="$TEST_STATE_DIR"; '
    'CONCEPT_FIGURE_MARKER_PATH="$OUTPUT_DIR/.concept_figure_complete"; '
    "CONCEPT_FIGURE_LOG_PREFIX=test; "
)
PRIVATE_FAILURE = (
    "CONCEPT_FIGURE_ATTEMPTED=1; CONCEPT_FIGURE_FAILURE_ACTIVE=1; CONCEPT_FIGURE_FAILURE_ID=overview; "
    f"CONCEPT_FIGURE_FAILURE_REASON={shlex.quote(RC125_REASON)}; "
)
RENDER_STUBS = (
    'CONCEPT_FIGURE_RUN_LOG="$OUTPUT_DIR/concept_figure_run.log"; AGENT=test; CONTAINERIZE=0; '
    "CONTAINER_IMAGE=test; CONTAINER_NETWORK=none; CONTAINER_HOME_DIR=/tmp; CONTAINER_ENV_FILE=; RESULTS_DIR=; "
    "CONCEPT_FIGURE_TIMEOUT_SECONDS=1; CONCEPT_FIGURE_ATTEMPTED=0; CONCEPT_FIGURE_REQUEST_FINGERPRINT=; "
    "CONCEPT_FIGURE_FAILURE_ACTIVE=0; "
    'prepare_concept_figure_render_workspace() { CONCEPT_FIGURE_RENDER_WORK_DIR="$OUTPUT_DIR"; }; '
    "select_concept_figure_backend() { CONCEPT_FIGURE_SECRET_KEYS=; CONCEPT_FIGURE_RUNTIME_CONFIG_KEYS=; "
    "CONCEPT_FIGURE_RUNTIME_KEYS=; }; "
    "prepare_concept_figure_sandbox() { CONCEPT_FIGURE_SCOPED_ENV_FILE=; CONCEPT_FIGURE_CONTAINER_HOME_DIR=/tmp; }; "
    "publish_concept_figure_outputs() { :; }; rollback_concept_figure_outputs() { :; }; "
    'build_concept_figure_command() { CONCEPT_FIGURE_COMMAND="$1"; }; '
    "persist_concept_figure_outcome() { :; }; publish_concept_marker() { :; }; "
)
STAGING_STUBS = (
    'CONCEPT_FIGURE_RUN_LOG="$OUTPUT_DIR/concept_figure_run.log"; AGENT=test; CONTAINER_IMAGE=test; '
    "CONCEPT_FIGURE_ATTEMPTED=0; CONCEPT_FIGURE_REQUEST_FINGERPRINT=; CONCEPT_FIGURE_FAILURE_ACTIVE=0; "
    "select_concept_figure_backend() { CONCEPT_FIGURE_SECRET_KEYS=; CONCEPT_FIGURE_RUNTIME_CONFIG_KEYS=; "
    "CONCEPT_FIGURE_RUNTIME_KEYS=; CONCEPT_FIGURE_SAFE_SETTING_KEYS=; }; "
    "prepare_concept_figure_sandbox() { CONCEPT_FIGURE_SCOPED_ENV_FILE=; CONCEPT_FIGURE_CONTAINER_HOME_DIR=; }; "
    "build_concept_figure_command() { CONCEPT_FIGURE_COMMAND=true; }; "
)
AUDIT_REPAIR_SCRIPT = (
    SRC + REPO + STATE + "CONCEPT_FIGURE_FAILURE_ACTIVE=0; "
    "CONCEPT_FIGURE_REPAIR_AUTHORIZED=0; CONCEPT_FIGURE_FALLBACK_ACTIVE=0; "
    'request="$(resolve_concept_figure_request_identity)"; '
    "IFS=$'\\t' read -r _ _ _ fingerprint <<<\"$request\"; "
    'CONCEPT_FIGURE_ATTEMPTED=1; CONCEPT_FIGURE_REQUEST_FINGERPRINT="$fingerprint"; '
    "authorize_concept_figure_repair; run_concept_figure_step; reassert_concept_figure"
)
INSTRUCTION_FAILURE_SCRIPT = (
    SRC + STATE + REPO + "CONCEPT_FIGURE_ATTEMPTED=1; "
    "CONCEPT_FIGURE_REQUEST_FINGERPRINT=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; "
    "CONCEPT_FIGURE_FAILURE_ACTIVE=1; CONCEPT_FIGURE_FAILURE_ID=overview; "
    'CONCEPT_FIGURE_FAILURE_REASON="old request failed"; '
    'prepare_concept_figure_render_workspace() { CONCEPT_FIGURE_RENDER_WORK_DIR="$TEST_STATE_DIR"; }; '
    "mktemp() { return 1; }; "
    'persist_concept_figure_outcome() { printf "%s\\n" "$@" > "$TEST_OUTCOME"; }; '
    'run_command_prompt() { printf called > "$TEST_CALLS"; }; '
    "reassert_concept_figure_failure() { :; }; reassert_concept_figure() { :; }; "
    "compile_paper_for_concept_figure() { :; }; "
    "prepare_concept_figure_for_validation 3 refresh"
)
FALLBACK_REQUEST_SCRIPT = (
    SRC + REPO + STATE + RENDER_STUBS + "configure_concept_figure_budget() { :; }; "
    "commit_concept_figure_outputs() { :; }; "
    'run_command_prompt() { printf "overview\\n" >> "$TEST_CALLS"; '
    'cp "$TEST_RENDERED_PNG" "$OUTPUT_DIR/figures/overview.png"; }; '
    "run_concept_figure_step"
)
SYMLINK_RECOVERY_SCRIPT = (
    SRC + STATE + REPO + RENDER_STUBS + 'run_command_prompt() { printf "overview\\n" >> "$TEST_CALLS"; '
    'cp "$TEST_VALID_PNG" "$OUTPUT_DIR/figures/overview.png"; '
    "printf '{\"panels\":[{\"id\":\"overview\"}]}' > \"$OUTPUT_DIR/figures/overview.prompts.json\"; }; "
    "reassert_concept_figure_failure() { :; }; compile_paper_for_concept_figure() { :; }; "
    "prepare_concept_figure_for_validation 2 refresh; "
    'ln -sf "$TEST_SENTINEL" "$OUTPUT_DIR/figures/overview.png"; '
    "prepare_concept_figure_for_validation 3 refresh"
)
DESTINATION_SAFETY_SCRIPT = (
    SRC + STATE + REPO + "CONCEPT_FIGURE_ATTEMPTED=0; CONCEPT_FIGURE_REQUEST_FINGERPRINT=; "
    "CONCEPT_FIGURE_FAILURE_ACTIVE=0; "
    'run_command_prompt() { printf called > "$TEST_CALLS"; '
    'printf overwritten > "$OUTPUT_DIR/figures/overview.png"; '
    'printf overwritten > "$OUTPUT_DIR/figures/overview.prompts.json"; }; '
    "run_concept_figure_step"
)
REQUEST_IDENTITY_SCRIPT = (
    SRC + STATE + REPO + RENDER_STUBS + "commit_concept_figure_outputs() { :; }; "
    'run_command_prompt() { id="${2#figures/}"; id="${id%.png}"; '
    'printf "%s\\n" "$id" >> "$TEST_CALLS"; printf rendered > "$OUTPUT_DIR/figures/$id.png"; }; '
    "reassert_concept_figure_failure() { :; }; reassert_concept_figure() { :; }; "
    "compile_paper_for_concept_figure() { :; }; "
    "prepare_concept_figure_for_validation 2 refresh; "
    'cp "$TEST_REPAIRED_STRATEGY" "$OUTPUT_DIR/paper_strategy.json"; '
    "prepare_concept_figure_for_validation 3 refresh; "
    "prepare_concept_figure_for_validation 4 refresh"
)



# ---------------------------------------------------------------------------
# run_paper_generation
# ---------------------------------------------------------------------------


def _run_paper(**overrides: Any) -> tuple[Any, Any]:
    kwargs: dict[str, Any] = dict(
        agent="codex",
        task_dir=Path("/tmp/task"),
        results_dir=Path("/tmp/results"),
        output_dir=Path("/tmp/results/paper/final"),
        instruct_path=Path("/tmp/paper.md"),
        timeout_seconds=60,
    )
    kwargs.update(overrides)
    with patch(POPEN, return_value=_fake_proc("Paper directory: /tmp/paper_final\n")) as popen_mock:
        result = investigation_cli.run_paper_generation(**kwargs)
    assert result.exit_code == 0
    return result, popen_mock.call_args


# (name, run_paper_generation overrides, expected flag, expected value following the flag or None)
PAPER_CLI_FLAG_CASES: list[tuple[str, dict[str, Any], str, str | None]] = [
    ("explicit_global_memory_dir", {"global_memory_dir": Path("/tmp/shared_memory")}, "--global-memory-dir", "/tmp/shared_memory"),
    ("global_memory_dir_defaults_to_results", {}, "--global-memory-dir", "/tmp/results"),
    (
        "explicit_skill_selection",
        {
            "agent": "claude",
            "instruct_path": Path("/tmp/paper_standalone.md"),
            "paper_skills": ("retrieve-references", "write-paper-standalone", "validate-paper", "memory-log"),
        },
        "--skills",
        "retrieve-references,write-paper-standalone,validate-paper,memory-log",
    ),
    ("two_stage_flag", {"agent": "claude", "two_stage": True}, "--two-stage", None),
    (
        "paper_profile",
        {"agent": "claude", "instruct_path": Path("/tmp/paper_standalone.md"), "paper_profile": "standalone"},
        "--paper-profile",
        "standalone",
    ),
    (
        "contributions_file",
        {"agent": "claude", "contributions_path": Path("/tmp/contributions.md")},
        "--contributions-file",
        "/tmp/contributions.md",
    ),
    ("review_rounds", {"agent": "claude", "paper_review_rounds": 2}, "--paper-review-rounds", "2"),
    ("max_repair_rounds", {"agent": "claude", "paper_max_repair_rounds": 4}, "--paper-max-repair-rounds", "4"),
]


def test_run_paper_generation_forwards_workflow_timeout_and_cli_flags() -> None:
    # the workflow timeout is forwarded in the environment
    _, call = _run_paper(timeout_seconds=900)
    assert call.kwargs["env"]["PAPER_WORKFLOW_TIMEOUT_SECONDS"] == "900"

    # each option maps to its CLI flag
    for name, overrides, flag, value in PAPER_CLI_FLAG_CASES:
        _, call = _run_paper(**overrides)
        cmd = call.args[0]
        assert flag in cmd, name
        if value is not None:
            assert cmd[cmd.index(flag) + 1] == value, name


def _final_paper_fixture(root: Path) -> tuple[Path, Path]:
    results_dir = root / "results"
    output_dir = results_dir / "paper" / "final"
    output_dir.mkdir(parents=True)
    (output_dir / "paper.pdf").write_bytes(b"%PDF-1.4\n")
    (results_dir / "paper" / "latest_manifest.json").write_text("{}\n", encoding="utf-8")
    return results_dir, output_dir


def _run_final_paper(root: Path, results_dir: Path, output_dir: Path, **overrides: Any) -> Any:
    with patch(POPEN, return_value=_fake_proc(f"Paper directory: {output_dir}\n")):
        return investigation_cli.run_paper_generation(
            agent="codex",
            task_dir=root / "task",
            results_dir=results_dir,
            output_dir=output_dir,
            instruct_path=root / "paper.md",
            timeout_seconds=60,
            **overrides,
        )


def test_run_paper_generation_final_ai_review_completed_failed_repo_env_and_default_skip(tmp_path: Path) -> None:
    # a successful final paper is reviewed
    root = tmp_path / "completed"
    results_dir, output_dir = _final_paper_fixture(root)
    review_result = Mock(
        status="completed", review_path=output_dir / "ai_review.json", log_path=output_dir / "ai_review.log", error=None
    )
    with (
        patch(f"{CLI}.run_final_ai_review", return_value=review_result) as review,
        patch(f"{CLI}.log_event") as log_event_mock,
    ):
        result = _run_final_paper(
            root, results_dir, output_dir, container_env_file=root / "container.env", final_ai_review="openai"
        )
    assert result.exit_code == 0
    assert result.ai_review_status == "completed"
    assert result.ai_review_path == output_dir / "ai_review.json"
    review.assert_called_once_with(
        repo_root=Path(investigation_cli.__file__).resolve().parents[2],
        paper_dir=output_dir,
        results_dir=results_dir,
        execute_command=ANY,
        env_file=root / "container.env",
    )
    log_event_mock.assert_any_call(
        "paper_ai_review_completed",
        paper_dir=output_dir,
        review_path=output_dir / "ai_review.json",
        log_path=output_dir / "ai_review.log",
    )

    # a review failure stays non-fatal
    root = tmp_path / "failed"
    results_dir, output_dir = _final_paper_fixture(root)
    review_result = Mock(
        status="failed", review_path=None, log_path=output_dir / "ai_review.log", error="OpenAI request failed"
    )
    with (
        patch(f"{CLI}.run_final_ai_review", return_value=review_result),
        patch(f"{CLI}.log_event") as log_event_mock,
    ):
        result = _run_final_paper(root, results_dir, output_dir, final_ai_review="openai")
    assert result.exit_code == 0
    assert result.ai_review_status == "failed"
    assert result.ai_review_path is None
    log_event_mock.assert_any_call(
        "paper_ai_review_failed",
        paper_dir=output_dir,
        reason="OpenAI request failed",
        log_path=output_dir / "ai_review.log",
    )

    # the repo .env is used for the final review when no container env file is given
    root = tmp_path / "repo-env"
    repo_root = root / "repo"
    repo_root.mkdir(parents=True)
    (repo_root / ".env").write_text("OPENAI_API_KEY=review-key\n", encoding="utf-8")
    results_dir, output_dir = _final_paper_fixture(root)
    review_result = Mock(
        status="completed", review_path=output_dir / "ai_review.json", log_path=output_dir / "ai_review.log", error=None
    )
    with (
        patch.object(investigation_cli, "__file__", str(repo_root / "src" / "investigation" / "cli.py")),
        patch(f"{CLI}.run_final_ai_review", return_value=review_result) as review,
        patch(f"{CLI}.log_event"),
    ):
        _run_final_paper(root, results_dir, output_dir, final_ai_review="openai")
    assert review.call_args.kwargs["env_file"] == repo_root / ".env"

    # the final review is skipped by default
    root = tmp_path / "default"
    results_dir, output_dir = _final_paper_fixture(root)
    with patch(f"{CLI}.run_final_ai_review") as review, patch(f"{CLI}.log_event"):
        result = _run_final_paper(root, results_dir, output_dir)
    review.assert_not_called()
    assert result.ai_review_status is None


# ---------------------------------------------------------------------------
# Repository / image / shell script text checks
# ---------------------------------------------------------------------------


def test_autoillustrate_is_not_vendored_and_image_pins_release_and_toolchain() -> None:
    # autoillustrate is not vendored as a submodule
    assert not Path(".gitmodules").exists()
    assert not Path("src/agents/skills/generate-concept-figure/drawio-figure").exists()

    # the native install script pins the image autoillustrate release
    dockerfile_text = _text(DOCKERFILE)
    script_text = _text("scripts/install_autoillustrate.sh")
    match = re.search(r"^ARG AUTOILLUSTRATE_VERSION=(\S+)$", dockerfile_text, re.MULTILINE)
    assert match is not None, "Dockerfile must declare ARG AUTOILLUSTRATE_VERSION"
    assert f'AUTOILLUSTRATE_VERSION="${{AUTOILLUSTRATE_VERSION:-{match.group(1)}}}"' in script_text
    assert '"autoillustrate@${AUTOILLUSTRATE_VERSION}"' in script_text

    # the agent image installs the pinned autoillustrate release and toolchain
    assert "drawio-figure" not in dockerfile_text
    assert "--json --usage" not in dockerfile_text
    for fragment in (
        "ARG AUTOILLUSTRATE_VERSION=0.4.8",
        "ARG AUTOILLUSTRATE_HOME=/usr/local/lib/node_modules/autoillustrate",
        'npm install -g "autoillustrate@${AUTOILLUSTRATE_VERSION}"',
        "@openai/codex@0.153.4",
        # Autoillustrate >= 0.4.4 defaults to claude-opus-5-5, unsupported before 2.1.280.
        "@anthropic-ai/claude-code@2.1.285",
        "@earendil-works/pi-coding-agent@0.85.1",
        "opencode-ai@1.18.29",
        '-r "$AUTOILLUSTRATE_HOME/requirements.txt"',
        'test -r "$AUTOILLUSTRATE_HOME/prompts/phase1.md"',
        'test -r "$AUTOILLUSTRATE_HOME/scripts/compose.py"',
        'test -r "$AUTOILLUSTRATE_HOME/assets/panel_geometry.py"',
        "ARG HEADLESS_COMMIT=93cd9b06b85f848af1308c41e018991b33907c5e",
        'git -C /opt/headless checkout --detach "$HEADLESS_COMMIT"',
        "npm ci --ignore-scripts --prefix /opt/headless",
        "npm run build --prefix /opt/headless",
        "npm prune --ignore-scripts --omit=dev --prefix /opt/headless",
        "headless --version",
        "for agent in codex claude opencode pi",
        'headless "$agent" --debug --usage --allow yolo --print-command',
        "ENV HEADLESS_BIN=/usr/local/bin/headless",
        "AUTOILLUSTRATE_HEADLESS_BIN=/usr/local/bin/headless",
        "ENV AUTOILLUSTRATE_CHROME=/usr/bin/chromium",
        "ENV VENV_PY=/opt/autoillustrate-venv/bin/python",
        "ARG TARGETARCH",
        "ARG DRAWIO_SHA256_AMD64=cb44a6770c7dcaef9adbf370beb365b740bde8c4c6e2de9d7a56824cd9ea49af",
        "ARG DRAWIO_SHA256_ARM64=cd57f501853846834a7f10f85c39ab5217f4aeb71320f309ccc5b02dcabfaa83",
        'drawio_arch="${TARGETARCH:-$(dpkg --print-architecture)}"',
        "drawio-${drawio_arch}-${DRAWIO_VERSION}.deb",
        'echo "${drawio_sha256}  /tmp/drawio.deb" | sha256sum -c -',
        'pip install --no-cache-dir "scholarly==1.7.11" "httpx==0.27.2"',
    ):
        assert fragment in dockerfile_text, fragment
    assert dockerfile_text.count('test "$(autoillustrate --version)" = "$AUTOILLUSTRATE_VERSION"') == 2
    install_doc_text = _text("docs/install.md")
    assert (
        "npx -y git+https://github.com/RobertTLange/headless-cli.git#93cd9b06b85f848af1308c41e018991b33907c5e"
        in install_doc_text
    )

    # the docker build setup needs no submodule checkout
    for text in (_text("README.md"), install_doc_text):
        assert "git submodule update" not in text
        assert "docker build -t agentic-scientist/agent-cli:latest" in text
    assert "ARG AUTOILLUSTRATE_VERSION" in install_doc_text


def test_shell_entrypoints_avoid_bash4_lowercase_and_keep_skill_step_and_author_block() -> None:
    # shell entrypoints avoid the bash4-only lowercase expansion
    for rel_path in ("run.sh", "idea.sh", "task.sh", "variant_agent.sh", "posthoc_agent.sh", "paper.sh"):
        assert "${AGENT_MODEL,,}" not in _text(f"src/scripts/{rel_path}"), rel_path

    # variant_agent supports host-evaluated steps
    script_text = _text("src/scripts/variant_agent.sh")
    assert "--skill-step" in script_text
    assert "selected_skills = skills_for_step(skill_step)" in script_text

    # paper_edit preserves the seeded author block
    assert 'if [ -f "$OUTPUT_DIR/paper_authors.tex" ]; then' in _text(PAPER_EDIT_SH)


# name -> (fragments that must be present in paper.sh, fragments that must be absent)
PAPER_SH_EXPECTATIONS: dict[str, tuple[list[str], list[str]]] = {
    "avoids_bash4_only_mapfile": ([], ["mapfile"]),
    "can_terminate_lingering_successful_stage_agents": (
        [
            "stage_expected_outputs_ready()",
            "stage_log_reports_success()",
            "produced required outputs and success result while agent still running",
            'terminate_pid_tree "$AGENT_PID" 0.2',
        ],
        [],
    ),
    "captures_review_gate_exit_code_before_branching": (
        [
            'run_paper_review_gate "$((review_round + 1))"',
            "review_stage_rc=$?",
            'if [ "$review_round" -lt "$PAPER_REVIEW_ROUNDS" ]; then',
            'if [ "$review_stage_rc" -ne 0 ]; then',
            "continue",
        ],
        ['if run_paper_review_gate "$((review_round + 1))"; then'],
    ),
    "can_run_review_rounds_even_when_validation_fails": (
        [
            "validate_paper_with_feedback",
            "validation_stage_rc=$?",
            "validation failed before review round",
            'if [ "$validation_stage_rc" -eq 0 ]; then',
        ],
        ["if validate_paper_with_feedback; then"],
    ),
    "repairs_validation_failures_before_marking_incomplete": (
        [
            'if [ "$validation_stage_rc" -ne 0 ]; then',
            "paper validation failed; starting repair round",
            "repair_round=$((repair_round + 1))",
            "continue",
        ],
        [],
    ),
    "evidence_audit_requires_containerized_readonly_execution": (
        [
            'if [ "$CONTAINERIZE" -ne 1 ]; then',
            "evidence audit read-only stage requires --containerize",
            "AGENT_WORKDIR_READONLY=1",
        ],
        [],
    ),
    "evidence_audit_workspace_lifecycle_uses_validated_helper": (
        ["prepare-audit-workspace", "cleanup-readonly-workspace"],
        [
            'mktemp -d "$readonly_root',
            'rmdir -- "$readonly_parent"',
            'readonly_root="/tmp/pi-work/agentic-evidence-audit"',
        ],
    ),
    "moves_prior_review_json_out_of_the_way_before_review_stage": (
        [
            'review_json_backup_path="$OUTPUT_DIR/.paper_review_previous.json"',
            'mv "$REVIEW_JSON_PATH" "$review_json_backup_path"',
            'if [ ! -s "$REVIEW_JSON_PATH" ] && [ -f "$review_json_backup_path" ]; then',
        ],
        [],
    ),
    "clears_stale_review_feedback_before_new_round": (
        [
            "clear_stale_review_artifacts()",
            'rm -f "$REVIEW_STAGE_FEEDBACK_JSON_PATH"',
            "clear_review_feedback_artifacts",
            "clear_stale_review_artifacts",
        ],
        [],
    ),
    "saves_pdf_before_review_repair_round": (
        [
            'REVISIONS_DIR="$OUTPUT_DIR/revisions"',
            'save_review_revision_pdf "$review_round" "$((repair_round + 1))"',
            "review_round_$(printf '%02d' \"$review_round\")_before_repair_round_$(printf '%02d' \"$next_repair_round\").pdf",
        ],
        [],
    ),
    "continues_into_repair_after_second_invalid_review_artifact": (
        [
            "retrying review stage after contract failure",
            "review artifact remained invalid after $review_contract_attempt attempts; continuing into manuscript repair without structured review feedback",
            "clear_review_feedback_artifacts",
        ],
        ['return "$finalize_rc"'],
    ),
    "writes_reference_retrieval_log": (["reference_retrieval.log", "run_reference_retrieval_with_log"], []),
    "attempts_openalex_fallback": (["openalex_fetch.py", "attempting OpenAlex fallback"], []),
    "accepts_venue_file_argument": (["--venue-file", "VENUE.md"], []),
    "accepts_contributions_file_argument": (["--contributions-file", "contributions.md"], []),
    "restages_venue_before_skill_and_prompt_rendering": (
        [
            "stage_paper_skills() {",
            '  local selected_skills_csv="$1"',
            '  local main_prompt_path="$2"',
            '  local include_review_skill="$3"',
            'stage_venue_file "$VENUE_SOURCE_PATH" "$PAPER_FORMAT" "$PAPER_MIN_REFS" "$PAPER_MAX_REFS"',
            'stage_contributions_file "$CONTRIBUTIONS_SOURCE_PATH" "$PAPER_FORMAT" "$PAPER_MIN_REFS" "$PAPER_MAX_REFS"',
            '  stage_author_block_file "$PAPER_FORMAT"',
            "render_prompt() {",
            '  local template_path="$1"',
        ],
        [],
    ),
    "runs_one_shot_postdraft_reference_enrichment_after_first_draft": (
        [
            "PAPER_REFERENCE_GAPS_INSTRUCT_PATH",
            "postdraft_reference_query_plan.json",
            "run_postdraft_reference_enrichment",
            'if [ "$repair_round" -eq 0 ] && [ ! -f "$POSTDRAFT_REFERENCE_ENRICHMENT_MARKER_PATH" ]; then',
            'touch "$POSTDRAFT_REFERENCE_ENRICHMENT_MARKER_PATH"',
        ],
        [],
    ),
    "removes_stale_skills_when_prompt_family_switches": (
        [
            "for existing_dir in formatted_skills_dir.iterdir()",
            "existing_dir.name not in selected_skill_set",
            "shutil.rmtree(existing_dir)",
        ],
        [],
    ),
    "evidence_audit_uses_exclusive_temp_workspace": (
        [
            "prepare-audit-workspace",
            '--workspace-tag "${TIMESTAMP}-${workspace_tag}"',
            'ACTIVE_READONLY_AUDIT_WORKSPACE="$readonly_dir"',
            'prepare_readonly_paper_audit_workspace "$audit_stage_name" readonly_workspace',
            "refusing to replace an audit workspace whose cleanup failed",
            'cleanup_readonly_paper_audit_workspace "$ACTIVE_READONLY_AUDIT_WORKSPACE"',
        ],
        ['readonly_workspace="$(prepare_readonly_paper_audit_workspace'],
    ),
}


def test_paper_script_contains_expected_fragments() -> None:
    script_text = _text(PAPER_SH)
    for name in sorted(PAPER_SH_EXPECTATIONS):
        present, absent = PAPER_SH_EXPECTATIONS[name]
        for fragment in present:
            assert fragment in script_text, f"{name}: {fragment}"
        for fragment in absent:
            assert fragment not in script_text, f"{name}: {fragment}"

    # shared fragments repeat at least as often as expected
    for fragment, min_count in (('--source-dir "$OUTPUT_DIR"', 2), ('paper_workspace_context_replacements(Path("."))', 2)):
        assert script_text.count(fragment) >= min_count, fragment


# name -> (present, absent) fragments expected in both paper.sh and paper_edit.sh
PAPER_SCRIPTS_EXPECTATIONS: dict[str, tuple[list[str], list[str]]] = {
    "stop_repairing_unchanged_validation_errors": (
        [
            "validation_failure_fingerprint",
            "previous_validation_failure_fingerprint",
            "validation errors and inputs were unchanged after repair",
        ],
        [],
    ),
    "feed_rendered_prompts_through_prompt_files": (
        [
            "Rendered paper prompts can exceed Linux",
            'AGENT_PROMPT_FILE="$agent_prompt_file"',
            'agent_prompt_file="$(relative_agent_prompt_path "$prompt_path" "$OUTPUT_DIR")"',
        ],
        [],
    ),
    "strategy_gates_require_substantive_supplementary_plan": (
        [
            'supplementary_plan = payload.get("supplementary_plan")',
            "paper_strategy.json must contain a supplementary_plan object",
            'supplementary_plan.get("purpose"',
            'supplementary_plan.get("planned_subsections"',
            "planned_subsections[{idx}] must be an object",
            "planned_subsections[{idx}].source_ids must be a non-empty list",
            'paper_data_path = output_dir / "paper_data.json"',
            'paper_data.get("grounding_sources")',
            "contains unknown grounding source ids",
        ],
        [],
    ),
    "posthoc_ranking_contract_repairs_before_augmentation": (
        [
            "validate_posthoc_ranking_with_feedback()",
            "validate_posthoc_ranking",
            "POSTHOC_RANKING_VALIDATION_FEEDBACK_JSON_PATH",
            "posthoc_ranking_repair_round",
            "run_prompt_stage_with_feedback",
            "write_stage_contract_feedback",
        ],
        ["posthoc ranking failed; continuing without posthoc analyses"],
    ),
    "copy_template_support_files": (
        ['"*.cls"', '"*.png"', '"*.eps"', "paper-format must name a venue under configs/venue"],
        [],
    ),
    "guard_env_file_key_detection_with_containerize": (['[ "$CONTAINERIZE" -eq 1 ]', "env_file_has_nonempty_key"], []),
    "allow_one_targeted_figure_quality_repair": (
        [
            'PAPER_FIGURE_QUALITY_REPAIR_INSTRUCT_PATH="$REPO_ROOT/src/prompts/paper_figure_quality_repair.md"',
            "figure_quality_repair_attempted=0",
            "has_actionable_figure_quality_warnings",
            "figure-quality warnings remain after one targeted repair",
            "targeted figure-quality repair requires containerization for safe rollback",
            "targeted figure-quality repair failed; restored",
            "targeted figure-quality repair regressed validation; restored",
        ],
        [],
    ),
    "source_shared_concept_figure_helper": (['source "$SCRIPT_DIR/lib/concept_figure.sh"'], []),
}


def test_paper_scripts_contain_expected_fragments_and_stop_repairing_unchanged_validation_errors() -> None:
    # both workflows contain the shared fragments
    for script_path in (PAPER_SH, PAPER_EDIT_SH):
        script_text = _text(script_path)
        for name in sorted(PAPER_SCRIPTS_EXPECTATIONS):
            present, absent = PAPER_SCRIPTS_EXPECTATIONS[name]
            for fragment in present:
                assert fragment in script_text, f"{script_path}: {name}: {fragment}"
            for fragment in absent:
                assert fragment not in script_text, f"{script_path}: {name}: {fragment}"

    # unchanged validation errors stop repairs after review and audit
    guard = 'if [ "$validation_failure_repeated" -eq 1 ]; then'
    for script_path in (PAPER_SH, PAPER_EDIT_SH):
        script_text = _text(script_path)
        review_index = script_text.index('if [ "$review_round" -lt "$PAPER_REVIEW_ROUNDS" ]; then')
        assert review_index < script_text.index(guard, review_index)

    paper_script = _text(PAPER_SH)
    audit_index = paper_script.index('paper_gate_start "evidence_audit"')
    assert audit_index < paper_script.index(guard, audit_index)


def test_paper_evidence_audit_script_structure() -> None:
    script_text = _text(PAPER_SH)

    # the audit prompt is locked before the read-only workspace is prepared
    prompt_write = 'write_agent_prompt_file "$audit_prompt_path" "$audit_prompt"'
    assert prompt_write in script_text
    assert script_text.index(prompt_write) < script_text.index(
        'prepare_readonly_paper_audit_workspace "$audit_stage_name"'
    )
    assert '> "$stage_work_dir/$agent_prompt_file"' not in script_text

    # the mismatch return reaches the repair loop
    start = script_text.index('if [ "$finalize_rc" -eq 2 ]; then')
    mismatch_block = script_text[start : script_text.index("    write_stage_contract_feedback", start)]
    assert "agentic evidence audit found mismatches" in mismatch_block
    assert re.search(r"set \+e\s+return 2", mismatch_block)
    assert "evidence audit reported revision findings; starting repair round" in script_text

    # the contract is validated after an agent exit failure
    start = script_text.index("run_agentic_evidence_audit_gate() {")
    audit_gate = script_text[start : script_text.index("save_audit_revision_pdf()", start)]
    finalize_index = audit_gate.index('finalize_output="$(finalize_paper_evidence_audit 2>&1)"')
    assert finalize_index < audit_gate.index('if [ "$audit_stage_rc" -ne 0 ]; then', finalize_index)
    assert "agentic evidence audit contract accepted despite agent exit code" in audit_gate

    # the read-only stage leaves errexit alone
    readonly_stage = script_text[
        script_text.index("run_readonly_prompt_stage()") : script_text.index("run_agentic_evidence_audit_gate()")
    ]
    assert "set +e" not in readonly_stage
    assert "set -e" not in readonly_stage


def test_paper_script_review_gate_finalization_and_reference_fallback_structure() -> None:
    script_text = _text(PAPER_SH)

    # the review gate handles stage failures without an errexit escape
    start = script_text.index("run_paper_review_gate() {")
    review_block = script_text[start : script_text.index("FINAL_PAPER_PIPELINE_INCOMPLETE", start)]
    assert "if run_prompt_stage_with_feedback" in review_block
    assert "review_stage_rc=$?" in review_block
    assert re.search(r'set \+e\s+return "\$review_stage_rc"', review_block)

    # an incomplete run is rejected before finalization
    assert script_text.rindex('if [ "$FINAL_PAPER_PIPELINE_INCOMPLETE" -eq 1 ]; then') < script_text.rindex(
        "finalize_paper_artifacts("
    )

    # the scholarly fallback block imports subprocess
    for raw_block in script_text.split("<<'PY'\n")[1:]:
        block, _, _ = raw_block.partition("\nPY\n")
        if "scholarly_run = subprocess.run(" in block:
            assert "import subprocess" in block
            break
    else:
        pytest.fail("Expected scholarly fallback Python block in paper.sh")

    # OpenAlex is attempted before scholarly
    assert script_text.index("attempting OpenAlex fallback") < script_text.index("attempting scholarly fallback")


def test_paper_workflows_order_figure_quality_repair_and_concept_figure_orchestration() -> None:
    # figure-quality repair runs after the review and audit gates
    for script_path in (PAPER_SH, PAPER_EDIT_SH):
        script_text = _text(script_path)
        final_loop = script_text[script_text.rindex("\nrepair_round=0\n") :]
        quality_repair_index = final_loop.index("actionable figure-quality warnings found")
        assert final_loop.index("run_paper_review_gate") < quality_repair_index, script_path
        if script_path == PAPER_SH:
            assert final_loop.index("run_agentic_evidence_audit_gate") < quality_repair_index

    # the shared concept-figure orchestration runs before validation
    for script_path in (PAPER_SH, PAPER_EDIT_SH):
        script_text = _text(script_path)
        concept_step = script_text.index("prepare_concept_figure_for_validation")
        assert concept_step < script_text.index("validate_paper_with_feedback", concept_step)

    helper_text = _text(CONCEPT_SH)
    shared_step = helper_text.index("prepare_concept_figure_for_validation()")
    render_step = helper_text.index("run_concept_figure_step", shared_step)
    reassert_step = helper_text.index("reassert_concept_figure", render_step)
    compile_step = helper_text.index("compile_paper_for_concept_figure", reassert_step)
    assert render_step < reassert_step < compile_step


CONCEPT_SH_EXPECTATIONS: dict[str, tuple[list[str], list[str]]] = {
    "uses_selected_agent_with_autoillustrate_cli": (
        [
            "autoillustrate_args=(./paper.tex",
            "--agent ${AGENT}",
            "${CONCEPT_FIGURE_MODE_FLAGS}",
            "--prompt-file ./${instructions_name}",
            "record_autoillustrate_costs",
        ],
        ["$(cat ./${instructions_name})", "CONCEPT_FIGURE_DRAWIO_RUN_REL", "--skip-verify"],
    ),
    "renders_in_private_staging_workspace": (
        [
            "prepare_concept_figure_render_workspace",
            '"$CONCEPT_FIGURE_RENDER_WORK_DIR" \\\n      "1"',
            'publish_concept_figure_outputs "$figure_id"',
            "from src.orchestrator.paper_step import publish_concept_figure_artifacts",
        ],
        [],
    ),
    "supports_native_macos_without_gnu_timeout": (
        [
            "command -v timeout || command -v gtimeout || true",
            '\\"\\$timeout_bin\\" -k ${CONCEPT_FIGURE_KILL_GRACE_SECONDS}s \\"\\${renderer_remaining}s\\" \\"\\$autoillustrate_bin\\"',
            "no timeout command found; skipping non-gating concept figure",
        ],
        ["timeout -k 60s ${CONCEPT_FIGURE_TIMEOUT_SECONDS}s autoillustrate"],
    ),
    "forwards_selected_keys_without_argv_values": (
        [
            'AGENT_CONTAINER_SECRET_PASSTHROUGH_KEYS="$CONCEPT_FIGURE_SECRET_KEYS"',
            'AGENT_CONTAINER_PASSTHROUGH_KEYS="${CONCEPT_FIGURE_RUNTIME_CONFIG_KEYS} ${CONCEPT_FIGURE_SAFE_SETTING_KEYS}"',
            'AGENT_CONTAINER_SESSION_NAME=""',
            'AGENT_CONTAINER_EXTRA_MOUNTS=""',
            'AGENT_CONTAINER_MOUNT_DOCKER_SOCK="0"',
            'TASK_SOURCE_DIR=""',
            'AGENT_CONTAINER_DISABLE_HOST_GATEWAY="1"',
            'AGENT_CONTAINER_GPUS=""',
            '"bridge" \\',
            '"$CONCEPT_FIGURE_CONTAINER_HOME_DIR"',
            "prepare_concept_figure_sandbox",
        ],
        ['"$RESULTS_DIR" \\', '"$CONTAINER_ENV_FILE" \\'],
    ),
    "persists_failure_and_success_provenance": (
        [
            'failure "$figure_id" "generate-concept-figure exited non-zero (rc=$figure_rc)"',
            "reported success but $output_rel was missing or empty",
            'persist_concept_figure_outcome success "$figure_id"',
            'rm -f -- "$CONCEPT_FIGURE_MARKER_PATH"',
        ],
        [],
    ),
}


def test_concept_figure_helper_text_expectations() -> None:
    helper_text = _text(CONCEPT_SH)

    # expected fragments are present / absent
    for name in sorted(CONCEPT_SH_EXPECTATIONS):
        present, absent = CONCEPT_SH_EXPECTATIONS[name]
        for fragment in present:
            assert fragment in helper_text, f"{name}: {fragment}"
        for fragment in absent:
            assert fragment not in helper_text, f"{name}: {fragment}"

    # the render is stashed before the marker and the success provenance
    stash_step = helper_text.index('cp "$OUTPUT_DIR/$output_rel" "$CONCEPT_FIGURE_RENDERED_DIR/figure.png"')
    marker_step = helper_text.index("if ! publish_concept_marker", stash_step)
    success_step = helper_text.index('persist_concept_figure_outcome success "$figure_id"', marker_step)
    assert stash_step < marker_step < success_step

    # a pre-existing workspace marker is not trusted
    run_step = helper_text.split("run_concept_figure_step() {", 1)[1].split("reassert_concept_figure() {", 1)[0]
    assert "marker present" not in run_step
    assert '[ -f "$CONCEPT_FIGURE_MARKER_PATH" ]' not in run_step


# ---------------------------------------------------------------------------
# concept_figure.sh behaviour (bash integration)
# ---------------------------------------------------------------------------


def test_concept_figure_reassert_rejects_traversal_and_restores_matching_provenance(tmp_path: Path) -> None:
    # traversal is rejected without touching anything outside the output dir
    root = tmp_path / "traversal"
    output_dir, figures, state_dir = _paper_dirs(root)
    (output_dir / ".concept_figure_complete").write_text("", encoding="utf-8")
    _write_valid_png(state_dir / "figure.png")
    (state_dir / "output_rel.txt").write_text("../sentinel", encoding="utf-8")
    sentinel = root / "sentinel"
    sentinel.write_text("unchanged", encoding="utf-8")
    env = _env(TEST_OUTPUT_DIR=str(output_dir), TEST_STATE_DIR=str(state_dir))

    _bash(SRC + STATE + "if reassert_concept_figure; then exit 1; fi", env)
    assert sentinel.read_text(encoding="utf-8") == "unchanged"

    (state_dir / "output_rel.txt").write_text("figures/concept.png", encoding="utf-8")
    _write_json(state_dir / "figure.prompts.json", PROMPT_PAYLOAD)
    _write_json(figures / "concept.prompts.json", PROMPT_PAYLOAD)
    _write_manifest(output_dir, _manifest_row("concept"))
    figure_path = figures / "concept.png"
    figure_path.symlink_to(sentinel)

    _bash(SRC + REPO + STATE + "reassert_concept_figure", env)

    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert not figure_path.is_symlink()
    assert figure_path.read_bytes() == (state_dir / "figure.png").read_bytes()

    # provenance is restored when the artifacts match
    output_dir, figures, state_dir = _paper_dirs(tmp_path / "match")
    _write_valid_png(figures / "concept.png")
    _write_json(figures / "concept.prompts.json", PROMPT_PAYLOAD)
    _write_valid_png(state_dir / "figure.png")
    _write_json(state_dir / "figure.prompts.json", PROMPT_PAYLOAD)
    (state_dir / "output_rel.txt").write_text("figures/concept.png", encoding="utf-8")
    _write_manifest(
        output_dir,
        _manifest_row(
            "concept", generator=None, attempted_generator="generate-concept-figure", fallback_reason="forged failure"
        ),
    )

    _bash(SRC + REPO + STATE + "reassert_concept_figure", _env(TEST_OUTPUT_DIR=str(output_dir), TEST_STATE_DIR=str(state_dir)))

    assert (figures / "concept.png").read_bytes() == (state_dir / "figure.png").read_bytes()
    assert _read_json(figures / "concept.prompts.json") == PROMPT_PAYLOAD
    assert (figures / "concept.png.prov.json").is_file()
    row = _read_json(output_dir / "figure_manifest.json")["figures"][0]
    assert row["generator"] == "generate-concept-figure"
    assert "fallback_reason" not in row
    assert (output_dir / ".concept_figure_complete").is_file()


def test_concept_figure_private_failure_is_reasserted_and_complete_fallback_preserved(tmp_path: Path) -> None:
    # the private failure is reasserted after an agent rewrite
    output_dir, figures, state_dir = _paper_dirs(tmp_path / "rewrite")
    _write_json(figures / "overview.prompts.json", {"forged": "success"})
    _write_manifest(output_dir, _manifest_row())

    _bash(
        SRC + STATE + REPO + PRIVATE_FAILURE + "run_concept_figure_step() { :; }; "
        "compile_paper_for_concept_figure() { :; }; prepare_concept_figure_for_validation 2 refresh",
        _env(TEST_OUTPUT_DIR=str(output_dir), TEST_STATE_DIR=str(state_dir)),
    )

    prompt_log = _read_json(figures / "overview.prompts.json")
    row = _read_json(output_dir / "figure_manifest.json")["figures"][0]
    assert prompt_log["status"] == "failed"
    assert prompt_log["reason"] == RC125_REASON
    assert row["generator"] is None
    assert row["fallback_reason"] == RC125_REASON

    # a complete post-failure fallback is preserved
    root = tmp_path / "fallback"
    output_dir, figures, state_dir = _paper_dirs(root)
    _write_valid_png(figures / "overview.png")
    (figures / "overview.pdf").write_bytes(PDF_BYTES)
    _write_json(figures / "overview.png.prov.json", {"artifact": {"sha256": "stale"}})
    _write_json(
        figures / "overview.prompts.json",
        {"status": "failed", "attempted_generator": "generate-concept-figure", "reason": RC125_REASON},
    )
    fallback_reason = "Replaced failed concept render with matplotlib."
    _write_manifest(
        output_dir,
        _diagram_row(
            pdf=True,
            content="Fallback",
            generator="matplotlib",
            attempted_generator="generate-concept-figure",
            fallback_reason=fallback_reason,
        ),
    )

    _bash(
        SRC + STATE + REPO + PRIVATE_FAILURE + "run_concept_figure_step() { :; }; "
        'compile_paper_for_concept_figure() { printf called > "$TEST_COMPILE"; }; '
        "prepare_concept_figure_for_validation 2 refresh",
        _env(TEST_OUTPUT_DIR=str(output_dir), TEST_STATE_DIR=str(state_dir), TEST_COMPILE=str(root / "compile-called")),
    )

    row = _read_json(output_dir / "figure_manifest.json")["figures"][0]
    assert row["generator"] == "matplotlib"
    assert row["fallback_reason"] == fallback_reason
    assert (figures / "overview.pdf").read_bytes() == PDF_BYTES
    assert (output_dir / ".concept_figure_complete").is_file()
    assert not (figures / "overview.png.prov.json").exists()
    assert (root / "compile-called").read_text(encoding="utf-8") == "called"


PRESERVE_STUBS = (
    "run_concept_figure_step() { printf 'run\\n'; }; reassert_concept_figure_failure() { :; }; "
    "reassert_concept_figure() { :; }; compile_paper_for_concept_figure() { :; }; concept_figure_warn() { :; }; "
)


def test_concept_figure_preserve_mode_keeps_errexit_and_rerenders_only_on_mutation(tmp_path: Path) -> None:
    # the helper preserves errexit and honors preserve mode
    result = _bash_capture(
        'source "$CONCEPT_HELPER"; set +e; CONCEPT_FIGURE_ATTEMPTED=1; CONCEPT_FIGURE_PRESERVE_SNAPSHOT=trusted; '
        "capture_concept_figure_preserve_snapshot() { printf trusted; }; "
        "run_concept_figure_step() { printf 'run\\n'; }; "
        "reassert_concept_figure() { printf 'reassert\\n'; return 7; }; "
        "compile_paper_for_concept_figure() { printf 'compile\\n'; }; "
        "concept_figure_log() { :; }; concept_figure_warn() { :; }; "
        "prepare_concept_figure_for_validation 1 refresh; case $- in *e*) exit 2;; esac; "
        "prepare_concept_figure_for_validation 0 preserve; case $- in *e*) exit 3;; esac",
        _env(),
    )
    assert result.stdout.splitlines() == ["run", "reassert", "compile"]

    # a new request is rendered after the snapshot
    output_dir, _, _ = _paper_dirs(tmp_path / "new-request")
    _write_strategy(output_dir)

    result = _bash_capture(
        SRC + REPO + 'CONCEPT_FIGURE_PRESERVE_SNAPSHOT="$(capture_concept_figure_preserve_snapshot)"; '
        "printf '{\"figure_specs\":[{\"figure_id\":\"overview\",\"preferred_generator\":\"generate-concept-figure\","
        "\"purpose\":\"forged\"}]}' > \"$OUTPUT_DIR/paper_strategy.json\"; "
        'printf forged > "$OUTPUT_DIR/figures/overview.png"; '
        "printf '{\"forged\":true}' > \"$OUTPUT_DIR/figures/overview.prompts.json\"; "
        + PRESERVE_STUBS
        + "prepare_concept_figure_for_validation 1 preserve",
        _env(TEST_OUTPUT_DIR=str(output_dir)),
    )
    assert result.stdout == "run\n"

    # a mutated trusted artifact is refreshed
    output_dir, figures, _ = _paper_dirs(tmp_path / "mutated")
    _seed_trusted_overview(output_dir, figures)

    result = _bash_capture(
        SRC + REPO + 'CONCEPT_FIGURE_PRESERVE_SNAPSHOT="$(capture_concept_figure_preserve_snapshot)"; '
        + PRESERVE_STUBS
        + "concept_figure_log() { :; }; prepare_concept_figure_for_validation 1 preserve; "
        'printf mutated > "$OUTPUT_DIR/figures/overview.png"; prepare_concept_figure_for_validation 2 preserve',
        _env(TEST_OUTPUT_DIR=str(output_dir)),
    )
    assert result.stdout == "run\n"

    # a symlink mutation is recorded as a failure
    root = tmp_path / "symlink"
    output_dir, figures, _ = _paper_dirs(root)
    _seed_trusted_overview(output_dir, figures)
    sentinel = root / "sentinel.png"
    sentinel.write_bytes(b"outside")

    _bash(
        SRC + REPO + 'normalize_positive_int() { printf "%s\\n" "$2"; }; initialize_concept_figure_step test; '
        'ln -sf "$TEST_SENTINEL" "$OUTPUT_DIR/figures/overview.png"; '
        "prepare_concept_figure_for_validation 1 preserve; cleanup_concept_figure_state",
        _env(TEST_OUTPUT_DIR=str(output_dir), TEST_SENTINEL=str(sentinel)),
    )

    assert sentinel.read_bytes() == b"outside"
    prompt_log = _read_json(figures / "overview.prompts.json")
    assert prompt_log["status"] == "failed"
    assert "unsafe concept-figure render destination" in prompt_log["reason"]
    assert _read_json(output_dir / "figure_manifest.json")["figures"][0]["generator"] is None


def test_concept_figure_initialization_failure_is_non_gating(tmp_path: Path) -> None:
    for failing_command in ("mktemp", "chmod"):
        output_dir, figures, _ = _paper_dirs(tmp_path / failing_command)
        _write_strategy(output_dir, _spec())
        (figures / "overview.png").write_bytes(b"placeholder")
        (figures / "overview.prompts.json").write_text("{}", encoding="utf-8")
        _write_manifest(output_dir, _manifest_row())

        result = _bash_capture(
            'source "$CONCEPT_HELPER"; set -euo pipefail; OUTPUT_DIR="$TEST_OUTPUT_DIR"; ' + REPO
            + 'normalize_positive_int() { printf "%s\\n" "$2"; }; '
            'mktemp() { [ "$TEST_FAIL_COMMAND" != mktemp ] || return 1; command mktemp "$@"; }; '
            'chmod() { [ "$TEST_FAIL_COMMAND" != chmod ] || return 1; command chmod "$@"; }; '
            "run_concept_figure_step() { exit 91; }; initialize_concept_figure_step test; "
            'prepare_concept_figure_for_validation 1 refresh; printf "%s\\n" "$CONCEPT_FIGURE_AVAILABLE"',
            _env(TEST_FAIL_COMMAND=failing_command, TEST_OUTPUT_DIR=str(output_dir)),
        )

        assert result.stdout == "0\n", failing_command
        reason = "concept-figure private state unavailable"
        prompt_log = _read_json(figures / "overview.prompts.json")
        row = _read_json(output_dir / "figure_manifest.json")["figures"][0]
        assert prompt_log["status"] == "failed", failing_command
        assert prompt_log["reason"] == reason, failing_command
        assert row["generator"] is None, failing_command
        assert row["fallback_reason"] == reason, failing_command


# (name, initial strategy specs, repaired spec, expected renderer calls)
REQUEST_IDENTITY_CASES: list[tuple[str, list[dict[str, Any]], dict[str, Any], list[str]]] = [
    ("renders_newly_valid_repaired_request_once", [], _spec(preferred_span="full_width"), ["overview"]),
    ("rerenders_when_identity_changes", [_spec()], _spec("detail", "Show the repaired method."), ["overview", "detail"]),
]


def test_concept_figure_request_identity_drives_rerender(tmp_path: Path) -> None:
    for name, initial_specs, repaired_spec, expected_calls in REQUEST_IDENTITY_CASES:
        root = tmp_path / name
        output_dir, _, state_dir = _paper_dirs(root)
        _write_strategy(output_dir, *initial_specs)
        repaired_strategy = root / "repaired_strategy.json"
        _write_json(repaired_strategy, {"figure_specs": [repaired_spec]})
        calls_path = root / "calls"

        _bash(
            REQUEST_IDENTITY_SCRIPT,
            _env(
                TEST_OUTPUT_DIR=str(output_dir),
                TEST_STATE_DIR=str(state_dir),
                TEST_REPAIRED_STRATEGY=str(repaired_strategy),
                TEST_CALLS=str(calls_path),
            ),
        )

        assert calls_path.read_text(encoding="utf-8").splitlines() == expected_calls, name


def _run_audit_repair_case(root: Path, *, attempted_generator: bool) -> tuple[Path, Path, Path, Path]:
    output_dir, figures, state_dir = _paper_dirs(root)
    _write_strategy(output_dir, _spec())
    _write_json(
        output_dir / "paper_evidence_audit_feedback.json",
        {"verdict": "fail", "ok": False, "findings": [{"id": "method-figure-equation", "evidence": "figures/overview.png"}]},
    )
    live_png = figures / "overview.png"
    _write_valid_png(live_png, width=3, height=2)
    if attempted_generator:
        (figures / "overview.pdf").write_bytes(PDF_BYTES)
    _write_json(figures / "overview.prompts.json", PROMPT_PAYLOAD)
    trusted_png = state_dir / "figure.png"
    _write_valid_png(trusted_png)
    _write_json(state_dir / "figure.prompts.json", PROMPT_PAYLOAD)
    (state_dir / "output_rel.txt").write_text("figures/overview.png", encoding="utf-8")
    (output_dir / ".concept_figure_complete").touch()
    extra = {"attempted_generator": "generate-concept-figure"} if attempted_generator else {}
    _write_manifest(
        output_dir,
        _diagram_row(
            pdf=True,
            generator="manual",
            fallback_reason="Evidence audit required an equation correction.",
            **extra,
        ),
    )

    _bash(AUDIT_REPAIR_SCRIPT, _env(TEST_OUTPUT_DIR=str(output_dir), TEST_STATE_DIR=str(state_dir)))
    return output_dir, figures, live_png, trusted_png


def test_concept_figure_audit_repair_fallback_preserved_when_authorized_and_rendered_when_not(tmp_path: Path) -> None:
    # an authorized audit-repair fallback is preserved
    output_dir, figures, live_png, _ = _run_audit_repair_case(tmp_path / "authorized", attempted_generator=True)
    repaired_bytes = live_png.read_bytes()

    assert live_png.read_bytes() == repaired_bytes
    assert _read_json(output_dir / "figure_manifest.json")["figures"][0]["generator"] == "manual"
    provenance = _read_json(figures / "overview.png.prov.json")
    assert provenance["named_outputs"]["renderer"] == "manual"
    assert provenance["named_outputs"]["attempted_generator"] == "generate-concept-figure"

    # an unauthorized fallback does not skip the initial render
    root = tmp_path / "unauthorized"
    output_dir, figures, state_dir = _paper_dirs(root)
    _write_strategy(output_dir, _spec())
    _write_valid_png(figures / "overview.png", width=3, height=2)
    (figures / "overview.pdf").write_bytes(PDF_BYTES)
    _write_json(figures / "overview.prompts.json", PROMPT_PAYLOAD)
    _write_manifest(
        output_dir,
        _diagram_row(
            pdf=True,
            generator="manual",
            attempted_generator="generate-concept-figure",
            fallback_reason="Evidence audit required an equation correction.",
        ),
    )
    rendered_png = root / "rendered.png"
    _write_valid_png(rendered_png, width=4, height=2)
    calls_path = root / "calls"

    _bash(
        FALLBACK_REQUEST_SCRIPT,
        _env(
            TEST_OUTPUT_DIR=str(output_dir),
            TEST_STATE_DIR=str(state_dir),
            TEST_RENDERED_PNG=str(rendered_png),
            TEST_CALLS=str(calls_path),
        ),
    )

    assert calls_path.read_text(encoding="utf-8") == "overview\n"


def test_concept_figure_reassert_replaces_incomplete_audit_repair_fallback(tmp_path: Path) -> None:
    _, _, live_png, trusted_png = _run_audit_repair_case(tmp_path, attempted_generator=False)
    assert live_png.read_bytes() == trusted_png.read_bytes()


def test_concept_figure_rejects_symlink_destinations_before_render(tmp_path: Path) -> None:
    for artifact_name in ("overview.png", "overview.prompts.json"):
        root = tmp_path / artifact_name
        output_dir, figures, state_dir = _paper_dirs(root)
        _write_strategy(output_dir, _spec())
        sentinel = root / "sentinel"
        sentinel.write_bytes(b"outside")
        (figures / artifact_name).symlink_to(sentinel)
        calls_path = root / "calls"

        _bash(
            DESTINATION_SAFETY_SCRIPT,
            _env(TEST_OUTPUT_DIR=str(output_dir), TEST_STATE_DIR=str(state_dir), TEST_CALLS=str(calls_path)),
        )

        assert sentinel.read_bytes() == b"outside", artifact_name
        assert not calls_path.exists(), artifact_name


def test_concept_figure_restores_trusted_render_over_live_symlink(tmp_path: Path) -> None:
    output_dir, figures, state_dir = _paper_dirs(tmp_path)
    _write_strategy(output_dir, _spec())
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"outside")
    valid_png = tmp_path / "rendered.png"
    _write_valid_png(valid_png)
    calls_path = tmp_path / "calls"

    _bash(
        SYMLINK_RECOVERY_SCRIPT,
        _env(
            TEST_OUTPUT_DIR=str(output_dir),
            TEST_STATE_DIR=str(state_dir),
            TEST_SENTINEL=str(sentinel),
            TEST_VALID_PNG=str(valid_png),
            TEST_CALLS=str(calls_path),
        ),
    )

    assert sentinel.read_bytes() == b"outside"
    assert not (figures / "overview.png").is_symlink()
    assert (figures / "overview.png").read_bytes() == valid_png.read_bytes()
    assert calls_path.read_text(encoding="utf-8") == "overview\n"


def test_concept_figure_staging_rebinds_instruction_failure_and_rolls_back_on_provenance_failure(tmp_path: Path) -> None:
    # the failure is rebound when instruction staging fails
    root = tmp_path / "instructions"
    output_dir, _, state_dir = _paper_dirs(root)
    _write_strategy(output_dir, _spec("detail", "Show repaired detail."))
    outcome_path = root / "outcome"
    calls_path = root / "calls"

    _bash(
        INSTRUCTION_FAILURE_SCRIPT,
        _env(
            TEST_OUTPUT_DIR=str(output_dir),
            TEST_STATE_DIR=str(state_dir),
            TEST_OUTCOME=str(outcome_path),
            TEST_CALLS=str(calls_path),
        ),
    )

    assert outcome_path.read_text(encoding="utf-8").splitlines() == [
        "failure",
        "detail",
        "could not create concept-figure instruction file",
    ]
    assert not calls_path.exists()

    # the live pair is rolled back when success provenance fails
    root = tmp_path / "rollback"
    output_dir, figures, state_dir = _paper_dirs(root)
    (output_dir / "paper.tex").write_text("paper", encoding="utf-8")
    _write_strategy(output_dir, _spec(purpose=None))
    live_png = figures / "overview.png"
    rendered_png = root / "rendered.png"
    _write_valid_png(live_png)
    _write_valid_png(rendered_png, width=3, height=2)
    old_png = live_png.read_bytes()
    old_prompts = '{"panels": [{"id": "old"}]}'
    (figures / "overview.prompts.json").write_text(old_prompts, encoding="utf-8")

    _bash(
        SRC + REPO + STATE + STAGING_STUBS + 'run_command_prompt() { cp "$TEST_VALID_PNG" "$3/figures/overview.png"; '
        "printf '{\"panels\":[{\"id\":\"new\"}]}' > \"$3/figures/overview.prompts.json\"; }; "
        'persist_concept_figure_outcome() { [ "$1" != success ]; }; run_concept_figure_step',
        _env(TEST_OUTPUT_DIR=str(output_dir), TEST_STATE_DIR=str(state_dir), TEST_VALID_PNG=str(rendered_png)),
    )

    assert live_png.read_bytes() == old_png
    assert (figures / "overview.prompts.json").read_text(encoding="utf-8") == old_prompts
    assert not (output_dir / ".concept_figure_complete").exists()
    assert not (figures / ".concept-figure-transaction.json").exists()


def test_concept_figure_staging_integration_keeps_live_paper_isolated(tmp_path: Path) -> None:
    output_dir, figures, state_dir = _paper_dirs(tmp_path)
    (output_dir / "paper.tex").write_text("trusted paper", encoding="utf-8")
    _write_strategy(output_dir, _spec())
    _write_manifest(output_dir, _manifest_row())
    valid_png = tmp_path / "rendered.png"
    _write_valid_png(valid_png)
    workdir_record = tmp_path / "workdir"

    _bash(
        SRC + REPO + STATE + STAGING_STUBS + 'run_command_prompt() { printf "renderer completed\\n"; '
        'printf "%s" "$3" > "$TEST_WORKDIR_RECORD"; '
        'test "$3" != "$OUTPUT_DIR"; grep -q "trusted paper" "$3/paper.tex"; '
        'printf "renderer mutation" > "$3/paper.tex"; cp "$TEST_VALID_PNG" "$3/figures/overview.png"; '
        "printf '{\"placement\":\"half-width\",\"panels\":[{\"id\":\"overview\"}]}' "
        '> "$3/figures/overview.prompts.json"; }; run_concept_figure_step',
        _env(
            TEST_OUTPUT_DIR=str(output_dir),
            TEST_STATE_DIR=str(state_dir),
            TEST_VALID_PNG=str(valid_png),
            TEST_WORKDIR_RECORD=str(workdir_record),
        ),
    )

    staged_workdir = Path(workdir_record.read_text(encoding="utf-8"))
    assert staged_workdir != output_dir
    assert not staged_workdir.exists()
    assert (output_dir / "paper.tex").read_text(encoding="utf-8") == "trusted paper"
    assert (figures / "overview.png").read_bytes() == valid_png.read_bytes()
    assert (output_dir / ".concept_figure_complete").is_file()
    assert (output_dir / "concept_figure_run.log").read_text(encoding="utf-8") == "renderer completed\n"
    assert not (figures / ".concept-figure-transaction.json").exists()


def _budget_output(script: str) -> subprocess.CompletedProcess[str]:
    env = _env()
    for key in ("PAPER_WORKFLOW_TIMEOUT_SECONDS", "PAPER_CONCEPT_FIGURE_MODE", "PAPER_CONCEPT_FIGURE_TIMEOUT_SECONDS"):
        env.pop(key, None)
    return _bash_capture(
        'source "$CONCEPT_HELPER"; ' + script + 'printf "%s\\n%s\\n" "$CONCEPT_FIGURE_TIMEOUT_SECONDS" "$CONCEPT_FIGURE_MODE_FLAGS"',
        env,
    )


# (name, budget script, expected [timeout, mode flags], required stderr fragment or None)
BUDGET_CASES: list[tuple[str, str, list[str], str | None]] = [
    (
        "caps_timeout_and_selects_light_mode",
        "PAPER_WORKFLOW_TIMEOUT_SECONDS=900; PAPER_CONCEPT_FIGURE_TIMEOUT_SECONDS=9000; "
        "PAPER_CONCEPT_FIGURE_POST_RENDER_RESERVE_SECONDS=300; configure_concept_figure_budget 100; ",
        ["440", "--mode light"],
        None,
    ),
    ("defaults_to_full_mode", "configure_concept_figure_budget 100; ", ["9000", "--mode full"], None),
    (
        "degrades_full_mode_to_medium",
        "PAPER_WORKFLOW_TIMEOUT_SECONDS=3000; configure_concept_figure_budget 100; ",
        ["2540", "--mode medium"],
        None,
    ),
    (
        "keeps_full_mode_with_enough_deadline",
        "PAPER_WORKFLOW_TIMEOUT_SECONDS=7200; configure_concept_figure_budget 100; ",
        ["6740", "--mode full"],
        None,
    ),
    (
        "honors_preferred_mode_cap",
        "PAPER_CONCEPT_FIGURE_MODE=medium; configure_concept_figure_budget 100; ",
        ["9000", "--mode medium"],
        None,
    ),
    (
        "rejects_unknown_preferred_mode",
        "PAPER_CONCEPT_FIGURE_MODE=turbo; configure_concept_figure_budget 100; ",
        ["9000", "--mode full"],
        "invalid PAPER_CONCEPT_FIGURE_MODE 'turbo'",
    ),
    (
        "treats_leading_zeroes_as_decimal",
        "PAPER_WORKFLOW_TIMEOUT_SECONDS=0900; PAPER_CONCEPT_FIGURE_TIMEOUT_SECONDS=09000; "
        "PAPER_CONCEPT_FIGURE_POST_RENDER_RESERVE_SECONDS=0300; PAPER_CONCEPT_FIGURE_MEDIUM_MODE_MIN_SECONDS=01800; "
        "configure_concept_figure_budget 0100; ",
        ["440", "--mode light"],
        None,
    ),
]


def test_concept_figure_budget_selects_timeout_and_mode() -> None:
    for name, script, expected, stderr_fragment in BUDGET_CASES:
        result = _budget_output(script)
        assert result.stdout.splitlines() == expected, name
        if stderr_fragment:
            assert stderr_fragment in result.stderr, name


def test_concept_figure_budget_skips_when_render_cannot_finish_and_rechecks_before_render(tmp_path: Path) -> None:
    # the render is skipped when the deadline is exhausted or light mode cannot finish
    for name, workflow_timeout, elapsed in (("deadline_exhausted", 900, 600), ("light_mode_cannot_finish", 899, 300)):
        result = _bash_capture(
            f'source "$CONCEPT_HELPER"; PAPER_WORKFLOW_TIMEOUT_SECONDS={workflow_timeout}; '
            "PAPER_CONCEPT_FIGURE_POST_RENDER_RESERVE_SECONDS=300; "
            f"if configure_concept_figure_budget {elapsed}; then exit 91; fi; "
            'printf "%s\\n" "$CONCEPT_FIGURE_UNAVAILABLE_REASON"',
            _env(),
        )
        assert result.stdout.strip() == BUDGET_EXHAUSTED, name

    # the budget is rechecked immediately before the render
    (tmp_path / "figures").mkdir()
    result = _bash_capture(
        SRC + "PAPER_WORKFLOW_TIMEOUT_SECONDS=900; PAPER_CONCEPT_FIGURE_POST_RENDER_RESERVE_SECONDS=300; SECONDS=100; "
        "capture_concept_figure_preserve_snapshot() { printf trusted; }; initialize_concept_figure_step test; "
        "SECONDS=600; resolve_concept_figure_request_identity() { "
        "printf 'overview\\tfigures/overview.png\\tfull-width\\t%s\\n' "
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; }; "
        "persist_concept_figure_outcome() { printf '%s\\n' \"$*\"; }; "
        'run_command_prompt() { printf called > "$OUTPUT_DIR/renderer-called"; }; '
        "run_concept_figure_step; cleanup_concept_figure_state",
        _env(TEST_OUTPUT_DIR=str(tmp_path)),
    )

    assert result.stdout.strip() == f"failure overview {BUDGET_EXHAUSTED}"
    assert not (tmp_path / "renderer-called").exists()


def test_concept_figure_command_skips_without_timeout_binary_and_kills_term_ignoring_renderer(tmp_path: Path) -> None:
    # the render is skipped when no timeout binary is available
    root = tmp_path / "no-timeout"
    fake_bin = root / "bin"
    fake_bin.mkdir(parents=True)
    (fake_bin / "cat").symlink_to("/bin/cat")
    renderer_marker = root / "renderer-called"
    fake_renderer = fake_bin / "autoillustrate"
    fake_renderer.write_text(f"#!/bin/sh\nprintf called > {shlex.quote(str(renderer_marker))}\n", encoding="utf-8")
    fake_renderer.chmod(0o755)
    (root / "instructions").write_text("overview", encoding="utf-8")

    result = _bash_capture(
        'source "$CONCEPT_HELPER"; AGENT=codex; CONCEPT_FIGURE_BACKEND_FLAGS=--skip-enhance; '
        "CONCEPT_FIGURE_ENV_COMMANDS=:; CONCEPT_FIGURE_TIMEOUT_SECONDS=1; "
        "build_concept_figure_command figures/overview.png full-width instructions; "
        'PATH="$FAKE_BIN" /bin/bash -c "$CONCEPT_FIGURE_COMMAND"',
        _env(FAKE_BIN=str(fake_bin)),
        cwd=root,
        check=False,
    )

    assert result.returncode == 125
    assert "skipping non-gating concept figure" in result.stderr
    assert not renderer_marker.exists()

    # the timeout kills a TERM-ignoring renderer within the grace period
    root = tmp_path / "kill"
    root.mkdir()
    fake_renderer = root / "autoillustrate"
    fake_renderer.write_text("#!/bin/bash\ntrap '' TERM\nwhile :; do sleep 0.1; done\n", encoding="utf-8")
    fake_renderer.chmod(0o755)
    (root / "instructions").write_text("overview", encoding="utf-8")
    started = time.monotonic()

    result = _bash_capture(
        'source "$CONCEPT_HELPER"; AGENT=codex; CONCEPT_FIGURE_BACKEND_FLAGS=--skip-enhance; CONCEPT_FIGURE_MODE_FLAGS=; '
        "CONCEPT_FIGURE_ENV_COMMANDS='export PATH=\"$TEST_RENDERER_DIR:$PATH\"'; "
        "CONCEPT_FIGURE_TIMEOUT_SECONDS=1; CONCEPT_FIGURE_KILL_GRACE_SECONDS=1; "
        "build_concept_figure_command figures/overview.png full-width instructions; "
        '/bin/bash -c "$CONCEPT_FIGURE_COMMAND"',
        _env(PATH=f"{root}:{os.environ['PATH']}", TEST_RENDERER_DIR=str(root)),
        cwd=root,
        check=False,
        timeout=5,
    )

    assert result.returncode in {-9, 124, 137}
    assert time.monotonic() - started < 4


def test_concept_figure_backend_selection_runtime_environment_and_outer_env_file(tmp_path: Path) -> None:
    # backend selection uses the available key
    result = _bash_capture(
        'source "$CONCEPT_HELPER"; concept_figure_log() { :; }; '
        "unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_BEARER_TOKEN_BEDROCK CLAUDE_CODE_USE_BEDROCK "
        "PI_CODING_AGENT_PROVIDER; CONTAINER_ENV_FILE=; "
        "AGENT=claude; PAPER_OPENAI_API_KEY_AVAILABLE=true; PAPER_GEMINI_API_KEY_AVAILABLE=true; "
        "select_concept_figure_backend; printf 'openai=%s|%s\\n' \"$CONCEPT_FIGURE_BACKEND_FLAGS\" \"$CONCEPT_FIGURE_SECRET_KEYS\"; "
        "AGENT=codex; AGENT_PROFILE=custom; PAPER_OPENAI_API_KEY_AVAILABLE=false; "
        "select_concept_figure_backend; printf 'gemini=%s|%s\\n' \"$CONCEPT_FIGURE_BACKEND_FLAGS\" \"$CONCEPT_FIGURE_SECRET_KEYS\"; "
        "AGENT=claude; PAPER_GEMINI_API_KEY_AVAILABLE=false; "
        "select_concept_figure_backend; printf 'none=%s|%s\\n' \"$CONCEPT_FIGURE_BACKEND_FLAGS\" \"$CONCEPT_FIGURE_SECRET_KEYS\"; "
        "AGENT=cursor; select_concept_figure_backend; printf 'cursor=%s|%s\\n' \"$CONCEPT_FIGURE_SECRET_KEYS\" \"$CONCEPT_FIGURE_RUNTIME_CONFIG_KEYS\"; "
        "AGENT=pi; select_concept_figure_backend; printf 'pi=%s|%s\\n' \"$CONCEPT_FIGURE_SECRET_KEYS\" \"$CONCEPT_FIGURE_RUNTIME_CONFIG_KEYS\"; "
        "AGENT=opencode; PAPER_CONCEPT_FIGURE_AGENT_PROVIDER=anthropic; select_concept_figure_backend; "
        "printf 'opencode=%s|%s\\n' \"$CONCEPT_FIGURE_SECRET_KEYS\" \"$CONCEPT_FIGURE_SEED_AGENT_AUTH\"; "
        "AGENT=claude; CLAUDE_CODE_USE_BEDROCK=false; AWS_ACCESS_KEY_ID=ambient; AWS_SECRET_ACCESS_KEY=ambient; "
        "select_concept_figure_backend; printf 'claude-false=%s|%s\\n' \"$CONCEPT_FIGURE_SECRET_KEYS\" \"$CONCEPT_FIGURE_SEED_AGENT_AUTH\"",
        _env(),
    )
    assert result.stdout.splitlines() == [
        "openai=--enhance-model gpt-image-2 --audit-vlm gpt-5.5 --unit-test-vlm openai-single|OPENAI_API_KEY AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_BEARER_TOKEN_BEDROCK ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN CLAUDE_CODE_OAUTH_TOKEN",
        "gemini=--enhance-model nanobanana-pro --audit-vlm gemini-3.1-pro --unit-test-vlm gemini-single|GEMINI_API_KEY CODEX_API_KEY OPENAI_API_KEY",
        "none=--skip-enhance --panel-inspects 0|AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_BEARER_TOKEN_BEDROCK ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN CLAUDE_CODE_OAUTH_TOKEN",
        "cursor=CURSOR_API_KEY|CURSOR_CLI_BIN",
        "pi=AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_BEARER_TOKEN_BEDROCK|PI_CODING_AGENT_PROVIDER PI_CODING_AGENT_MODEL PI_CODING_AGENT_MODELS AWS_REGION AWS_DEFAULT_REGION",
        "opencode=ANTHROPIC_API_KEY|0",
        "claude-false=AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_BEARER_TOKEN_BEDROCK ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN CLAUDE_CODE_OAUTH_TOKEN|1",
    ]

    # the runtime environment drops unselected secrets
    result = _bash_capture(
        'source "$CONCEPT_HELPER"; concept_figure_log() { :; }; AGENT=codex; PAPER_OPENAI_API_KEY_AVAILABLE=true; '
        'PAPER_GEMINI_API_KEY_AVAILABLE=false; select_concept_figure_backend; eval "$CONCEPT_FIGURE_ENV_COMMANDS"; '
        'printf "openai=%s\\naws=%s\\nvenv=%s\\n" "${OPENAI_API_KEY:-}" "${AWS_SECRET_ACCESS_KEY:-}" "${VENV_PY:-}"',
        _env(OPENAI_API_KEY="selected", AWS_SECRET_ACCESS_KEY="unselected", VENV_PY="/opt/selected-python"),
    )
    assert result.stdout.splitlines() == ["openai=selected", "aws=", "venv=/opt/selected-python"]

    # the backend reads a key from the native outer env file
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=selected\n", encoding="utf-8")
    result = _bash_capture(
        'source "$CONTAINER_HELPER"; source "$CONCEPT_HELPER"; concept_figure_log() { :; }; AGENT=codex; '
        'CONTAINER_ENV_FILE="$TEST_ENV_FILE"; PAPER_OPENAI_API_KEY_AVAILABLE=false; PAPER_GEMINI_API_KEY_AVAILABLE=false; '
        'select_concept_figure_backend; printf "%s\\n" "$CONCEPT_FIGURE_BACKEND_FLAGS"',
        _env(TEST_ENV_FILE=str(env_file)),
    )
    assert "--enhance-model gpt-image-2" in result.stdout


def test_concept_figure_sandbox_stages_selected_codex_profile(tmp_path: Path) -> None:
    auth_root = tmp_path / "auth"
    codex_home = auth_root / ".codex"
    codex_home.mkdir(parents=True)
    catalog = codex_home / "custom.json"
    catalog.write_text('{"models":[]}\n', encoding="utf-8")
    (codex_home / "auth.json").write_text("{}\n", encoding="utf-8")
    (codex_home / "config.toml").write_text(
        '[model_providers.custom]\nname = "Custom API"\nbase_url = "https://example.test/v1"\n'
        'env_key = "CUSTOM_API_KEY"\nwire_api = "responses"\nhttp_headers = { Authorization = "secret" }\n',
        encoding="utf-8",
    )
    (codex_home / "custom.config.toml").write_text(
        f'model_provider = "custom"\nmodel_catalog_json = "{catalog}"\n', encoding="utf-8"
    )

    result = _bash_capture(
        'source "$CONTAINER_HELPER"; source "$CONCEPT_HELPER"; docker() { :; }; '
        'AGENT=codex; CONTAINER_AUTH_ROOT="$AUTH_ROOT"; CONTAINER_ENV_FILE=; CONCEPT_FIGURE_SEED_AGENT_AUTH=1; '
        "CONCEPT_FIGURE_CONTAINER_HOME_DIR=; CONCEPT_FIGURE_SCOPED_ENV_FILE=; prepare_concept_figure_sandbox; "
        'printf "%s\\n" "$CONCEPT_FIGURE_CONTAINER_HOME_DIR"',
        _env(AGENT_PROFILE="custom", AUTH_ROOT=str(auth_root), TMPDIR=str(tmp_path)),
    )

    staged_codex_home = Path(result.stdout.strip()) / ".codex"
    assert (staged_codex_home / "custom.json").read_text(encoding="utf-8") == '{"models":[]}\n'
    assert (staged_codex_home / "custom.config.toml").read_text(encoding="utf-8") == (
        'model_provider = "custom"\nmodel_catalog_json = "/home/agent-home/.codex/custom.json"\n'
    )
    staged_config = (staged_codex_home / "config.toml").read_text(encoding="utf-8")
    assert 'env_key = "CUSTOM_API_KEY"' in staged_config
    assert "Authorization" not in staged_config
    for name in ("auth.json", "config.toml", "custom.config.toml", "custom.json"):
        assert stat.S_IMODE((staged_codex_home / name).stat().st_mode) == 0o600


def test_codex_profile_stager_rejects_escaping_catalogs(tmp_path: Path) -> None:
    stager = Path("src/scripts/lib/stage_codex_profile.py").resolve()
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    outside_catalog = tmp_path / "private.json"
    outside_catalog.write_text('{"secret":true}\n', encoding="utf-8")
    symlink_catalog = codex_home / "linked.json"
    symlink_catalog.symlink_to(outside_catalog)

    for configured_path in (outside_catalog, symlink_catalog):
        (codex_home / "unsafe.config.toml").write_text(f'model_catalog_json = "{configured_path}"\n', encoding="utf-8")
        destination = tmp_path / f"destination-{configured_path.name}"
        destination.mkdir()
        result = subprocess.run(
            [sys.executable, str(stager), str(codex_home), str(destination), "unsafe"], capture_output=True, text=True
        )
        assert result.returncode != 0, configured_path
        assert list(destination.iterdir()) == []


def test_container_helper_forwards_secrets_sanitizes_env_scrubs_shell_and_detects_blank_keys(tmp_path: Path) -> None:
    # secret keys are forwarded without argv values
    result = _bash_capture(
        'source "$CONTAINER_HELPER"; docker_cmd=(docker run); AGENT_CONTAINER_SECRET_PASSTHROUGH_KEYS=OPENAI_API_KEY; '
        "append_secret_passthrough_envs; printf '%s\\n' \"${docker_cmd[@]}\"",
        _env(OPENAI_API_KEY="secret-value"),
    )
    assert result.stdout.splitlines() == ["docker", "run", "-e", "OPENAI_API_KEY"]
    assert "secret-value" not in result.stdout

    # the env file sanitizer can retain only selected keys
    source = tmp_path / "source.env"
    destination = tmp_path / "destination.env"
    source.write_text("OPENAI_API_KEY=selected\nAWS_SECRET_ACCESS_KEY=unselected\nVENV_PY=/host/venv/python\n", encoding="utf-8")

    _bash(
        'source "$CONTAINER_HELPER"; sanitize_env_file_for_docker "$TEST_SOURCE" "$TEST_DESTINATION" OPENAI_API_KEY',
        _env(TEST_SOURCE=str(source), TEST_DESTINATION=str(destination)),
    )

    assert destination.read_text(encoding="utf-8") == "OPENAI_API_KEY=selected\n"

    # a native command scrubs the environment before starting the login shell
    hook = tmp_path / "bash-env.sh"
    marker = tmp_path / "bash-env-ran"
    output = tmp_path / "environment.txt"
    hook.write_text(f"printf ran > {shlex.quote(str(marker))}\n", encoding="utf-8")
    command = (
        'printf "openai=%s\\naws=%s\\n" "${OPENAI_API_KEY:-}" "${AWS_SECRET_ACCESS_KEY:-}" > '
        f"{shlex.quote(str(output))}"
    )

    _bash(
        'source "$CONTAINER_HELPER"; '
        'export BASH_ENV="$TEST_HOOK" AWS_SECRET_ACCESS_KEY=unselected OPENAI_API_KEY=selected; '
        "AGENT_ENV_ALLOWLIST_KEYS='HOME PATH OPENAI_API_KEY'; "
        'run_command_prompt codex "$TEST_COMMAND" "$TEST_ROOT" 0 unused none unused',
        _env(TEST_ROOT=str(tmp_path), TEST_HOOK=str(hook), TEST_COMMAND=command),
    )

    assert output.read_text(encoding="utf-8") == "openai=selected\naws=\n"
    assert not marker.exists()

    # env file key detection rejects blank values
    env_file = tmp_path / "keys.env"
    env_file.write_text(
        "OPENAI_API_KEY=openai-value\nOPENAI_API_KEY=\"\"\nGEMINI_API_KEY=''\nexport GEMINI_API_KEY = 'gemini-value'\n",
        encoding="utf-8",
    )
    result = _bash_capture(
        'source "$CONTAINER_HELPER"; '
        'env_file_has_nonempty_key "$TEST_ENV_FILE" OPENAI_API_KEY || printf "openai-blank\\n"; '
        'env_file_has_nonempty_key "$TEST_ENV_FILE" GEMINI_API_KEY && printf "gemini-set\\n"',
        _env(TEST_ENV_FILE=str(env_file)),
    )
    assert result.stdout.splitlines() == ["openai-blank", "gemini-set"]


def test_concept_figure_sandbox_stages_only_selected_auth_and_env_and_cleanup_is_non_gating(tmp_path: Path) -> None:
    # only the selected auth and env are staged
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    docker.chmod(0o755)
    auth_root = tmp_path / "auth"
    (auth_root / ".codex").mkdir(parents=True)
    (auth_root / ".codex" / "auth.json").write_text('{"selected": true}', encoding="utf-8")
    (auth_root / ".codex" / "config.toml").write_text('mcp_token = "unselected"', encoding="utf-8")
    (auth_root / ".claude").mkdir()
    (auth_root / ".claude" / "auth.json").write_text('{"unselected": true}', encoding="utf-8")
    source_env = tmp_path / "source.env"
    source_env.write_text("OPENAI_API_KEY=selected\nAWS_SECRET_ACCESS_KEY=unselected\n", encoding="utf-8")

    result = _bash_capture(
        'source "$CONTAINER_HELPER"; source "$CONCEPT_HELPER"; AGENT=codex; CONTAINER_AUTH_ROOT="$TEST_AUTH_ROOT"; '
        'CONTAINER_ENV_FILE="$TEST_ENV_FILE"; TMPDIR="$TEST_TMPDIR"; CONCEPT_FIGURE_SECRET_KEYS=OPENAI_API_KEY; '
        "CONCEPT_FIGURE_ALLOWED_ENV_KEYS=OPENAI_API_KEY; prepare_concept_figure_sandbox; "
        'printf "%s\\n%s\\n" "$CONCEPT_FIGURE_CONTAINER_HOME_DIR" "$CONCEPT_FIGURE_SCOPED_ENV_FILE"',
        _env(
            TEST_AUTH_ROOT=str(auth_root),
            TEST_ENV_FILE=str(source_env),
            TEST_TMPDIR=str(tmp_path),
            PATH=f"{fake_bin}:{os.environ['PATH']}",
        ),
    )
    home_path, env_path = map(Path, result.stdout.splitlines())

    assert (home_path / ".codex" / "auth.json").is_file()
    assert not (home_path / ".codex" / "config.toml").exists()
    assert not (home_path / ".claude").exists()
    assert env_path.read_text(encoding="utf-8") == "OPENAI_API_KEY=selected\n"

    # sandbox cleanup is non-gating
    home = tmp_path / "agentic-concept-home-test"
    locked = home / "locked"
    locked.mkdir(parents=True)
    locked.chmod(0)

    result = _bash_capture(
        'source "$CONCEPT_HELPER"; set -e; concept_figure_warn() { :; }; '
        'CONCEPT_FIGURE_CONTAINER_HOME_DIR="$TEST_HOME"; cleanup_concept_figure_container_home; printf survived',
        _env(TEST_HOME=str(home), TMPDIR=str(tmp_path)),
    )

    assert result.stdout == "survived"
    assert not home.exists()


# ---------------------------------------------------------------------------
# paper_figure_quality.sh
# ---------------------------------------------------------------------------


def _figure_quality(script: str, **env: str) -> subprocess.CompletedProcess[str]:
    return _bash_capture(
        'source "$HELPER"; ' + script,
        os.environ | {"HELPER": str(Path(FIGURE_QUALITY_SH).resolve())} | env,
        check=False,
    )


def test_figure_quality_snapshot_restores_workspace_rejects_tampering_and_writable_temp_root(tmp_path: Path) -> None:
    # the snapshot restores the pre-repair workspace
    output_dir = tmp_path / "paper"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True)
    artifact = output_dir / "paper.tex"
    artifact.write_text("original", encoding="utf-8")
    (figures_dir / "plot.png").write_text("plot", encoding="utf-8")

    result = _figure_quality(
        'paper_figure_quality_snapshot_workspace "$OUTPUT" SNAPSHOT DIGEST "$OUTPUT"; '
        'printf changed > "$OUTPUT/paper.tex"; touch "$OUTPUT/injected.sty"; rm -rf "$OUTPUT/figures"; '
        'printf attacker > "$OUTPUT/figures"; ln -s paper.tex "$OUTPUT/injected-link"; '
        'paper_figure_quality_restore_workspace "$SNAPSHOT" "$OUTPUT" "$DIGEST"',
        OUTPUT=str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    assert artifact.read_text(encoding="utf-8") == "original"
    assert not (output_dir / "injected.sty").exists()
    assert not (output_dir / "injected-link").exists()
    assert (output_dir / "figures").is_dir()
    assert (figures_dir / "plot.png").read_text(encoding="utf-8") == "plot"

    # a tampered snapshot is rejected
    output_dir = tmp_path / "tampered"
    output_dir.mkdir()
    artifact = output_dir / "paper.tex"
    artifact.write_text("current", encoding="utf-8")

    result = _figure_quality(
        'paper_figure_quality_snapshot_workspace "$OUTPUT" SNAPSHOT DIGEST "$OUTPUT"; '
        'printf tampered > "$SNAPSHOT/workspace/paper.tex"; '
        'if paper_figure_quality_restore_workspace "$SNAPSHOT" "$OUTPUT" "$DIGEST"; then exit 9; fi; '
        'paper_figure_quality_discard_snapshot "$SNAPSHOT"',
        OUTPUT=str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    assert "snapshot content changed unexpectedly" in result.stderr
    assert artifact.read_text(encoding="utf-8") == "current"

    # an agent-writable temp root is rejected
    writable_root = tmp_path / "results"
    writable_root.mkdir()
    output_dir = tmp_path / "custom-paper"
    output_dir.mkdir()
    (output_dir / "paper.tex").write_text("current", encoding="utf-8")

    result = _figure_quality(
        'paper_figure_quality_snapshot_workspace "$OUTPUT" SNAPSHOT DIGEST "$WRITABLE_ROOT"',
        OUTPUT=str(output_dir),
        TMPDIR=str(output_dir),
        WRITABLE_ROOT=str(writable_root),
    )

    assert result.returncode != 0
    assert "inside the agent-writable root" in result.stderr
    assert (output_dir / "paper.tex").read_text(encoding="utf-8") == "current"


def test_figure_quality_container_stop_requires_confirmed_absence() -> None:
    for name, inspect_error, expected_code in (
        ("confirmed_absent", "Error: No such object: paper-session", 0),
        ("inspect_failed", "permission denied while connecting to Docker", 1),
    ):
        result = _figure_quality(
            "stop_container_session_force() { return 0; }; "
            'docker() { printf "%s\\n" "$INSPECT_ERROR" >&2; return 1; }; '
            "paper_figure_quality_stop_agent_session 1 paper-session",
            INSPECT_ERROR=inspect_error,
        )
        assert result.returncode == expected_code, f"{name}: {result.stderr}"
