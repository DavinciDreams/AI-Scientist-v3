"""Shell repair/qualification loops: ordering, progress reporting, retained edit transactions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.task_drafting.test_smoke_repair_transitions import _function
from tests.task_drafting.test_task_repair_scope import _write_task


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "src/scripts/lib/task_qualification.sh"
TASK_SCRIPT = ROOT / "src/scripts/task.sh"


def _script_section(start: str, end: str) -> str:
    return TASK_SCRIPT.read_text().split(start, 1)[1].split(end, 1)[0]


def _rollback_body() -> str:
    return _script_section("rollback_pending_qualification_repair() {", "\n}\n")


def _acceptance_branch() -> str:
    return _script_section(
        "  unset TASK_ALIGNMENT_REPORT_ATTEMPT\n", "\ndone\n\npublication_started_at="
    )


def _shell(
    body: str, tmp_path: Path, env: dict[str, str] | None = None, timeout: float = 10
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", 'set -eu\nsource "$RECOVERY_HELPER"\n' + body],
        cwd=tmp_path,
        env=os.environ | {"RECOVERY_HELPER": str(HELPER), "PYTHON_BIN": sys.executable} | (env or {}),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------- repair_and_qualify_until_valid


def test_repair_loop_reaches_production_before_judge_and_never_returns_to_alignment(
    tmp_path: Path,
) -> None:
    # a repaired scaffold reaches production before returning to the judge
    for first_failure in ("production", "validation"):
        run = _shell(
            '''
qualification_repairs_used=0
QUALIFICATION_MAX_REPAIRS=4
QUALIFICATION_REPAIR_FAILURE_STAGE=""
repair_qualification_transactionally() {
  qualification_repairs_used=$((qualification_repairs_used + 1))
  printf 'repair %s using %s\n' "$qualification_repairs_used" "$1"
  if [ "$qualification_repairs_used" -eq 1 ] && [ "'''
            + first_failure
            + """" = validation ]; then
    QUALIFICATION_REPAIR_FAILURE_STAGE=validation
    return 3
  fi
  PENDING_QUALIFICATION_TRANSACTION=transaction
}
qualify_with_infrastructure_retry() {
  printf 'production %s\n' "$qualification_repairs_used"
  if [ "$qualification_repairs_used" -eq 1 ]; then
    FAILED_QUALIFICATION_REPORT=new_failure
    return 3
  fi
  ACCEPTED_QUALIFICATION_REPORT=success
}
rollback_pending_qualification_repair() { printf 'rollback\n'; }
repair_and_qualify_until_valid original_failure
printf 'judge with %s\n' "$ACCEPTED_QUALIFICATION_REPORT"
""",
            tmp_path,
        )
        assert run.returncode == 0, run.stderr
        assert run.stdout.endswith("production 2\njudge with success\n"), first_failure
        if first_failure == "production":
            assert "production 1\nrepair 2 using new_failure" in run.stdout
        else:
            assert "production 1" not in run.stdout
            assert "repair 2 using original_failure" in run.stdout
            assert "post-repair contract or smoke validation rejected" in run.stderr

    # a failed production search never returns to alignment
    for failure_code, expected_repairs in ((3, 2), (4, 1)):
        run = _shell(
            f"""
qualification_repairs_used=0
QUALIFICATION_MAX_REPAIRS=2
repair_qualification_transactionally() {{
  qualification_repairs_used=$((qualification_repairs_used + 1))
}}
qualify_with_infrastructure_retry() {{ FAILED_QUALIFICATION_REPORT=failure; return {failure_code}; }}
rollback_pending_qualification_repair() {{ printf 'rollback\\n'; }}
if repair_and_qualify_until_valid original; then
  printf 'unexpected judge\\n'
else
  printf 'exit=%s repairs=%s\\n' "$?" "$qualification_repairs_used"
fi
""",
            tmp_path,
        )
        assert run.returncode == 0, run.stderr
        assert "unexpected judge" not in run.stdout, failure_code
        assert f"exit={failure_code} repairs={expected_repairs}" in run.stdout
        assert run.stdout.count("rollback") == 1, failure_code


@pytest.mark.parametrize("repair_budget", [2, 4])
def test_real_acceptance_loop_requalifies_repairs_and_commits_once(
    tmp_path: Path, repair_budget: int
) -> None:
    (tmp_path / "alignment-diagnostic.json").write_text(
        json.dumps({"verdict": "fail", "failed": ["execution_feasibility"]})
    )
    run = _shell(
        "rollback_pending_qualification_repair() {"
        + _rollback_body()
        + """
}
restore_repair_transaction() { printf 'rollback %s\n' "$candidate"; candidate=original; }
commit_pending_alignment_repair() { :; }
qualify_with_infrastructure_retry() {
  printf 'qualify %s\n' "$candidate"
  if [ "$candidate" = original ] || [ "$candidate" = repair1 ]; then
    FAILED_QUALIFICATION_REPORT="failure-$candidate"
    return 3
  fi
  ACCEPTED_QUALIFICATION_REPORT="success-$candidate"
  ACCEPTED_QUALIFICATION_RECEIPT="receipt-$candidate"
}
repair_qualification_transactionally() {
  qualification_repairs_used=$((qualification_repairs_used + 1))
  candidate="repair$qualification_repairs_used"
  printf 'repair %s\n' "$candidate"
  PENDING_QUALIFICATION_TRANSACTION="$candidate"
  QUALIFICATION_REPAIR_SOURCE_REPORT="$1"
}
validate_accepted_qualification() {
  [ "$ACCEPTED_QUALIFICATION_REPORT" = "success-$candidate" ]
  [ "$ACCEPTED_QUALIFICATION_RECEIPT" = "receipt-$candidate" ]
}
commit_pending_qualification_repair() {
  validate_accepted_qualification
  [ "$ACCEPTED_ALIGNMENT_REPORT" = "alignment-$candidate" ]
  printf 'commit %s\n' "$candidate"
}
fake_python() { case "$*" in *tree-fingerprint*) printf 'sha256:%064d' 0 ;; esac; }
PYTHON_BIN=fake_python
CONTAINERIZE=1
STAGED_TASK_DIR=staged
QUALIFICATION_REPAIR_HISTORY=history
ALIGNMENT_DIAGNOSTIC=alignment-diagnostic.json
QUALIFICATION_MAX_REPAIRS="""
        + str(repair_budget)
        + """
qualification_repairs_used=0
review_round=0
candidate=original
PENDING_QUALIFICATION_TRANSACTION=""
ACCEPTED_QUALIFICATION_REPORT=""
ACCEPTED_QUALIFICATION_RECEIPT=""
while true; do
  REPORT_PATH="alignment-$candidate"
  printf 'judge %s\n' "$candidate"
  if [ "$candidate" = repair2 ]; then alignment_exit=3; else alignment_exit=0; fi
"""
        + _acceptance_branch()
        + """
done
""",
        tmp_path,
    )
    assert "production search passed; independent alignment rejected" in run.stderr
    assert '"failed": ["execution_feasibility"]' in run.stderr
    if repair_budget == 2:
        assert run.stdout.count("rollback repair2") == 1
        assert run.returncode == 3, run.stdout + run.stderr
        assert "2/2 (0 remaining)" in run.stderr
        assert "last failed gate: alignment" in run.stderr
        assert "commit " not in run.stdout
        assert "repair repair3" not in run.stdout
        return
    assert run.returncode == 0, run.stdout + run.stderr
    actions = [
        line
        for line in run.stdout.splitlines()
        if line.split()[0] in {"judge", "qualify", "repair", "rollback", "commit"}
    ]
    assert actions == [
        "judge original",
        "qualify original",
        "repair repair1",
        "qualify repair1",
        "repair repair2",
        "qualify repair2",
        "judge repair2",
        "repair repair3",
        "qualify repair3",
        "judge repair3",
        "commit repair3",
    ]


# ---------------------------------------------------------------- progress reporting


def test_exhaustion_names_last_failed_gate_and_production_success_is_pending_alignment(
    tmp_path: Path,
) -> None:
    # exhaustion names the last failed gate
    for stage in ("production-search", "post-repair-validation", "alignment"):
        run = _shell(
            """
qualification_repairs_used=4
QUALIFICATION_MAX_REPAIRS=4
QUALIFICATION_LAST_FAILURE_STAGE="$STAGE"
rollback_pending_qualification_repair() { :; }
if repair_and_qualify_until_valid report; then exit 99; else exit "$?"; fi
""",
            tmp_path,
            env={"STAGE": stage},
            timeout=5,
        )
        assert run.returncode == 3, stage
        assert f"last failed gate: {stage}" in run.stderr
        assert "4/4" in run.stderr, stage
        assert "0 remaining" in run.stderr, stage

    # production success is reported as pending alignment
    run = _shell(
        """
qualification_repairs_used=0
QUALIFICATION_MAX_REPAIRS=4
repair_qualification_transactionally() { qualification_repairs_used=1; }
qualify_with_infrastructure_retry() { return 0; }
repair_and_qualify_until_valid report
""",
        tmp_path,
        timeout=5,
    )
    assert run.returncode == 0, run.stderr
    assert "production search passed; independent alignment pending" in run.stderr
    assert "1/4" in run.stderr
    assert "3 remaining" in run.stderr


def test_ordinary_failures_report_retained_edits_and_rollback_clears_provisional_acceptance(
    tmp_path: Path,
) -> None:
    # ordinary failures report retained edits and remaining budget
    for stage in ("validation_failed", "production_failed", "alignment_rejected"):
        run = _shell(
            """
qualification_repairs_used=2
QUALIFICATION_MAX_REPAIRS=4
qualification_repair_checkpoint "$STAGE"
""",
            tmp_path,
            env={"STAGE": stage},
            timeout=5,
        )
        assert run.returncode == 0, run.stderr
        assert "edits retained" in run.stderr, stage
        assert "2/4" in run.stderr, stage
        assert "2 remaining" in run.stderr, stage

    # rollback clears provisional acceptance
    run = _shell(
        "rollback_pending_qualification_repair() {"
        + _rollback_body()
        + """
}
restore_repair_transaction() { :; }
PENDING_QUALIFICATION_TRANSACTION=transaction
ACCEPTED_QUALIFICATION_REPORT=report
ACCEPTED_QUALIFICATION_RECEIPT=receipt
ACCEPTED_ALIGNMENT_REPORT=alignment
rollback_pending_qualification_repair
printf '%s|%s|%s\n' "$ACCEPTED_QUALIFICATION_REPORT" "$ACCEPTED_QUALIFICATION_RECEIPT" "$ACCEPTED_ALIGNMENT_REPORT"
""",
        tmp_path,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "||"


# ---------------------------------------------------------------- retained qualification repairs


def _run_chain(tmp_path: Path, outcome: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    tmp_path.mkdir(exist_ok=True)
    task = _write_task(tmp_path)
    validation = tmp_path / "validation"
    recovery = tmp_path / "recovery"
    validation.mkdir()
    recovery.mkdir(mode=0o700)
    (validation / "alignment.json").write_text('{"failed": ["execution_feasibility"]}')
    body = r'''
qualification_repairs_used=0
repairs_used=0
QUALIFICATION_MAX_REPAIRS=4
case "$TEST_OUTCOME" in exhausted|restore_failure) QUALIFICATION_MAX_REPAIRS=3 ;; esac
PENDING_QUALIFICATION_TRANSACTION=""
ACTIVE_REPAIR_TRANSACTION=""
ACCEPTED_ALIGNMENT_REPORT=""
ACCEPTED_QUALIFICATION_REPORT=""
ACCEPTED_QUALIFICATION_RECEIPT=""
QUALIFICATION_CALIBRATION_CONTEXT_PATH=""
QUALIFICATION_REPAIR_FAILURE_STAGE=""
LAST_CONTRACT_DIAGNOSTIC=""
LAST_SMOKE_DIAGNOSTIC='{"category":"missing_dependency"}'
QUALIFICATION_REPAIR_HISTORY="$VALIDATION_ROOT/history.json"
ALIGNMENT_DIAGNOSTIC="$VALIDATION_ROOT/alignment.json"
CONTAINERIZE=1
review_round=0

python_dispatch() {
  case "$2" in
    src.task_drafting.qualification_repair|src.task_drafting.qualification_calibration)
      printf '%s %s\n' "$2" "$3" >> "$VALIDATION_ROOT/operations"
      if [ "$TEST_OUTCOME" = calibration_failure ] \
        && [ "$2" = src.task_drafting.qualification_calibration ]; then return 4; fi
      return 0
      ;;
    src.investigation.task_repair_transaction)
      printf 'transaction %s\n' "$3" >> "$VALIDATION_ROOT/operations"
      if [ "$TEST_OUTCOME" = restore_failure ] && [ "$3" = restore ]; then return 4; fi
      ;;
  esac
  "$REAL_PYTHON" "$@"
}
PYTHON_BIN=python_dispatch
if [ "$TEST_OUTCOME" != real_history ]; then
  retain_rejected_qualification_context() { :; }
fi
discard_staged_runtime_for_repair() { :; }
render_qualification_repair_prompt() { printf 'repair\n'; }
validate_staged_paths() {
  if [ "$TEST_OUTCOME" = unsafe_paths ] && [ "$qualification_repairs_used" -eq 2 ]; then
    printf 'Unsafe canonical task.md\n'
    return 1
  fi
}
canonicalize_staged_scaffold() { :; }
repair_contract_until_valid() { :; }
sanitize_preflight_diagnostic() { cat "$1"; }
write_agent_prompt_file() { printf '%s\n' "$2" > "$1"; }
task_progress_emit() { :; }
commit_pending_alignment_repair() { :; }
run_scoped_repair() {
  local next="$((qualification_repairs_used + 1))"
  printf 'agent %s\n' "$next"
  printf '%s\n' "$ACTIVE_REPAIR_TRANSACTION" >> "$VALIDATION_ROOT/transactions"
  cp "$STAGED_TASK_DIR/evaluate.py" "$VALIDATION_ROOT/before-agent-$next.py"
  printf '%s|%s|%s\n' "$ACCEPTED_ALIGNMENT_REPORT" \
    "$ACCEPTED_QUALIFICATION_REPORT" "$ACCEPTED_QUALIFICATION_RECEIPT" \
    >> "$VALIDATION_ROOT/acceptance-before-agent"
  printf '# useful edit %s\n' "$next" >> "$STAGED_TASK_DIR/evaluate.py"
  if [ "$TEST_OUTCOME" = agent_failure ] && [ "$next" -eq 2 ]; then return 9; fi
  if [ "$TEST_OUTCOME" = interrupt ] && [ "$next" -eq 2 ]; then return 130; fi
}
smoke_until_valid() {
  printf 'smoke %s\n' "$qualification_repairs_used"
  if [ "$qualification_repairs_used" -eq 1 ]; then return 3; fi
  if [ "$TEST_OUTCOME" = smoke_infrastructure ]; then return 4; fi
}
qualify_with_infrastructure_retry() {
  printf 'production %s\n' "$qualification_repairs_used"
  FAILED_QUALIFICATION_REPORT="$VALIDATION_ROOT/failure-$qualification_repairs_used.json"
  printf '{}\n' > "$FAILED_QUALIFICATION_REPORT"
  if [ "$qualification_repairs_used" -eq 0 ] || [ "$qualification_repairs_used" -eq 2 ]; then
    if [ "$TEST_OUTCOME" = infrastructure ] && [ "$qualification_repairs_used" -eq 2 ]; then
      return 4
    fi
    return 3
  fi
  ACCEPTED_QUALIFICATION_REPORT="report-$qualification_repairs_used"
  ACCEPTED_QUALIFICATION_RECEIPT="receipt-$qualification_repairs_used"
}
validate_accepted_qualification() {
  [ "$ACCEPTED_QUALIFICATION_REPORT" = "report-$qualification_repairs_used" ]
  [ "$ACCEPTED_QUALIFICATION_RECEIPT" = "receipt-$qualification_repairs_used" ]
}
'''
    for name in (
        "restore_repair_transaction", "rollback_pending_qualification_repair",
        "commit_pending_qualification_repair", "repair_qualification_transactionally",
    ):
        body += _function(name)
    body += r'''
while true; do
  REPORT_PATH="alignment-$qualification_repairs_used"
  printf 'judge %s\n' "$qualification_repairs_used"
  if [ "$qualification_repairs_used" -eq 3 ]; then alignment_exit=3; else alignment_exit=0; fi
'''
    body += _acceptance_branch() + "\ndone\n"
    run = subprocess.run(
        ["bash", "-c", 'set -eu\nsource "$RECOVERY_HELPER"\n' + body], cwd=ROOT,
        env=os.environ | {
            "PYTHONPATH": str(ROOT), "REAL_PYTHON": sys.executable,
            "RECOVERY_HELPER": str(HELPER),
            "STAGED_TASK_DIR": str(task), "TASK_DIR": str(task),
            "VALIDATION_ROOT": str(validation), "HARNESS_RECOVERY_ROOT": str(recovery),
            "OUTPUT_DIR": str(tmp_path), "TEST_OUTCOME": outcome,
        },
        capture_output=True, text=True, timeout=60,
    )
    return run, task


def test_retained_chain_keeps_incremental_edits_and_private_state_on_failed_restore(
    tmp_path: Path,
) -> None:
    # smoke, production and alignment failures retain incremental edits
    for outcome in ("success", "real_history"):
        root = tmp_path / outcome
        run, task = _run_chain(root, outcome)
        assert run.returncode == 0, run.stdout + run.stderr
        validation = root / "validation"
        for attempt in range(2, 5):
            previous = (validation / f"before-agent-{attempt}.py").read_text()
            assert previous == "# search\n" + "".join(
                f"# useful edit {index}\n" for index in range(1, attempt)
            ), (outcome, attempt)
        assert "# useful edit 4" in (task / "evaluate.py").read_text(), outcome
        assert len(set((validation / "transactions").read_text().splitlines())) == 1, outcome
        operations = (validation / "operations").read_text().splitlines()
        assert operations.count("transaction create-authored-tree") == 1, outcome
        assert operations.count("transaction commit") == 1, outcome
        assert "transaction restore" not in operations, outcome
        assert (validation / "acceptance-before-agent").read_text().splitlines() == ["||"] * 4
        actions = [
            line for line in run.stdout.splitlines()
            if line.startswith(("agent ", "smoke ", "production ", "judge "))
        ]
        assert actions == [
            "judge 0", "production 0", "agent 1", "smoke 1", "agent 2", "smoke 2",
            "production 2", "agent 3", "smoke 3", "production 3", "judge 3",
            "agent 4", "smoke 4", "production 4", "judge 4",
        ], outcome
        if outcome == "real_history":
            histories = sorted(validation.glob("qualification_*_transaction.json"))
            assert len(histories) == 3
            records = [json.loads(path.read_text()) for path in histories]
            assert len({record["base_fingerprint"] for record in records}) == 1
            assert len({record["rejected_fingerprint"] for record in records}) == 3
            for attempt, record in enumerate(records, start=1):
                assert f"# useful edit {attempt}" in json.dumps(record)

    # a failed restoration retains the private recovery state
    root = tmp_path / "restore_failure"
    run, task = _run_chain(root, "restore_failure")
    assert run.returncode == 4, run.stdout + run.stderr
    assert "# useful edit 3" in (task / "evaluate.py").read_text()
    transactions = (root / "validation" / "transactions").read_text().splitlines()
    assert len(set(transactions)) == 1
    assert (Path(transactions[0]) / "manifest.json").is_file()
    operations = (root / "validation" / "operations").read_text().splitlines()
    assert operations.count("transaction restore") == 1
    assert "transaction commit" not in operations


@pytest.mark.parametrize("outcome, expected_exit", [
    ("exhausted", 3), ("infrastructure", 4), ("agent_failure", 4), ("interrupt", 4),
    ("unsafe_paths", 4), ("smoke_infrastructure", 4), ("calibration_failure", 4),
])
def test_terminal_failure_restores_original_before_the_whole_chain(
    tmp_path: Path, outcome: str, expected_exit: int
) -> None:
    run, task = _run_chain(tmp_path, outcome)
    assert run.returncode == expected_exit, run.stdout + run.stderr
    assert (task / "evaluate.py").read_text() == "# search\n"
    validation = tmp_path / "validation"
    assert "# useful edit 1" in (validation / "before-agent-2.py").read_text()
    operations = (validation / "operations").read_text().splitlines()
    assert operations.count("transaction create-authored-tree") == 1
    assert operations.count("transaction restore") == 1
    assert operations.count("transaction commit") == 1


# ---------------------------------------------------------------- retained alignment repairs


def _run_alignment(tmp_path: Path, body: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    tmp_path.mkdir(exist_ok=True)
    task = _write_task(tmp_path)
    for name in ("validation", "recovery", "round"):
        (tmp_path / name).mkdir(mode=0o700)
    functions = [
        "repair_alignment_transactionally", "render_alignment_validation_retry_prompt",
        "restore_repair_transaction",
    ]
    script = TASK_SCRIPT.read_text()
    for name in ("rollback_pending_alignment_repair", "commit_pending_alignment_repair"):
        if name + "() {" in script:
            functions.append(name)
    run = subprocess.run(
        ["bash", "-c", "set -eu\n" + "".join(map(_function, functions)) + '''
source "$REPO_ROOT/src/scripts/lib/task_qualification.sh"
alignment_repairs_used=0
repairs_used=0
ALIGNMENT_MAX_REPAIRS=3
ACTIVE_REPAIR_TRANSACTION=""
ACTIVE_ALIGNMENT_TRANSACTION=""
discard_staged_runtime_for_repair() { :; }
task_progress_emit() { printf '%s\n' "$*" >> "$TRACE"; }
repair_budget_used() { printf '%s' "$1"; }
repair_budget_exhausted() { [ "$1" -ge "$2" ]; }
repair_alignment_until_structurally_valid() {
  alignment_repairs_used=$((alignment_repairs_used + 1))
  printf '%s\n' "$1" >> "$PROMPTS"
  cat "$STAGED_TASK_DIR/evaluate.py" >> "$SEEN"
  printf '# repair %s\n' "$alignment_repairs_used" >> "$STAGED_TASK_DIR/evaluate.py"
}
repair_contract_until_valid() { return 0; }
''' + body],
        cwd=ROOT,
        env=os.environ | {
            "REPO_ROOT": str(ROOT), "PYTHONPATH": str(ROOT), "PYTHON_BIN": sys.executable,
            "STAGED_TASK_DIR": str(task), "VALIDATION_ROOT": str(tmp_path / "validation"),
            "HARNESS_RECOVERY_ROOT": str(tmp_path / "recovery"),
            "ROUND_DIR": str(tmp_path / "round"), "TRACE": str(tmp_path / "trace"),
            "PROMPTS": str(tmp_path / "prompts"), "SEEN": str(tmp_path / "seen"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True, text=True, timeout=30,
    )
    return run, task


def test_alignment_repairs_retain_edits_across_smoke_judge_and_contract_retries(
    tmp_path: Path,
) -> None:
    # a smoke retry keeps the previous edit and the original snapshot
    root = tmp_path / "smoke"
    run, task = _run_alignment(root, '''
smoke_until_valid() {
  if [ "$alignment_repairs_used" -eq 1 ]; then
    LAST_SMOKE_DIAGNOSTIC='{"category":"runtime_baseline_validation","diagnostic":"missing adapter"}'
    return 3
  fi
}
repair_alignment_transactionally findings "$ROUND_DIR/prompt"
printf 'pending=%s\n' "$ACTIVE_ALIGNMENT_TRANSACTION"
''')
    assert run.returncode == 0, run.stderr
    assert (task / "evaluate.py").read_text() == "# search\n# repair 1\n# repair 2\n"
    assert "# repair 1" in (root / "seen").read_text()
    snapshot = root / "recovery/alignment_0_transaction/files/evaluate.py"
    assert snapshot.read_text() == "# search\n"
    assert "pending=" + str(snapshot.parents[1]) in run.stdout
    assert "edits retained" in (root / "trace").read_text()
    assert "rolled back" not in (root / "prompts").read_text()

    # a judge rejection keeps the chain until alignment accepts
    run, task = _run_alignment(tmp_path / "judge", '''
smoke_until_valid() { :; }
repair_alignment_transactionally first-findings "$ROUND_DIR/first"
first_transaction="$ACTIVE_ALIGNMENT_TRANSACTION"
repair_alignment_transactionally next-findings "$ROUND_DIR/next"
[ "$first_transaction" = "$ACTIVE_ALIGNMENT_TRANSACTION" ]
commit_pending_alignment_repair
[ -z "$ACTIVE_REPAIR_TRANSACTION" ]
[ -z "$ACTIVE_ALIGNMENT_TRANSACTION" ]
''')
    assert run.returncode == 0, run.stderr
    assert (task / "evaluate.py").read_text() == "# search\n# repair 1\n# repair 2\n"

    # a contract retry clears acceptance without discarding edits
    root = tmp_path / "contract"
    run, task = _run_alignment(root, '''
repair_contract_until_valid() {
  [ -z "$ACCEPTED_ALIGNMENT_REPORT" ] || return 99
  [ -z "$ACCEPTED_QUALIFICATION_REPORT" ] || return 99
  [ -z "$ACCEPTED_QUALIFICATION_RECEIPT" ] || return 99
  if [ "$alignment_repairs_used" -eq 1 ]; then
    LAST_CONTRACT_DIAGNOSTIC='Invalid contract: missing resource digest'
    return 3
  fi
}
smoke_until_valid() { :; }
ACCEPTED_ALIGNMENT_REPORT=stale-alignment
ACCEPTED_QUALIFICATION_REPORT=stale-production
ACCEPTED_QUALIFICATION_RECEIPT=stale-receipt
repair_alignment_transactionally findings "$ROUND_DIR/prompt"
''')
    assert run.returncode == 0, run.stderr
    assert (task / "evaluate.py").read_text() == "# search\n# repair 1\n# repair 2\n"
    assert "edits retained" in (root / "trace").read_text()


@pytest.mark.parametrize("failure", [3, 4, 75, 130])
def test_terminal_failure_restores_original_chain(tmp_path: Path, failure: int) -> None:
    run, task = _run_alignment(tmp_path, f'''
ALIGNMENT_MAX_REPAIRS=2
smoke_until_valid() {{
  if [ "$alignment_repairs_used" -eq 1 ]; then return 3; fi
  return {failure}
}}
if repair_alignment_transactionally findings "$ROUND_DIR/prompt"; then exit 99; else code=$?; fi
printf 'exit=%s active=%s\n' "$code" "$ACTIVE_REPAIR_TRANSACTION"
''')
    assert run.returncode == 0, run.stderr
    assert run.stdout == f"exit={failure} active=\n"
    assert (task / "evaluate.py").read_text() == "# search\n"


def test_failed_restore_retains_recovery_handle_and_agent_failure_is_terminal(
    tmp_path: Path,
) -> None:
    # a failed restore retains the recovery handle
    root = tmp_path / "restore"
    run, _ = _run_alignment(root, '''
smoke_until_valid() { return 4; }
discard_staged_runtime_for_repair() { return 4; }
if repair_alignment_transactionally findings "$ROUND_DIR/prompt"; then exit 99; else code=$?; fi
printf 'exit=%s pending=%s\n' "$code" "$ACTIVE_REPAIR_TRANSACTION"
''')
    assert run.returncode == 0, run.stderr
    assert run.stdout == f"exit=4 pending={root}/recovery/alignment_0_transaction\n"

    # an agent failure is terminal even with an authored exit code
    for failure in (1, 3, 130):
        run, task = _run_alignment(
            tmp_path / f"agent-{failure}",
            _function("repair_alignment_until_structurally_valid") + f'''
review_round=0
run_scoped_repair() {{
  printf '# incomplete agent edit\\n' >> "$STAGED_TASK_DIR/evaluate.py"
  return {failure}
}}
if repair_alignment_transactionally findings "$ROUND_DIR/prompt"; then exit 99; else code=$?; fi
printf 'exit=%s\\n' "$code"
''',
        )
        assert run.returncode == 0, run.stderr
        assert run.stdout == f"exit={failure}\n"
        assert (task / "evaluate.py").read_text() == "# search\n", failure


def test_changed_canonical_task_and_outer_judge_exit_settle_retained_alignment_tree(
    tmp_path: Path,
) -> None:
    # a changed canonical task cannot reach another repair
    root = tmp_path / "canonical-change"
    root.mkdir()
    canonical = root / "canonical"
    canonical.mkdir()
    (canonical / "task.md").write_text("# task\n")
    run, task = _run_alignment(
        root,
        _function("validate_staged_paths")
        + _function("repair_alignment_until_structurally_valid")
        + f'TASK_SOURCE_DIR="{canonical}"\n'
        + '''
review_round=0
run_scoped_repair() {
  printf '# rewritten task\n' > "$STAGED_TASK_DIR/task.md"
  printf 'agent\n' >> "$TRACE"
}
if repair_alignment_transactionally findings "$ROUND_DIR/prompt"; then exit 99; else code=$?; fi
printf 'exit=%s\n' "$code"
''',
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout == "exit=1\n"
    assert (root / "trace").read_text().count("agent\n") == 1
    assert (task / "task.md").read_text() == "# task\n"

    # an outer judge exit or signal settles the retained alignment tree
    for index, stop in enumerate(("exit 3", "exit 4", "kill -TERM $$")):
        run, task = _run_alignment(
            tmp_path / f"stop-{index}",
            _function("cleanup_run_processes") + _function("on_interrupt") + '''
CONTAINERIZE=0
LOG_PIPE=""
HARNESS_VALIDATION_ROOT="$VALIDATION_ROOT"
TASK_TEMP_ROOT="$VALIDATION_ROOT"
stop_task_log_writer() { :; }
task_progress_cleanup() { :; }
smoke_until_valid() { :; }
trap on_interrupt INT TERM
trap cleanup_run_processes EXIT
repair_alignment_transactionally first-findings "$ROUND_DIR/first"
repair_alignment_transactionally next-findings "$ROUND_DIR/next"
''' + stop,
        )
        assert run.returncode == (130 if "TERM" in stop else int(stop.split()[1])), run.stderr
        assert (task / "evaluate.py").read_text() == "# search\n", stop
