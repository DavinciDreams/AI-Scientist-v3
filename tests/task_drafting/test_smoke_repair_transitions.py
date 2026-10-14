"""Production shell repair loops: smoke transitions, gate failure flow, gate arguments, env helpers."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tests.task_drafting.test_task_repair_scope import _write_task

ROOT = Path(__file__).resolve().parents[2]
CONTAINER_AGENT_LIB = ROOT / "src/scripts/lib/container_agent.sh"


def _function(name: str) -> str:
    script = (ROOT / "src/scripts/task.sh").read_text()
    return name + "() {" + script.split(name + "() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"


# --- smoke loop with real scope and rollback operations -------------------------------------


def _run_loop(tmp_path: Path, outcome: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    tmp_path.mkdir(exist_ok=True)
    task = _write_task(tmp_path)
    if "removal" in outcome:
        from src.task_drafting.contract_sync import synchronize_declared_digests

        (task / "requirements.txt").write_text("fixture==1.0.0\nunused==2.0.0\n")
        contract_path = task / "task_contract.json"
        contract = json.loads(contract_path.read_text())
        contract["provenance"]["dependencies"].append({"name": "unused", "version": "2.0.0"})
        contract_path.write_text(json.dumps(contract))
        synchronize_declared_digests(contract_path, canonical_task=task / "task.md")
    validation = tmp_path / "validation"
    recovery = tmp_path / "recovery"
    validation.mkdir()
    recovery.mkdir(mode=0o700)
    driver = tmp_path / "smoke.py"
    driver.write_text('''import json, os
from pathlib import Path
task = Path(os.environ["STAGED_TASK_DIR"])
repaired = "repaired" in (task / "evaluate.py").read_text()
outcome = os.environ["TEST_OUTCOME"]
success = repaired and outcome in {"success", "removal_success"}
if outcome == "removal_followup":
    success = "repaired 1" in (task / "evaluate.py").read_text()
infrastructure = repaired and outcome == "infrastructure"
code = 0 if success else 4 if infrastructure else 3
print(json.dumps({"status": "pass" if success else "infrastructure_failure" if infrastructure else "authored_failure",
 "category": "passed" if success else "infrastructure" if infrastructure else "search_nonzero_exit",
 "gate": "acceptance-smoke", "exit_code": code}))
raise SystemExit(code)
''')
    authorization_driver = tmp_path / "authorize.py"
    authorization_driver.write_text('''import sys
from pathlib import Path
from src.task_drafting.gate import TaskGateReport
from src.investigation.task_alignment import task_tree_fingerprint
from src.investigation.task_repair_authorization import DependencyRepairExecutor
task, snapshot, transaction, authorization = map(Path, sys.argv[1:5])
session = sys.argv[5]
def dependency_failure(operation):
    return TaskGateReport.authored_failure(gate=operation, component="runtime", stage="bootstrap",
        category="runtime_dependencies", scaffold_fingerprint=task_tree_fingerprint(task),
        exit_code=3, elapsed_seconds=0.1, diagnostic="missing transitive dependency")
gate = DependencyRepairExecutor(task, snapshot, transaction, authorization, session, dependency_failure)
assert not authorization.exists()
report = gate("acceptance-smoke")
assert report.category == "runtime_dependencies", report
assert "Harness authorized dependency removal" in report.diagnostic
assert authorization.is_file(), "agent received permission before receipt existed"
''')
    body = r'''
set -eu
TASK_DRAFTING_WORKFLOW=default
SMOKE_MAX_REPAIRS=1
if [ "$TEST_OUTCOME" = removal_followup ]; then SMOKE_MAX_REPAIRS=2; fi
SMOKE_MAX_INFRASTRUCTURE_ATTEMPTS=1
QUALIFICATION_TIMEOUT_SECONDS=300
EVALUATOR_SMOKE_TIMEOUT_SECONDS=10
PRODUCTION_ENTRY_TIMEOUT_SECONDS=10
smoke_repairs_used=0
repairs_used=0
smoke_attempt=0
ACTIVE_REPAIR_TRANSACTION=""
BYTECODE_QUARANTINE_ROOT="$VALIDATION_ROOT/bytecode"
ACCEPTED_BYTECODE_CACHE="$("$PYTHON_BIN" -c 'import sys,importlib.util; print(sys.implementation.cache_tag + ":" + importlib.util.MAGIC_NUMBER.hex())')"
repair_budget_used() { printf '%s\n' "$1"; }
repair_budget_exhausted() { [ "$1" -ge "$2" ]; }
task_progress_emit() { :; }
smoke_required_budget_for_operation() { printf '10\n'; }
smoke_staged_evaluators() { "$PYTHON_BIN" "$TEST_DRIVER"; }
typed_smoke_diagnostic() { cat "$1"; }
write_agent_prompt_file() { printf '%s\n' "$2" > "$1"; }
smoke_report_field() { "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
smoke_infrastructure_retryable() { return 1; }
smoke_repair_contract_policy() { printf 'synchronized-digests\n'; }
render_smoke_repair_prompt() { printf 'repair transaction %s\n' "$2"; }
discard_staged_runtime_for_repair() { :; }
validate_staged_paths() { :; }
canonicalize_staged_scaffold() { :; }
repair_contract_until_valid() { :; }
smoke_target_operations() { printf 'acceptance-smoke\n'; }
run_scoped_repair() {
  if [ -z "$ACTIVE_REPAIR_TRANSACTION" ] || [ ! -d "$ACTIVE_REPAIR_TRANSACTION" ]; then
    echo 'rollback missing before agent' >&2
    return 7
  fi
  if [ "$TEST_OUTCOME" = removal_followup ]; then
    printf '# repaired %s\n' "$smoke_repairs_used" > "$STAGED_TASK_DIR/evaluate.py"
    if [ "$smoke_repairs_used" -eq 1 ]; then
      [ "$2" = 'repair transaction active' ] || return 8
      return 0
    fi
  else
    printf '# repaired\n' > "$STAGED_TASK_DIR/evaluate.py"
  fi
  case "$TEST_OUTCOME" in
    removal_*)
      "$PYTHON_BIN" "$AUTH_DRIVER" "$STAGED_TASK_DIR" "$TASK_REPAIR_SNAPSHOT" \
        "$TASK_REPAIR_TRANSACTION" "$TASK_REPAIR_AUTHORIZATION" "$TASK_REPAIR_SESSION_ID" || return $?
      printf 'fixture==1.0.0\n' > "$STAGED_TASK_DIR/requirements.txt"
      ;;
    unauthorized_removal)
      printf 'fixture==1.0.0\n' > "$STAGED_TASK_DIR/requirements.txt"
      ;;
  esac
  if [ "$TEST_OUTCOME" = agent_failure ]; then return 9; fi
  if [ "$TEST_OUTCOME" = interrupt ]; then return 130; fi
}
'''
    body += _function("restore_repair_transaction") + _function("smoke_until_valid")
    body += '\nsmoke_until_valid\n'
    env = os.environ | {
        "PYTHON_BIN": sys.executable, "PYTHONPATH": str(ROOT),
        "STAGED_TASK_DIR": str(task), "VALIDATION_ROOT": str(validation),
        "HARNESS_RECOVERY_ROOT": str(recovery), "TEST_DRIVER": str(driver),
        "TEST_OUTCOME": outcome, "AUTH_DRIVER": str(authorization_driver),
    }
    run = subprocess.run(["bash", "-c", body], cwd=ROOT, env=env, text=True,
                         capture_output=True, timeout=20)
    return run, task


def test_successful_smoke_repairs_commit_authorize_removal_and_record_progress(
    tmp_path, monkeypatch
):
    # test_evaluator_only_smoke_repair_has_transaction_before_agent_and_commits
    root = tmp_path / "evaluator-only"
    run, task = _run_loop(root, "success")
    assert run.returncode == 0, run.stderr
    assert (task / "evaluate.py").read_text() == "# repaired\n"
    assert not list((root / "recovery").glob("*_repair_transaction"))
    snapshot = json.loads((root / "validation/smoke_1_repair_scope.json").read_text())
    assert snapshot["contract_policy"] == "synchronized-digests"

    # test_new_dependency_failure_authorizes_removal_and_syncs_contract
    run, task = _run_loop(tmp_path / "removal", "removal_success")
    assert run.returncode == 0, run.stderr
    assert (task / "requirements.txt").read_text() == "fixture==1.0.0\n"
    contract = json.loads((task / "task_contract.json").read_text())
    assert contract["provenance"]["dependencies"] == [{"name": "fixture", "version": "1.0.0"}]
    assert (task / "evaluate.py").read_text() == "# repaired\n"

    # test_authorization_progress_records_session_and_trigger
    from src.task_drafting import progress

    events = []
    monkeypatch.setattr(progress, "emit_event", lambda **fields: events.append(fields))
    assert progress.main([
        "emit", "--gate", "smoke", "--state", "completed", "--label", "Dependency repair authorized",
        "--repair-session-id", "repair-session", "--trigger-category", "runtime_dependencies",
    ]) == 0
    assert events[0]["fields"]["repair_session_id"] == "repair-session"
    assert events[0]["fields"]["trigger_category"] == "runtime_dependencies"


@pytest.mark.parametrize("outcome, exit_code, requirements, dependencies", [
    ("agent_failure", 9, "fixture==1.0.0\n", 1), ("interrupt", 130, "fixture==1.0.0\n", 1),
    ("exhaustion", 3, "fixture==1.0.0\n", 1), ("infrastructure", 4, "fixture==1.0.0\n", 1),
    ("removal_exhaustion", None, "fixture==1.0.0\nunused==2.0.0\n", 2),
    ("unauthorized_removal", None, "fixture==1.0.0\nunused==2.0.0\n", 2),
])
def test_failed_smoke_or_dependency_chain_restores_original_scaffold(
    tmp_path, outcome, exit_code, requirements, dependencies
):
    # test_failed_smoke_chain_restores_evaluator_and_requirements (exact exit code cases)
    # test_failed_or_unauthorized_dependency_transition_restores_original_scaffold (removal cases)
    run, task = _run_loop(tmp_path, outcome)
    if exit_code is None:
        assert run.returncode != 0
    else:
        assert run.returncode == exit_code, run.stderr
        assert not list((tmp_path / "recovery").glob("*_repair_transaction"))
    assert (task / "evaluate.py").read_text() == "# search\n"
    assert (task / "requirements.txt").read_text() == requirements
    contract = json.loads((task / "task_contract.json").read_text())
    assert len(contract["provenance"]["dependencies"]) == dependencies


def test_new_dependency_transition_remains_provisional_through_followup_repair(tmp_path):
    run, task = _run_loop(tmp_path, "removal_followup")
    assert run.returncode == 0, run.stderr
    assert (task / "evaluate.py").read_text() == "# repaired 1\n"
    assert (task / "requirements.txt").read_text() == "fixture==1.0.0\n"
    assert len(list((tmp_path / "recovery").glob("*.committed.json"))) == 1


# --- gate startup failures remain terminal through production repair callers ----------------

GATE_SETUP = '''
CONTAINERIZE=1
CONTAINER_ENV_FILE=""
CONTAINER_AUTH_ROOT=fixture
AGENT=fixture
CONTRACT_MAX_REPAIRS=3
SMOKE_MAX_REPAIRS=3
ALIGNMENT_MAX_REPAIRS=3
QUALIFICATION_MAX_REPAIRS=3
contract_repairs_used=0
alignment_repairs_used=0
qualification_repairs_used=0
repairs_used=0
preflight_attempt=0
review_round=0
TASK_DRAFTING_WORKFLOW=default
DRAFT_ENTRYPOINT_PATH=./task/program_0/main.py
TASK_GATE_EXTRA_MOUNTS=""
ACTIVE_REPAIR_TRANSACTION=""
seed_auth_into_home() { :; }
discard_staged_runtime_for_repair() { :; }
discard_gate_created_runtime_after_repair() { :; }
task_progress_emit() { :; }
start_task_repair_gate() { printf 'gate\n' >> "$TRACE"; return 4; }
stop_task_repair_gate() { :; }
task_gate_prompt() { printf 'gate prompt\n'; }
write_agent_prompt_file() { printf 'prompt\n' >> "$TRACE"; printf '%s' "$2" > "$1"; }
run_logged_agent_step() { printf 'agent\n' >> "$TRACE"; }
repair_budget_used() { printf '%s' "$1"; }
repair_budget_exhausted() { [ "$1" -ge "$2" ]; }
validate_staged_paths() { printf 'paths\n' >> "$TRACE"; }
canonicalize_staged_scaffold() { printf 'canonicalize\n' >> "$TRACE"; }
'''

GATE_OVERRIDES = '''
start_task_repair_gate() { printf 'gate\n' >> "$TRACE"; }
stop_task_repair_gate() { printf 'stop\n' >> "$TRACE"; }
'''


def _run_gate_flow(tmp_path: Path, functions: tuple[str, ...], body: str) -> subprocess.CompletedProcess[str]:
    tmp_path.mkdir(exist_ok=True)
    for name in ("validation", "task", "recovery", "round"):
        (tmp_path / name).mkdir()
    return subprocess.run(
        ["bash", "-c", "set -eu\n" + GATE_SETUP + "".join(map(_function, functions)) + body],
        cwd=tmp_path,
        env=os.environ | {
            "PYTHON_BIN": sys.executable, "REAL_PYTHON": sys.executable,
            "VALIDATION_ROOT": str(tmp_path / "validation"),
            "STAGED_TASK_DIR": str(tmp_path / "task"),
            "HARNESS_RECOVERY_ROOT": str(tmp_path / "recovery"),
            "ROUND_DIR": str(tmp_path / "round"), "TRACE": str(tmp_path / "trace"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True, text=True, timeout=10,
    )


def _trace(tmp_path: Path) -> list[str]:
    return (tmp_path / "trace").read_text().splitlines()


def _invoke(command: str, context: str) -> str:
    if context == "errexit_disabled":
        body = f"set +e\ncaller_options=$-\n{command}\ncode=$?\n"
    else:
        body = f"caller_options=$-\nif {command}; then code=0; else code=$?; fi\n"
    return body + '[ "$caller_options" = "$-" ] || exit 97\n'


@pytest.mark.parametrize("label", [
    "preflight_repair_0", "smoke_repair_0", "repair_0", "qualification_repair_0",
])
@pytest.mark.parametrize("context", ["conditional", "errexit_disabled"])
def test_gate_start_failure_never_launches_repair_or_writes_prompt(
    tmp_path: Path, label: str, context: str,
) -> None:
    invocation = f'run_scoped_repair {label} fixture "$VALIDATION_ROOT/audit.md"'
    if context == "conditional":
        body = f'if {invocation}; then code=0; else code=$?; fi\n'
    else:
        body = f'set +e\n{invocation}\ncode=$?\n'
    result = _run_gate_flow(tmp_path, ("run_scoped_repair",), body + 'printf "exit=%s\\n" "$code"\n')
    assert result.returncode == 0, result.stderr
    assert result.stdout == "exit=4\n"
    assert _trace(tmp_path) == ["gate"]
    assert not (tmp_path / "task/.agentic_repair_prompt.md").exists()
    assert not (tmp_path / "validation/audit.md").exists()


@pytest.mark.parametrize("context", ["conditional", "errexit_disabled"])
def test_each_repair_caller_propagates_gate_failure_and_rolls_back(
    tmp_path: Path, context: str,
) -> None:
    # test_contract_caller_propagates_gate_failure_without_revalidation
    root = tmp_path / "contract"
    result = _run_gate_flow(root, ("repair_contract_until_valid",), '''
checks=0
validate_staged_contract() { checks=$((checks + 1)); if [ "$checks" -eq 1 ]; then return 3; fi; }
sanitize_preflight_diagnostic() { printf 'invalid contract'; }
render_preflight_repair_prompt() { printf 'repair'; }
run_scoped_repair() { printf 'repair\n' >> "$TRACE"; return 4; }
''' + _invoke("repair_contract_until_valid", context) + '''
printf 'exit=%s checks=%s repairs=%s\n' "$code" "$checks" "$contract_repairs_used"
''')
    assert result.returncode == 0, result.stderr
    assert result.stdout == "exit=4 checks=1 repairs=0\n"
    assert _trace(root) == ["prompt", "repair"]

    # test_alignment_caller_rolls_back_and_stops_infrastructure_failure
    root = tmp_path / "alignment"
    result = _run_gate_flow(root, (
        "repair_alignment_until_structurally_valid", "repair_alignment_transactionally",
        "rollback_pending_alignment_repair", "restore_repair_transaction",
    ), '''
clear_provisional_acceptance() { :; }
fixture_python() { printf '%s\n' "$3" >> "$TRACE"; }
PYTHON_BIN=fixture_python
run_scoped_repair() { printf 'repair\n' >> "$TRACE"; return 4; }
repair_contract_until_valid() { printf 'contract\n' >> "$TRACE"; }
smoke_until_valid() { printf 'smoke\n' >> "$TRACE"; }
render_alignment_validation_retry_prompt() { printf 'retry'; }
''' + _invoke('repair_alignment_transactionally fixture "$VALIDATION_ROOT/prompt"', context) + '''
printf 'exit=%s active=%s\n' "$code" "$ACTIVE_ALIGNMENT_TRANSACTION"
''')
    assert result.returncode == 0, result.stderr
    assert result.stdout == "exit=4 active=\n"
    assert _trace(root) == ["create-authored-tree", "repair", "restore", "commit"]

    # test_smoke_gate_failure_rolls_back_before_authorization_validation
    root = tmp_path / "smoke"
    result = _run_gate_flow(root, ("smoke_until_valid",), '''
SMOKE_MAX_INFRASTRUCTURE_ATTEMPTS=1
QUALIFICATION_TIMEOUT_SECONDS=300
EVALUATOR_SMOKE_TIMEOUT_SECONDS=10
PRODUCTION_ENTRY_TIMEOUT_SECONDS=10
smoke_repairs_used=0
smoke_attempt=0
BYTECODE_QUARANTINE_ROOT="$VALIDATION_ROOT/bytecode"
ACCEPTED_BYTECODE_CACHE=fixture
fixture_python() {
  if [ "$1" = - ]; then "$REAL_PYTHON" "$@"; return $?; fi
  if [ "$1" = -c ]; then printf 'session'; return 0; fi
  printf '%s %s\n' "$2" "$3" >> "$TRACE"
  if [ "$2" = src.investigation.task_repair_authorization ]; then return 3; fi
}
PYTHON_BIN=fixture_python
smoke_required_budget_for_operation() { printf '10'; }
smoke_staged_evaluators() { printf 'authored failure'; return 3; }
typed_smoke_diagnostic() { cat "$1"; }
smoke_report_field() {
  case "$2" in
    status) printf 'authored_failure';;
    category) printf 'search_nonzero_exit';;
    gate) printf 'acceptance-smoke';;
  esac
}
smoke_repair_contract_policy() { printf 'synchronized-digests'; }
render_smoke_repair_prompt() { printf 'repair'; }
run_scoped_repair() { printf 'repair\n' >> "$TRACE"; return 4; }
restore_repair_transaction() { printf 'rollback\n' >> "$TRACE"; ACTIVE_REPAIR_TRANSACTION=""; }
''' + _invoke("smoke_until_valid", context) + '''
printf 'exit=%s repairs=%s active=%s\n' "$code" "$smoke_repairs_used" "$ACTIVE_REPAIR_TRANSACTION"
''')
    assert result.returncode == 0, result.stderr
    assert result.stdout == "exit=4 repairs=0 active=\n"
    trace = _trace(root)
    assert trace[-2:] == ["repair", "rollback"]
    assert not any("task_repair_authorization" in line for line in trace)

    # test_qualification_gate_failure_rolls_back_without_post_validation
    root = tmp_path / "qualification"
    result = _run_gate_flow(root, (
        "repair_qualification_transactionally", "rollback_pending_qualification_repair",
    ), '''
fixture_python() {
  printf '%s %s\n' "$2" "$3" >> "$TRACE"
  if [ "$2" = src.investigation.task_alignment ]; then printf 'sha256:fixture'; fi
}
PYTHON_BIN=fixture_python
render_qualification_repair_prompt() { printf 'repair'; }
run_scoped_repair() { printf 'repair\n' >> "$TRACE"; return 4; }
restore_repair_transaction() { printf 'rollback\n' >> "$TRACE"; ACTIVE_REPAIR_TRANSACTION=""; }
''' + _invoke("repair_qualification_transactionally report", context) + '''
printf 'exit=%s repairs=%s active=%s\n' "$code" "$qualification_repairs_used" "$ACTIVE_QUALIFICATION_TRANSACTION"
''')
    assert result.returncode == 0, result.stderr
    assert result.stdout == "exit=4 repairs=0 active=\n"
    trace = _trace(root)
    assert trace[-2:] == ["repair", "rollback"]
    assert len(trace) == 4


_SETUP_FAILURE_STEPS = [
    "seed_auth_into_home", "discard_staged_runtime_for_repair", "write_agent_prompt_file",
    "stop_task_repair_gate", "discard_gate_created_runtime_after_repair",
]


def test_scoped_repair_reports_setup_cleanup_and_agent_exits_as_infrastructure_or_verbatim(
    tmp_path: Path,
) -> None:
    # test_scoped_repair_setup_and_cleanup_failures_remain_infrastructure_failures
    for failure_step in _SETUP_FAILURE_STEPS:
        root = tmp_path / failure_step
        body = GATE_OVERRIDES + f'{failure_step}() {{ printf "failure\\n" >> "$TRACE"; return 75; }}\n'
        body += '''
if run_scoped_repair smoke_repair_0 fixture "$VALIDATION_ROOT/audit"; then code=0; else code=$?; fi
printf 'exit=%s\n' "$code"
'''
        result = _run_gate_flow(root, ("run_scoped_repair",), body)
        assert result.returncode == 0, result.stderr
        assert result.stdout == "exit=4\n", failure_step
        trace = _trace(root)
        assert "failure" in trace, failure_step
        if failure_step in {"stop_task_repair_gate", "discard_gate_created_runtime_after_repair"}:
            assert trace.count("agent") == 1, failure_step
        else:
            assert "agent" not in trace, failure_step
        if failure_step == "write_agent_prompt_file":
            assert trace[-1] == "stop"

    # test_scoped_repair_preserves_agent_exit_and_disabled_errexit
    for agent_exit in (0, 4, 9, 75):
        root = tmp_path / f"agent-exit-{agent_exit}"
        body = GATE_OVERRIDES + (
            f'run_logged_agent_step() {{ printf "agent\\n" >> "$TRACE"; return {agent_exit}; }}\n'
        )
        body += '''
set +e
before_options=$-
run_scoped_repair smoke_repair_0 fixture "$VALIDATION_ROOT/audit"
code=$?
after_options=$-
printf 'exit=%s options_unchanged=%s\n' "$code" "$([ "$before_options" = "$after_options" ] && echo yes)"
'''
        result = _run_gate_flow(root, ("run_scoped_repair",), body)
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"exit={agent_exit} options_unchanged=yes\n"
        trace = _trace(root)
        assert trace.count("agent") == 1, agent_exit
        assert trace[-1] == "stop", agent_exit

    # test_gate_runtime_cleanup_failure_retains_marker_for_outer_cleanup
    root = tmp_path / "runtime-cleanup"
    result = _run_gate_flow(root, ("discard_gate_created_runtime_after_repair",), '''
GATE_RUNTIME_CREATED_MARKER="$VALIDATION_ROOT/runtime-created"
printf 'gate-created\n' > "$GATE_RUNTIME_CREATED_MARKER"
discard_staged_runtime_for_repair() { printf 'discard\n' >> "$TRACE"; return 75; }
if discard_gate_created_runtime_after_repair; then code=0; else code=$?; fi
[ "$code" -ne 0 ] || exit 98
[ "$GATE_RUNTIME_CREATED_MARKER" = "$VALIDATION_ROOT/runtime-created" ] || exit 99
''')
    assert result.returncode == 0, result.stderr
    assert _trace(root) == ["discard"]
    assert (root / "validation/runtime-created").read_text() == "gate-created\n"


# --- launcher forwards private repair context only to smoke gate sessions -------------------


@pytest.mark.parametrize("has_repair", [False, True])
def test_gate_launcher_forwards_complete_optional_repair_context(has_repair: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="gate-args-") as temporary:
        root = Path(temporary)
        shim = root / "python"
        shim.write_text(f"#!{sys.executable}\n" + '''import json, os, signal, socket, sys
from pathlib import Path
if sys.argv[1:3] != ["-m", "src.task_drafting.service"]:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
Path(os.environ["ARGUMENT_TRACE"]).write_text(json.dumps(sys.argv[1:]))
address = sys.argv[sys.argv.index("--socket") + 1]
server = socket.socket(socket.AF_UNIX)
server.bind(address)
server.listen(1)
if "--ready-file" in sys.argv:
    Path(sys.argv[sys.argv.index("--ready-file") + 1]).write_text("ready\\n")
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
try:
    signal.pause()
finally:
    server.close()
    Path(address).unlink(missing_ok=True)
''')
        shim.chmod(0o700)
        env = os.environ | {
            "PYTHON_BIN": str(shim), "REPO_ROOT": str(ROOT), "VALIDATION_ROOT": str(root),
            "STAGED_TASK_DIR": str(root / "task"), "TASK_SOURCE_DIR": str(root / "source"),
            "CONTAINER_IMAGE": "fixture", "CONTAINER_NETWORK": "none",
            "EVALUATOR_SMOKE_TIMEOUT_SECONDS": "10", "PRODUCTION_ENTRY_TIMEOUT_SECONDS": "10",
            "TASK_DRAFT_MAX_RUNTIME_BYTES": "1000", "MAX_GATE_CHECKS_PER_REPAIR": "3",
            "RUNTIME_INSTALL_CACHE_ROOT": str(root / "cache"), "CONTAINERIZE": "0",
            "ARGUMENT_TRACE": str(root / "arguments.json"),
        }
        if has_repair:
            env |= {
                "TASK_REPAIR_SNAPSHOT": str(root / "snapshot"),
                "TASK_REPAIR_TRANSACTION": str(root / "transaction"),
                "TASK_REPAIR_AUTHORIZATION": str(root / "authorization"),
                "TASK_REPAIR_SESSION_ID": "repair-session",
            }
        run = subprocess.run(["bash", "-c", '''set -eu
source "$REPO_ROOT/src/scripts/lib/task_gate.sh"
write_task_gate_token() { printf 'fixture' > "$1"; }
trap stop_task_repair_gate EXIT
start_task_repair_gate fixture
'''], cwd=ROOT, env=env, text=True, capture_output=True, timeout=10)
        assert run.returncode == 0, run.stderr
        arguments = json.loads((root / "arguments.json").read_text())
        for flag, variable in (
            ("--repair-snapshot", "TASK_REPAIR_SNAPSHOT"),
            ("--repair-transaction", "TASK_REPAIR_TRANSACTION"),
            ("--repair-authorization", "TASK_REPAIR_AUTHORIZATION"),
            ("--repair-session-id", "TASK_REPAIR_SESSION_ID"),
        ):
            if has_repair:
                assert arguments[arguments.index(flag) + 1] == env[variable]
            else:
                assert flag not in arguments


# --- repair environment helpers propagate failures inside shell conditionals ----------------


def _run_env(tmp_path: Path, body: str) -> subprocess.CompletedProcess[str]:
    tmp_path.mkdir(exist_ok=True)
    source = CONTAINER_AGENT_LIB.read_text()
    functions = []
    for name in ("seed_auth_into_home", "seed_managed_agent_files", "sanitize_env_file_for_docker"):
        match = re.search(rf"^{name}\(\) \{{.*?^\}}", source, re.MULTILINE | re.DOTALL)
        assert match
        functions.append(match.group())
    setup = '''
chatgpt_subscription_enabled() { return 1; }
seed_paths_for_agent() { printf '%s\\n' .codex/auth.json .config/fixture; }
agent_config_rel_dir() { printf '.codex'; }
agent_env_key_is_denied() { [ "$1" = DENIED ]; }
'''
    return subprocess.run(
        ["bash", "-c", "set -eu\n" + "\n".join(functions) + setup + body],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
        env=os.environ | {
            "PYTHON_BIN": sys.executable, "CONTAINER_AGENT_LIB_DIR": str(CONTAINER_AGENT_LIB.parent),
        },
    )


_AUTH_SETUP_FAULTS = {
    "mkdir_home": 'mkdir() { if [ "$2" = home ]; then return 9; fi; command mkdir "$@"; }',
    "mkdir_parent": 'mkdir() { if [ "$2" = home/.codex ]; then return 9; fi; command mkdir "$@"; }',
    "copy_file": 'cp() { if [ "$1" != -R ]; then return 9; fi; command cp "$@"; }',
    "copy_dir": 'cp() { if [ "$1" = -R ]; then return 9; fi; command cp "$@"; }',
    "managed": 'seed_managed_agent_files() { return 9; }',
    "paths": 'seed_paths_for_agent() { return 9; }',
    "config": 'agent_config_rel_dir() { return 9; }',
}


def test_auth_setup_propagates_failures_and_allows_missing_optional_credentials(tmp_path: Path):
    # test_auth_setup_failure_is_not_masked_by_later_success
    for failure, fault in _AUTH_SETUP_FAULTS.items():
        root = tmp_path / failure
        auth = root / "auth"
        (auth / ".codex").mkdir(parents=True)
        (auth / ".codex/auth.json").write_text("fixture")
        (auth / ".config/fixture").mkdir(parents=True)
        result = _run_env(root, fault + '''
if seed_auth_into_home fixture auth home; then code=0; else code=$?; fi
printf 'exit=%s\n' "$code"
''')
        assert result.returncode == 0, result.stderr
        assert result.stdout == "exit=9\n", failure

    # test_auth_setup_allows_missing_optional_credentials
    root = tmp_path / "missing-optional"
    result = _run_env(root, '''
if seed_auth_into_home fixture missing-auth home; then code=0; else code=$?; fi
printf 'exit=%s\n' "$code"
''')
    assert result.returncode == 0, result.stderr
    assert result.stdout == "exit=0\n"
    assert (root / "home/.codex").is_dir()


_ENVIRONMENT_WRITE_FAILURES = [
    ("append-restricted", "ALLOWED=fixture\n", "/dev/full", " ALLOWED"),
    ("append-unrestricted", "legacy line\n", "/dev/full", ""),
    ("truncate-directory", "# no values\n", "destination", " ALLOWED"),
]


def test_managed_auth_and_environment_file_failures_propagate_and_filtering_is_preserved(
    tmp_path: Path,
):
    # test_managed_subscription_auth_failure_stops_later_setup
    for agent, failed_call in (("pi", 1), ("opencode", 1), ("opencode", 2)):
        result = _run_env(tmp_path / f"managed-{agent}-{failed_call}", f'''
chatgpt_subscription_enabled() {{ return 0; }}
calls=0
fixture_python() {{ calls=$((calls + 1)); if [ "$calls" -eq {failed_call} ]; then return 9; fi; }}
PYTHON_BIN=fixture_python
if seed_managed_agent_files {agent} auth home; then code=0; else code=$?; fi
printf 'exit=%s calls=%s\n' "$code" "$calls"
''')
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"exit=9 calls={failed_call}\n", (agent, failed_call)

    # test_environment_write_failures_remain_failures
    for name, source_env, destination, allowed in _ENVIRONMENT_WRITE_FAILURES:
        root = tmp_path / name
        root.mkdir()
        (root / "source.env").write_text(source_env)
        if destination == "destination":
            (root / "destination").mkdir()
        result = _run_env(root, f'''
if sanitize_env_file_for_docker source.env {destination}{allowed}; then code=0; else code=$?; fi
printf 'exit=%s\n' "$code"
''')
        assert result.returncode == 0, result.stderr
        assert result.stdout != "exit=0\n", name

    # test_environment_filtering_semantics_are_preserved
    root = tmp_path / "filtering"
    root.mkdir()
    (root / "source.env").write_text('ALLOWED="fixture"\nDENIED=secret-fixture\nOTHER=omit\n')
    result = _run_env(root, '''
if sanitize_env_file_for_docker source.env output.env 'ALLOWED DENIED'; then code=0; else code=$?; fi
printf 'exit=%s\n' "$code"
''')
    assert result.returncode == 0, result.stderr
    assert result.stdout == "exit=0\n"
    assert (root / "output.env").read_text() == "ALLOWED=fixture\n"
