# Temporary storage

New pipeline and investigation CLI invocations select scratch before starting work and report the
selected location and available bytes. Selection lives in
[`temp_storage.py`](../src/investigation/temp_storage.py); entrypoint reporting and compatibility
live in [`temp_bootstrap.py`](../src/investigation/temp_bootstrap.py).

## Selecting a base

`AGENTIC_TEMP_ROOT` selects an **existing, writable, absolute directory**. The initializer creates
its private namespace beneath that base; it does not create a missing base or mount a disk.
Explicit paths cannot contain parent traversal, symlink components, or line-control characters.
Writable shared ancestors must have the sticky bit; private namespace directories must be owned
by the current user with mode `0700`.
An invalid explicit setting fails initialization rather than silently falling back to `/tmp`.

Selection order:

1. `AGENTIC_TEMP_ROOT`.
2. `TMPDIR`, if explicitly set.
3. `/mnt/localssd`, when writable and actually mounted on a filesystem distinct from `/`.
4. The platform temporary directory, including `TMP`/`TEMP` when configured.

For a standalone host such as Katya, choose a scratch location already provisioned on that
host. The following path is a placeholder, not a claim about Katya's inspected configuration:

```bash
export AGENTIC_TEMP_ROOT=/path/to/existing/scratch
agsci run task.dir=examples/circpack --dry-run
agsci run task.dir=examples/circpack
```

Replace the placeholder with the existing writable scratch directory before running the commands.
The dry run previews selection and capacity without creating the scratch namespace.

## Host and container layout

The initializer creates user-owned directories with mode `0700`:

```text
<base>/agentic-scientist-<uid>/
  temp/run-<uuid>/             host temporary files and bounded log spools
  cache/                      runtime installation caches shared across phases
  containers/run-<uuid>/       private parent for individual agent scratch leaves
```

Host `TMPDIR`, `TMP`, and `TEMP` point to the invocation's `temp/run-<uuid>` directory.
Python's cached temporary-directory setting is updated too. Runtime caches keep their
task-path-derived identities beneath `cache/`. Child processes validate and reuse inherited
`AGENTIC_TEMP_DIR`, `AGENTIC_CACHE_ROOT`, and `AGENTIC_CONTAINER_TEMP_ROOT`; partial, unrelated,
symlinked, incorrectly owned, or nonprivate inherited state fails validation.

Agent containers receive an individual `agent-tmp-*` leaf mounted at `/agent-tmp`, with container
`TMPDIR`, `TMP`, and `TEMP` set to that path. Host control files, log spools, and installation
caches are outside that leaf. This avoids passing an unusable host scratch path into the
container or exposing the host's whole temporary namespace.

[`container_temp.py`](../src/scripts/lib/container_temp.py) validates the leaf, container label,
and writable bind mount when reusing a session. A missing or mismatched scratch identity requires
an explicit restart. Cleanup retains a leaf if the container still exists or Docker inspection
cannot establish its removal. One-shot successful runs attempt cleanup after container removal;
successful task publication, paper/revision finalization, figure-repair stages, and investigation CLI
completion also retire their verified reusable-session scratch after stopping the container.
Failed or interrupted runs retain scratch for diagnosis.
Direct `run.sh` callers using reusable sessions own final teardown: call
`complete_container_session` on overall success, or `stop_container_session_force` on failure
or interruption. Individual successful attempts keep the session available to later attempts.

This policy changes application scratch placement. It does not relocate Docker's daemon data,
image layers, or other applications' storage, and it does not prune their data.

Candidate brokers use a separate, short socket root at
`/tmp/agentic-scientist-broker-<uid>`.
This fixed path avoids Unix-domain socket length limits regardless of `TMPDIR`. The root
must be a real directory owned by the current user with mode `0700`; unsafe existing roots
are rejected without changing their permissions. Containers mount only the required broker
directory. Ordinary broker cleanup and GPU ownership checks still apply.

Previously, these brokers used `/dev/shm`. On hosts with systemd-logind `RemoveIPC=yes`,
the last login-session logout can remove user-owned entries there while work continues.
Moving the socket root prevents that logout cleanup from unlinking an active broker socket.
Matching legacy broker state blocks a new launch pending inspection; it is not silently
migrated or deleted. Unrelated legacy broker directories remain untouched. Do not clear either
root while work is active.

Task-repair gates use a separate, short IPC path:
`/dev/shm/agentic-task-gates-<uid>/gate-<random>/s`. Linux Unix-domain socket paths must fit
within 107 encoded bytes; managed scratch paths plus validation and repair labels can exceed
that limit. The per-user IPC root and each session directory must be owned by the current user
with mode 0700, without symlink ancestors; the socket has mode 0600. Missing or unsafe
`/dev/shm` fails startup explicitly, without falling back to bulk scratch.

Tokens, service logs, readiness markers, summaries, runtime markers, and repair authorization
stay under the existing private validation/recovery roots. Containers receive only the socket
at `/run/agentic-task-gate.sock` and the read-only client. Normal shutdown removes the empty
session directory after the service closes. Startup abort terminates and reaps the partial
service; residual sockets require a matching captured identity before cleanup. Unknown or
replaced contents are retained and reported; the shared per-user root is never removed.

## Existing runs and recovery

Direct temporary-directory and runtime-cache helpers retain legacy `/tmp` paths when no managed
storage options are present.

A configured scratch failure is reported instead of silently moving work to another filesystem.
The bounded logger still drains agent output and attempts the shared log if its local spool
cannot be created; it reports the storage failure. Scratch and recovery spools remain temporary,
bounded recovery aids, not durable experiment archives. Keep run artifacts and logs under the
configured results directory.

Recover capacity by identifying specific inactive owned directories. Check active task identities,
Docker mounts, and live process references; acquire an existing runtime cache's `.lock` before
removal. Preserve active caches, logs, quarantine records, and unfamiliar directories. Do not use
blanket `/tmp` deletion or Docker pruning as experiment recovery.

## Focused verification

```bash
./.venv/bin/pytest -q tests/runtime/test_temp_storage.py tests/pipeline/test_pipeline_cli_dryrun.py
./.venv/bin/pytest -q tests/runtime/test_temp_storage.py tests/runtime/test_bounded_process.py
./.venv/bin/pytest -q tests/runtime/test_temp_storage.py
./.venv/bin/pytest -q tests/final_evaluation/test_candidate_broker.py
./.venv/bin/pytest -q tests/investigation/test_investigation_cli_events.py tests/runtime/test_temp_storage.py
./.venv/bin/pytest -q tests/task_drafting/test_task_gate.py
```

These checks cover selection, invalid/inherited paths, legacy compatibility, startup ordering,
dry-run behavior, container scratch isolation, and log recovery. They do not prove that a
particular host's scratch mount has sufficient capacity.
