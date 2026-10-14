# GPU ownership, automatic recovery, and quarantine

A later run can be given a GPU that agentic-scientist previously quarantined. Host GPU
assignment and the application's cleanup state are separate. A new run, an ended run, an idle
GPU, or a reboot does not itself establish safe reuse. The launcher uses visible
GPU UUIDs and verifies the same UUID set inside Docker; lease identities use those UUIDs rather
than job-relative CUDA ordinals.

Quarantine blocks a GPU when evaluator/container/launch-process shutdown could not be verified.
Managed GPU runtimes now record ownership before launch and automatically reclaim abandoned work
when shutdown can be verified. It is an application cleanup flag, not a hardware-health diagnosis.
Lease files are node-local,
private to the user, and shared across that user's jobs:

```text
/tmp/agentic-scientist-gpu-leases-<uid>/<sha256(casefold GPU UUID)>.lock
```

V3 markers bind to a private `runtime-<random-id>.json` ownership journal in the same directory.
The journal records the owner and launcher PID, kernel boot ID, process start time and UID; all
allocated GPU UUIDs; the Docker daemon identity; and acknowledged container IDs with ownership
labels. Provenance timestamps and host names remain diagnostic, never termination authority.
Legacy v1 markers and diagnostic-only v2 markers cannot support automatic termination.
Malformed, missing, oversized, or inconsistent ownership records continue to block allocation.

## Automatic recovery

Contracted GPU drafting qualification, production canaries, final smoke, investigation/variant
search, and containerized final evaluation prepare ownership while holding
the complete allocation's advisory locks, before either the evaluator or candidate broker starts.
Durable markers therefore survive an owner crash, including `SIGKILL`. CPU execution keeps its
existing launch path. Preflight checks can recover old ownership but create no new runtime journal.
This lifecycle covers managed evaluator/candidate runtimes. Legacy tasks without contracts,
native execution, and coding-agent containers retain their existing lifecycle.

Each managed Docker launch uses an isolated trusted Python helper. It registers its own identity
under the journal lock, creates the container, persists its exact ID, and only then starts it.
Create and start clients also register before executing Docker; a delayed child cannot bypass
revocation. Container IDs never enter candidate RPC stdout, and attached stdin/stdout/stderr and
the container's exit status remain intact. Managed commands pin the local Docker Unix endpoint
`unix:///var/run/docker.sock`; remote Docker contexts are not used for these launches.

Lease acquisition attempts bounded recovery of a V3 runtime:

1. Hold every GPU lock from the old allocation, including devices outside the new request.
2. Confirm the recorded owner is gone. A live owner or a busy lock remains untouched; contention
   on an additional old-allocation GPU returns the retryable busy category.
3. Persist a revoked journal state, fencing all future supervisor and broker launches.
4. Stop recorded supervisors, then create/start clients using pidfds; verify their exit.
5. Check the original Docker daemon and container ownership labels; remove exact container IDs
   and verify absence. An unacknowledged create with no discoverable container stays quarantined:
   the daemon may still be processing the old creation request.
6. Persist recovery, clear the original GPU marker inodes under their locks, and continue acquisition.

Normal lease release uses the same cleanup with authority from the current owner. It may terminate
that evaluation's remaining launchers after the caller has ended the evaluation. Failed cleanup
retains the journal and markers. A later acquisition retries recovery; no age-based expiry occurs.
Cleanup uncertainty takes precedence over ordinary evaluator errors as an infrastructure failure;
the original error remains chained. Keyboard interrupts and process-exit signals retain their
control flow while leaving quarantine intact.
Ownership journals retain bounded recovery history; do not blanket-prune the lease directory.

Drafting keeps its short busy-lease wait and allows up to 30 seconds for ownership recovery.
Runtime acquisition remains bounded by its execution deadline; release cleanup has a 30-second
bound. Remaining quarantine is terminal for that pipeline attempt. Recovery does not kill
another active lease, reschedule a run, or silently reduce the requested GPU count.

## Inspect on the affected node

Use the same user and node as the failed run. Copy the stable UUID from the preflight message or
the allocation's `nvidia-smi --query-gpu=index,uuid --format=csv,noheader` output.

```bash
./.venv/bin/python -m src.investigation.gpu_lease_admin status --device "GPU-<uuid>"
```

`status` creates no files. It reports clear/missing/quarantined state, bounded origin metadata,
and the marker's SHA-256. A busy lease cannot be inspected consistently and returns exit 4.
`--lease-root` exists for alternate controlled storage; ordinary operation uses the default root.
Do not choose a fresh root to bypass an existing quarantine.

## Legacy recovery after verified external cleanup

First verify that the affected evaluator and candidate containers, their Docker launch processes,
and any broker workers that could start/restart them have stopped. Use retained container IDs,
run logs, and process state; absence of current GPU compute processes alone is insufficient.
If those identities or shutdown facts cannot be established, keep the quarantine and use a healthy
allocation or involve the node operator.

After that verification, copy the exact inspected marker digest and explicitly attest cleanup:

```bash
./.venv/bin/python -m src.investigation.gpu_lease_admin recover \
  --device "GPU-<uuid>" \
  --marker-sha256 "<inspected-sha256>" \
  --confirm-cleanup
```

This command enforces the original inode's exclusive lock, private ownership, unchanged file/root
identity, and matching bounded marker bytes. It truncates and synchronizes that inode in place.
For legacy markers, it does not terminate processes. For V3 markers, it runs the full owned-runtime
recovery above; `--confirm-cleanup` cannot bypass a live owner or incomplete cleanup proof.
It never deletes or replaces a lock or changes host GPU assignment. A changed
marker, busy lease, unsafe file, or oversized record refuses recovery. Recovery restores the original
marker on a failed update when storage permits; an unverified restoration is reported explicitly.
Legacy external shutdown verification remains the operator's responsibility.

After successful recovery, rerun the normal launcher/preflight. Runtime gates still reacquire their
own leases. Repeat verification separately for each affected GPU; never blanket-prune lease files.

## Focused regressions

```bash
./.venv/bin/pytest -q tests/gpu/
```

Interrupted baseline evaluation avoids quarantine only when trusted host cleanup establishes both
evaluator launch-process termination and container removal, and broker teardown verifies candidate
shutdown. Missing identity or incomplete cleanup keeps quarantine. Broker socket setup failures
before launch remain infrastructure failures without quarantining unused GPUs.

Crash recovery tests use disposable host processes, real advisory locks and pidfds, and a fake
Docker CLI. They cover acquisition/preflight recovery, removal failure and retry, and live-lease
protection without GPUs or a Docker daemon. They do not replace a cluster GPU smoke run.
