# Evaluation evidence

Sealed search and final evaluators may write `evaluation_evidence.json` alongside
`metrics.json` and `correct.json`. Existing evaluators producing only those two files
remain supported; their scientific evidence availability is reported as `unavailable`.

New drafted tasks should retain task-relevant validated outputs before aggregation.
For circle packing, retain circles, per-seed scores and measured episode durations.
The evaluator owns this file. Candidates return finite inline JSON through the existing
broker; they cannot write into evaluator output directories or provide artifact paths.

```json
{
  "schema_version": 1,
  "episodes": [
    {
      "seed": 7,
      "score": 2.5,
      "correct": true,
      "duration_seconds": 0.1,
      "solution": [[0.5, 0.5, 0.25]],
      "candidate_diagnostics": {"objective_calls": 12},
      "candidate_timeseries": [{"series_name": "loss", "step": 0, "value": 0.5}]
    }
  ]
}
```

The first four episode fields are required. The last three are optional.
The evaluator must compute score/correctness and measure duration itself; persist
solutions only after task-specific validation. Candidate diagnostic numbers and
timeseries are explicitly **unverified**, even inside an evaluator-authored file.
They do not certify feasibility, correctness or hard objective-count budgets.

Limits: 16 MiB per file; 1–10,000 unique signed 64-bit seeds; nesting depth at most 12;
100,000 total timeseries rows. All numbers must be finite. Duration is nonnegative;
timeseries steps are nonnegative integers. Diagnostic/series names use letters,
digits, underscores, hyphens and periods, at most 80 characters, beginning with a
letter, digit, underscore or hyphen. Duplicate JSON keys, unknown schema fields,
symlinks and nonregular files fail validation.

Production task qualification and sealed search use the same read-only output validator for
contracted metrics, correctness, and optional evidence. Malformed evidence is an authored
evaluator-contract defect even when metrics are finite and qualification reports feasible.
Drafting, repair, and alignment prompts receive one repository-maintained evidence schema.

During investigation, a host-confirmed evaluator output-contract violation terminates the
campaign before promotion or another idea. It is not a scientific rejection or a completed
no-winner search. The host records `evaluator_output_contract_invalid` with task scope; campaign
control uses a private inherited channel, not candidate-written manifests or console text.
The pipeline stops on this terminal result even with `execution.fail_fast=false`.

Search promotion retains the JSON and emits candidate series under
`candidate/seed_<seed>/<series_name>` with host-owned evaluation/variant IDs.
Posthoc staging hashes and seals retained files. Paper data reports availability
and carries per-seed records; legacy linked search directories report unavailable
rather than following links.

When verified final observations are available, paper preparation registers
`final.evidence.episodes` as their canonical claim source. It identifies the evidence
file, hash and `/episodes` JSON pointer; the records remain in
`final_evaluation.evaluation_evidence.episodes`. Scalar final metrics retain
`final.metric.*` IDs. Candidate diagnostics and timeseries remain unverified.
Authors must not extend the reserved final grounding registry themselves. Legacy
papers without the episode source remain readable, but fabricated or altered sources
still fail the final-evidence check.

Final manifest schema 3 seals evidence through `results_tree_sha256` and records
its digest/availability plus host-measured `compute_usage`. Readers continue to
accept schema 1 and 2. Hard limits are wall-clock duration and broker RPC count;
internal objective counts are advisory. Missing broker measurements are `null`,
never fabricated zeroes.

Brokered ablation preparation preserves the selected candidate byte-for-byte.
Additional scientific measurements belong in the evaluator before the task is
sealed, rather than in an instrumented replacement candidate.

## Revision posthoc retention

Revision preparation rebuilds paper data from the run, then restores the original
posthoc records from `previous_paper/paper_data.json` before ranking and augmentation.
Original ranks, posthoc totals, and `posthoc.*` grounding records remain authoritative;
unrelated rebuilt evidence is retained and newly discovered analyses may be appended.
Legacy papers without posthoc inventory remain supported. Malformed original inventory
or nonregular original paper data fails preparation before revision agents run.

The manuscript evidence audit checks current claims against current evidence. The later
revision-retention check also compares against the original paper; a passing audit alone
does not permit changed ranks or lost evidence. Restoring historical revision data requires
fresh audit validation before promotion; this preparation fix does not modify saved runs.

Focused preparation, augmentation, and retention checks:

```bash
.venv/bin/pytest -q tests/revision/test_paper_revision_workspace.py tests/paper/test_paper_data.py tests/revision/test_revision_evidence.py tests/revision/test_pipeline_revision.py
```

## Starting a new circpack experiment

For a new run, start from `examples/circpack_task_drafting`, retaining the desired
model/search limits. Preview the complete phase sequence first:

```bash
agsci run \
  task.dir=examples/circpack_task_drafting agent.name=codex \
  execution.phases=task_drafting,investigation,ablations,final_evaluation,posthoc,paper,review,revision \
  --dry-run
```

A subsequent real run must qualify a new evaluator with enforced runtime/RPC
limits and retained scientific observations, then repeat investigation, controlled
ablations, and final evaluation before producing posthoc analyses and a manuscript.
Compare controls under the same declared budget and inspect per-seed results;
historical retrofit-score loss remains an empirical question requiring replay.
The command above only previews; removing `--dry-run` incurs model and compute costs.
