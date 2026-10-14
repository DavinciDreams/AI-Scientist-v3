# Papers from no-winner studies

The pipeline separates investigation success from selecting a program to study. With the default
`execution.no_winner_policy=best_available`, selected downstream phases continue after a verified
`completed_no_winner` search. `execution.no_winner_policy=skip` restores the previous behavior.

## Selection and evidence

Selection uses completed search evaluations with `correct=true` and a finite score. Generated
programs are ranked by descending score, then recorded attempt order and program index. Seeded
`program_0` copies are excluded. A failed improvement rubric does not exclude an otherwise valid
program. If no generated candidate qualifies, the pipeline uses a valid scored retained baseline.
Without either, it skips downstream phases explicitly.

The host creates `.downstream_selection.json` and `downstream/program` beneath the run directory.
The selection binds the original completion receipt, attempt history, source artifacts, baseline,
and frozen program. All downstream consumers use this selection; the investigation winner record
remains unchanged. Extraction provenance rejects bundles prepared for a different selection.

On filesystems such as NFS that lack exclusive directory rename, publication reserves
`downstream/program` with an exclusive directory creation, copies and verifies the staged code,
then atomically creates the selection manifest without replacement. Consumers require that
manifest; a partially copied tree is never a valid selection. Existing destination entries are
never adopted or overwritten. An interrupted copy without a manifest still fails closed and
requires inspection before retrying.

Final evaluation uses the frozen program. A completed, verified `not_qualified` result may support
a negative-results manuscript, with the failed qualification and its limitations preserved.
Non-qualification is not automatically evidence of inferior performance: the evaluator may also
reject validity or other criteria. Failed, incomplete, or unverifiable required final evaluations
still block publication; invalid optional evidence remains excluded. Mismatched selection hashes
and unsafe paths remain errors. The task presets permit papers after a winner's completed
non-qualifying final evaluation; pipeline calls without a preset require an explicit policy.
See [final outcome policy and resume commands](paper_final_outcomes.md).

Paper evidence distinguishes scored and unscored attempts and retains readable selected-program
and baseline sources for audit and revision. Before each audit, the host checks these copies and
study facts against the frozen selection; author changes cannot replace the evaluated source.
Invalid baselines do not become grounded score comparisons. Compilation, references, provenance,
and evidence-audit gates remain mandatory.

## Resume a run with no winner

When search retained its baseline and only one attempt produced a valid, lower-scoring
candidate, the fallback under `best_available` is that single measured candidate. Attempts
with no recorded score are not measured losses.

Preview the downstream continuation without restarting drafting or investigation:

```bash
study_run=results/<task>/<agent>/<run_id>
agsci run \
  task.dir="$study_run/task" \
  paths.run_dir="$study_run" \
  paths.paper_dir="$study_run/paper" \
  agent.name=codex agent.model=gpt-5.6-sol agent.reasoning_effort=high \
  container.enabled=true \
  execution.no_winner_policy=best_available \
  execution.phases=ablations,final_evaluation,posthoc,paper,review,revision \
  paper.standalone_paper=true \
  --dry-run
```

Dry-run mode validates the continuation without launching evaluations or agents; it
still emits pipeline logs and history checkpoints. Remove `--dry-run` to perform the
evaluations and agent calls. Resume reuses a valid existing selection; it will not silently
repair changed evidence or choose another program after final evaluation. A partially
published snapshot without its selection manifest fails closed and needs inspection.

## Focused verification

```bash
.venv/bin/pytest -q tests/revision/test_downstream_publication.py tests/revision/test_downstream_selection.py
.venv/bin/pytest -q tests/pipeline/test_pipeline_cli_dryrun.py
.venv/bin/pytest -q tests/paper/test_paper_final_outcome.py tests/revision/test_revision_evidence.py
```

Tests use temporary artifacts and stub external execution. They do not require GPUs, live agents,
or rerunning the recorded Bayesian-optimization experiment.
