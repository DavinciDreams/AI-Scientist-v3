# Papers after final non-qualification

The task presets under `configs/preset/` set `paper.final_outcome_policy` to
`allow_not_qualified`, permitting paper writing, review, and revision after a
completed, verified final evaluation returns `not_qualified`. `preset=<task>` applies
the policy on new runs and resumes. Pass `paper.final_outcome_policy=qualified_only`
to require qualification before writing a paper; repeat that override on each
invocation because the pipeline does not recover the policy from a previous run.

This changes manuscript eligibility, not scientific results. The final outcome
remains `not_qualified`, and manuscripts must report the qualification failure and
its limitations. Statistical uncertainty and adaptivity failures need different
interpretations; neither alone proves worse performance. Required evidence must
still be complete and verified. Missing, failed, incomplete, or unverifiable finals
remain blocking. Existing no-winner studies retain their selection and evidence
requirements.

Pipeline calls without a preset retain the `qualified_only` default; set
`paper.final_outcome_policy=allow_not_qualified` explicitly there to permit
non-qualifying final results.

## Resume a recorded run for paper writing

Run from the repository root with Docker available and the Python environment
installed. These commands use existing experiment artifacts and run only paper,
review, and revision. The pipeline recovers the run's original idea-cycle budget
from the run directory; the preset's GPU allocation still applies (`preset=grpo`
requires two visible GPUs), so pass the original harness and model and keep a
matching GPU allocation.

Claude:

```bash
agsci run examples/grpo preset=grpo \
  agent.name=claude agent.model=claude-opus-5-5 \
  paths.run_dir=results/grpo/claude/<run_id> \
  execution.phases=paper,review,revision \
  paper.final_outcome_policy=allow_not_qualified
```

Codex:

```bash
agsci run examples/grpo preset=grpo \
  agent.name=codex agent.model=gpt-5.6-sol \
  paths.run_dir=results/grpo/codex/<run_id> \
  execution.phases=paper,review,revision \
  paper.final_outcome_policy=allow_not_qualified
```

If the run lives outside `results/`, point `paths.run_dir` at its actual run
directory.

Focused validation:

```bash
./.venv/bin/pytest -q tests/pipeline/test_pipeline_phases.py tests/pipeline/test_pipeline_cli_dryrun.py -k 'preset or budget'
```
