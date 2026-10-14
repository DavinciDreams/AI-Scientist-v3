# Post-hoc Pipeline

The post-hoc pipeline runs after a completed scientist run and produces
analysis-only artifacts for downstream paper writing. It does not search for new
program variants or ablations. Instead, it:

1. extracts the best program from a completed run,
2. bootstraps a single checkpoint-ready base bundle,
3. proposes and filters analysis ideas that use existing artifacts only,
4. materializes one workspace per accepted analysis,
5. runs one agent per analysis with retries, and
6. publishes all valid analyses for downstream paper-agent selection.

## Design architecture

The post-hoc pipeline is a standalone package under `src/posthoc_pipeline/` with
its own CLI:

```bash
python -m src.posthoc_pipeline.cli \
    paths.task_dir=examples/circpack \
    paths.results_dir=results/circpack/claude/20260331_094917
```

Key design rules:

- It is standalone and run-scoped. Each invocation writes to a fresh output dir
  under `results_dir/posthoc/` unless `paths.output_dir` is set explicitly.
- It is analysis-only. Generated analyses must inspect existing checkpoints,
  logs, metrics, predictions, and manifests. They must not retrain, fine-tune,
  rerun `evaluate.py`, or create new program variants.
- Stage 1 is the only checkpoint-creation stage. If the extracted winner does
  not already have a usable checkpoint, bootstrap may create one once. Later
  stages treat `task_bundle/program_0` as read-only model state.
- Final ranking is delegated to the paper agent. Stage 6 publishes all valid
  analyses instead of choosing a single winner.

## Pipeline stages

The pipeline has 7 stages:

| Stage | Name | LLM? | Description |
|-------|------|------|-------------|
| 0 | Extract bundle | No | Clone the winning program and task files into `_extracted_bundle/` |
| 1 | Bootstrap analysis-ready bundle | Yes | Ensure a usable checkpoint exists and record bootstrap guardrails |
| 2 | Generate analysis candidates | Yes | Propose artifact-based post-hoc analyses as JSON |
| 3 | Validate and select analyses | Yes | Accept/reject generated analyses and apply hard constraints |
| 4 | Materialize analysis workspaces | No | Create one workspace per accepted analysis |
| 5 | Execute analysis agents | Yes | Run one analysis agent per workspace with retry-on-inspection-failure |
| 6 | Aggregate and Publish Valid Analyses | No | Publish all valid analyses for downstream paper-agent selection |

### Stage 0: Extract bundle

Stage 0 reuses the best-program extraction flow and writes:

- `_extracted_bundle/program_0/`
- `_extracted_bundle/reference_program/`
- `_extracted_bundle/evaluate.py`
- `_extracted_bundle/capture.py` if present
- `_extracted_bundle/task.md`
- `_extracted_bundle/extraction_manifest.json`

This is the immutable source bundle for the rest of the run.

### Stage 1: Bootstrap analysis-ready bundle

Stage 1 is unique to the post-hoc pipeline. It creates
`_analysis_ready_bundle/`, validates the primary metric guardrail, and ensures a
usable checkpoint exists.

Behavior:

- If the extracted winner already has a valid checkpoint, bootstrap only copies
  and validates it.
- If no checkpoint exists and
  `execution.retrain_if_checkpoint_missing=true`, bootstrap may run a one-time
  repair/rerun flow to create one.
- Guardrails compare the current primary metric against the baseline extracted
  winner using `execution.primary_metric_key` or metric inference.

Key outputs:

- `_analysis_ready_bundle/`
- `bootstrap_report.json`
- `bootstrap_attempts/attempt_XX/bootstrap_agent_stdout.log`
- `bootstrap_attempts/attempt_XX/bootstrap_agent_stderr.log`
- `bootstrap_attempts/attempt_XX/bootstrap_evaluator_stdout.log`
- `bootstrap_attempts/attempt_XX/bootstrap_evaluator_stderr.log`

Important:

- `bootstrap_report.json` is the authoritative checkpoint/guardrail record.
- If bootstrap degrades and no later valid analyses are produced, the pipeline
  returns nonzero.

### Stage 2: Generate analysis candidates

Stage 2 launches the generator agent and writes:

- `generation/posthoc_generate_prompt.md`
- `generation/posthoc_bundle.json`
- `generation/change_summary.json`

`posthoc_bundle.json` uses this LLM-facing schema:

```json
{
  "decision": "propose_analyses",
  "summary": "short summary",
  "analyses": [
    {
      "id": "weight_distribution_analysis",
      "title": "Weight Distribution and Non-Negative Clamping Analysis",
      "family": "weight_introspection",
      "hypothesis": "string",
      "relevance_reason": "string",
      "feasibility_reason": "string",
      "required_artifacts": ["string"],
      "requires_retrain": false,
      "implementation_plan": ["string"],
      "expected_claim": "string",
      "expected_outputs": ["string"]
    }
  ]
}
```

Notes:

- `id` is the canonical candidate key for stage-2 and stage-3 JSON.
- The parser still accepts legacy `analysis_id` for backward compatibility.

### Stage 3: Validate and select analyses

Stage 3 reviews generated candidates and writes:

- `generation/posthoc_select_prompt.md`
- `generation/posthoc_selection.json`

Selector outputs may use a compact decision list:

```json
{
  "decisions": [
    {
      "id": "weight_distribution_analysis",
      "accepted": true,
      "reason": "grounded and paper-useful"
    }
  ]
}
```

The persisted selection file written by the pipeline contains:

- `accepted`: accepted candidate objects
- `rejected`: rejected candidate objects with `reason`
- `max_selected_analyses`

Hard constraints enforced in code:

- reject ablation-family candidates
- reject anything with `requires_retrain=true`
- reject plans that imply retraining, `evaluate.py`, or evaluator reruns
- reject missing-artifact proposals
- deduplicate by analysis id and family/title
- enforce `execution.max_selected_analyses`

### Stage 4: Materialize analysis workspaces

Stage 4 creates:

```text
analyses/<analysis_id>/
  analysis_spec.json
  rubric_contract.json
  target_rubrics.json
  run_manifest.json
  task_bundle/
    program_0/
    reference_program/
    task.md
    original_task.md
    analysis_spec.json
    rubric_contract.json
    target_rubrics.json
    ...
```

Notes:

- `analysis_spec.json` and later execution artifacts use `analysis_id`
  internally. The `id` form is only for the stage-2/3 LLM-facing JSON.
- `task_bundle/program_0` is the checkpoint-ready base bundle copied from stage
  1 and is treated as read-only by stage 5.
- stale unselected analysis dirs are removed when rematerializing.

### Stage 5: Execute analysis agents

Stage 5 runs one analysis at a time. Each analysis gets up to
`execution.max_attempts_per_analysis` retries. The next retry starts from the
previous attempt workspace but special files such as FIFOs are skipped.

Per-analysis outputs:

```text
analyses/<analysis_id>/
  analysis_result.json
  run_manifest.json
  attempt_1/
  attempt_2/
  ...
```

Each attempt directory contains:

- `posthoc_attempt_prompt.md`
- `analysis_spec.json`
- `rubric_contract.json`
- `target_rubrics.json`
- `task_bundle/`
- `analysis_summary.md`
- `interestingness_rubric.json`
- `analysis_manifest.json`
- `rubric_status.json`
- `attempt_result.json`
- any analysis-specific files declared in `expected_outputs`
- agent logs such as `posthoc_agent_<agent>_<timestamp>.log`

An attempt only passes if all of these are true:

- required root files exist
- `rubric_status.json` validates
- `analysis_manifest.json` validates
- all expected outputs are present
- a usable checkpoint is still available
- `analysis_manifest.reran_evaluator` is `false`

### Stage 6: Aggregate and Publish Valid Analyses

Stage 6 is a run-level publication step. It does not choose a local winner.

It writes:

- `ranking.json`
- `latest_manifest.json` if at least one valid analysis exists
- `pipeline_summary.json`
- stable pointer: `results_dir/posthoc/latest_manifest.json`

Current semantics:

- `ranking.json` is a compatibility file that lists `valid` and `rejected`
  analyses and sets `delegated_to_paper_agent=true`
- `latest_manifest.json` is the paper-agent handoff manifest containing all
  valid analyses
- `pipeline_summary.json` records run-level counts and final status

## Output layout

When `paths.output_dir` is unset, output defaults to:

```text
<results_dir>/posthoc/<task_name>_<agent_name>_<run_id>/
```

The run directory looks like:

```text
output_dir/
  _extracted_bundle/
  _analysis_ready_bundle/
  bootstrap_report.json
  bootstrap_attempts/
  generation/
    posthoc_generate_prompt.md
    posthoc_bundle.json
    change_summary.json
    posthoc_select_prompt.md
    posthoc_selection.json
  analyses/
    <analysis_id>/
      analysis_spec.json
      rubric_contract.json
      target_rubrics.json
      run_manifest.json
      analysis_result.json
      task_bundle/
      attempt_1/
      attempt_2/
      ...
  ranking.json
  latest_manifest.json
  pipeline_summary.json
```

## Most important artifacts

These are the files you usually inspect first:

- `bootstrap_report.json`
  - whether stage 1 succeeded
  - inferred or requested primary metric
  - checkpoint existence and selected checkpoint path
  - bootstrap retry history and persisted stdout/stderr paths
- `generation/posthoc_bundle.json`
  - raw generated candidate analyses
- `generation/posthoc_selection.json`
  - accepted and rejected analyses after hard constraints
- `analyses/<analysis_id>/run_manifest.json`
  - per-analysis lifecycle status
- `analyses/<analysis_id>/analysis_result.json`
  - final execution result, selected attempt, rubric/output pass state
- `analyses/<analysis_id>/attempt_N/analysis_manifest.json`
  - grounded evidence paths, checkpoint paths, declared outputs
- `latest_manifest.json`
  - all valid analyses published for downstream paper-agent selection
- `pipeline_summary.json`
  - final run-level status

## Configuration

The pipeline uses `configs/posthoc_config.yaml` with Hydra overrides.

Important keys:

```yaml
agent:
  name: claude
  model: null

paths:
  task_dir: ""
  results_dir: ""
  evidence_dir: null
  output_dir: null

execution:
  bootstrap_timeout_seconds: 3600
  bootstrap_max_attempts: 3
  generation_timeout_seconds: 1800
  selection_timeout_seconds: 1800
  analysis_timeout_seconds: 3600
  analysis_timeout_grace_seconds: 30
  max_selected_analyses: 3
  max_attempts_per_analysis: 3
  primary_metric_key: null
  primary_metric_tolerance: 0.01
  retrain_if_checkpoint_missing: true

container:
  enabled: false
  image: agentic-scientist/agent-cli:latest
  network: bridge
  auth_root: "~"
  env_file: ""
```

Examples:

- `agent.name=claude`
- `agent.model=claude-opus-5-5`
- `paths.evidence_dir=results/.../variants/<completed-run>`
- `container.enabled=true`
- `container.image=agentic-scientist/agent-cli:latest`
- `execution.max_selected_analyses=4`
- `execution.max_attempts_per_analysis=2`
- `execution.primary_metric_key=val_accuracy`

When `paths.evidence_dir` is set, stage 0 requires a completed variants
`pipeline_summary.json` and copies only the root summary/README, generated variant
proposal JSON, and the fixed aggregate CSV filenames. Files are size-bounded,
hash-manifested, and staged read-only; per-variant run manifests and arbitrary files
are excluded.

## Running the pipeline

### Full run

```bash
python -m src.posthoc_pipeline.cli \
    paths.task_dir=examples/circpack \
    paths.results_dir=results/circpack/claude/20260331_094917 \
    agent.name=claude \
    agent.model=claude-opus-5-5 \
    container.enabled=false
```

### Full run in container mode

```bash
python -m src.posthoc_pipeline.cli \
    paths.task_dir=examples/circpack \
    paths.results_dir=results/circpack/claude/20260331_094917 \
    agent.name=claude \
    agent.model=claude-opus-5-5 \
    container.enabled=true \
    container.image=agentic-scientist/agent-cli:latest \
    container.network=host \
    container.auth_root=$HOME \
    container.env_file=.env
```

Stage numbers:

- `0 = extract_bundle`
- `1 = bootstrap_analysis_bundle`
- `2 = generate_analysis_candidates`
- `3 = validate_and_select_analyses`
- `4 = materialize_analysis_workspaces`
- `5 = execute_analysis_agents`
- `6 = aggregate_and_publish_valid_analyses`

## Resuming a failed run

To resume, pass the original `paths.output_dir` and rerun only the needed
stages.

### Resume from stage 2 after bootstrap succeeded

Use this when `_analysis_ready_bundle/` and `bootstrap_report.json` already
exist and you want to regenerate candidates onward:

```bash
python -m src.posthoc_pipeline.cli --stages 2,3,4,5,6 \
    paths.task_dir=examples/circpack \
    paths.results_dir=results/circpack/claude/20260331_094917 \
    paths.output_dir=results/circpack/claude/20260331_094917/posthoc/circpack_claude_20260407_043818 \
    agent.name=claude \
    agent.model=claude-opus-5-5 \
    container.enabled=false
```

### Resume execution only

Use this when selection and workspaces are already correct and only execution
needs to be retried:

```bash
python -m src.posthoc_pipeline.cli --stages 5,6 \
    paths.task_dir=examples/circpack \
    paths.results_dir=results/circpack/claude/20260331_094917 \
    paths.output_dir=results/circpack/claude/20260331_094917/posthoc/circpack_claude_20260407_043818 \
    agent.name=claude \
    agent.model=claude-opus-5-5 \
    container.enabled=false
```

### Republish only

Use this when execution results already exist and you only changed stage-6
publication logic:

```bash
python -m src.posthoc_pipeline.cli --stages 6 \
    paths.task_dir=examples/circpack \
    paths.results_dir=results/circpack/claude/20260331_094917 \
    paths.output_dir=results/circpack/claude/20260331_094917/posthoc/circpack_claude_20260407_043818 \
    agent.name=claude \
    agent.model=claude-opus-5-5 \
    container.enabled=false
```

### Rerun bootstrap

Use this if stage 1 failed and you want to reuse the existing extracted bundle
but try bootstrap again:

```bash
python -m src.posthoc_pipeline.cli --stages 1,2,3,4,5,6 \
    paths.task_dir=examples/circpack \
    paths.results_dir=results/circpack/claude/20260331_094917 \
    paths.output_dir=results/circpack/claude/20260331_094917/posthoc/circpack_claude_20260407_043818 \
    agent.name=claude \
    agent.model=claude-opus-5-5 \
    container.enabled=false
```

## Failure triage

When a run fails in the middle, check these first:

- bootstrap issues:
  - `bootstrap_report.json`
  - `bootstrap_attempts/attempt_XX/bootstrap_agent_stdout.log`
  - `bootstrap_attempts/attempt_XX/bootstrap_agent_stderr.log`
- generation issues:
  - `generation/posthoc_bundle.json`
  - `generation/change_summary.json`
- selection issues:
  - `generation/posthoc_selection.json`
- analysis execution issues:
  - `analyses/<analysis_id>/run_manifest.json`
  - `analyses/<analysis_id>/analysis_result.json`
  - `analyses/<analysis_id>/attempt_N/attempt_result.json`
  - `analyses/<analysis_id>/attempt_N/posthoc_agent_<...>.log`
- run-level status:
  - `pipeline_summary.json`

## Relationship to the paper pipeline

The post-hoc pipeline currently publishes valid analyses for downstream
consumption. It does not directly write the paper or choose the final narrative.
That final selection is delegated to the paper agent via `latest_manifest.json`.
