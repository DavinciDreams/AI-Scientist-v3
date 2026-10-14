# End-to-End Pipeline

`agsci run` chains the six default stages, task drafting → investigation → ablations →
final evaluation → post-hoc analysis → paper writing, with a single command. Add the
explicit `review` phase to send the completed manuscript to the OpenAI-backed final
reviewer. Every inter-stage directory is derived from one shared `run_id`, so you never
hand-copy paths between stages.

`agsci run` is the pipeline subcommand of the console script installed by `uv sync`; a
leading bare argument is the task directory (shorthand for `task.dir=<path>`), and
`python -m src.pipeline.cli` accepts the same arguments. Check the host, credentials,
image, and GPUs for a resolved run first with `agsci doctor <task_dir> preset=<task>`.

## Commands

```bash
# Reproduce a campaign run from its checked-in task preset:
agsci run examples/grpo preset=grpo

# Full pipeline (task drafting -> investigation -> ablations -> final evaluation -> post-hoc -> paper):
agsci run examples/circpack
# equivalent long form:
python -m src.pipeline.cli task.dir=examples/circpack

# Full pipeline plus external final-manuscript review:
agsci run examples/circpack \
  agent.name=claude agent.model=claude-sonnet-5 agent.reasoning_effort=high \
  paper.paper_format=iclr \
  execution.phases=task_drafting,investigation,ablations,final_evaluation,posthoc,paper,review

# Draft and validate a missing task scaffold, then stop before ideation:
agsci run examples/circpack_task_drafting execution.phases=task_drafting

# Preview the exact commands without running anything:
agsci run examples/circpack --dry-run

# Run only paper writing from a completed investigation:
agsci run examples/circpack execution.phases=paper \
  paths.run_dir=results/circpack/claude/20260331_094917

# Resume with ablations, post-hoc analysis, and paper writing after investigation:
agsci run examples/circpack execution.phases=ablations,posthoc,paper,review \
  paths.run_dir=results/circpack/claude/20260331_094917

# Retry only the non-gating final review for an existing published paper:
agsci run execution.phases=review \
  paths.run_dir=results/circpack/claude/20260331_094917
```

Resume any phase against the shared `paths.run_dir` anchor; `paths.ablations_dir`,
`paths.posthoc_dir`, `paths.final_evaluation_dir`, and `paths.paper_dir` are optional advanced
overrides. The orchestrator validates the anchor and prints a named-phase resume recipe
on failure. Run `agsci run --help` for every key, type, default, and description;
`src/pipeline/config.py` is the schema and `configs/pipeline_config.yaml` only states
values that differ from a stage's own default.

Selecting `review` is explicit consent to send the complete final manuscript to OpenAI.
Review failures are recorded as degraded but do not fail the pipeline or remove the
published paper.
The same reviewer is available standalone as `agsci review <paper.pdf> [--output review.json]`
(it reads `OPENAI_API_KEY` from the environment or the repo `.env` and requires `uv`).

## Campaign task presets

The experiment settings used for the paper campaign (phases, idea and attempt
budgets, timeouts, paper format, and paper eligibility policy) live in checked-in task
presets under `configs/preset/`. `configs/preset/paper.yaml` holds the shared campaign
settings and `configs/preset/<task>.yaml` composes it for one task, stating only what
differs for that task (for example its GPU allocation). The repository ships the preset
for the GRPO example, `configs/preset/grpo.yaml`; add a `configs/preset/<task>.yaml`
next to it for your own `examples/<task>` (every task preset must have a matching
`examples/<task>/task.md`; `agsci doctor` checks this). Reproduce a campaign run with
`preset=<task>` plus run-time values only (harness, model, idea budget, image):

```bash
agsci run examples/grpo preset=grpo
```

`preset=<name>` applies a checked-in task preset (`paper` is the shared campaign
preset; one file per task composes it), and explicit `key=value` overrides still win
over the preset. Without a preset the pipeline keeps the strict
`paper.final_outcome_policy=qualified_only` default; the campaign presets set
`allow_not_qualified`.

Override the harness, model, or cycle budget as needed:

```bash
agsci run examples/grpo preset=grpo \
  agent.name=opencode \
  agent.model=amazon-bedrock/global.anthropic.claude-opus-5-5 \
  investigation.execution.max_idea_cycles=20
```

GPU task presets enable GPU-backed baseline and candidate harness evaluation
(`investigation.execution.gpu_evaluation`). `execution.phases` and
`paper.final_outcome_policy` override the preset only when given; the pipeline
drops task drafting on resume when phases are not explicit.

Drafted tasks create their own task harness. This measures full-run variation,
but generated evaluators can differ between model rows.

### GPU allocation

Each task preset declares the GPUs the task requires as `container.gpu_count`:
the GRPO preset needs two (one for vLLM sampling, one for training); a CPU-only task
declares zero. Before compute
phases the pipeline asks `nvidia-smi` for GPU UUIDs and uses the first
`container.gpu_count` of them; set `AGENT_CONTAINER_GPUS` to a `device=<UUID,...>`
selector naming exactly that many GPUs to choose them yourself. The pipeline refuses
to start compute phases unless the visible GPUs match the preset exactly.
With `investigation.execution.workers=N` the run needs `N x container.gpu_count`
GPUs: discovery takes that many, an explicit selector must name exactly that many,
and each investigation worker receives its own `container.gpu_count` slice. The
pool is exported as `AGENT_CONTAINER_GPU_POOL`; every other phase uses the first
slice.
`container.gpu_count=<n>` overrides the preset for one run; `container.gpu_count=0`
together with `investigation.execution.gpu_evaluation=false` turns harness GPU
evaluation off.

A task draft that requires more GPUs than its preset declares needs the matching
`container.gpu_count=<n>` and that many visible GPUs; changing the preset does not
resize saved contracts.

### Resuming a run

Resume an interrupted run through its run directory. Pass the original harness and
model again, with a matching GPU allocation; the pipeline recovers the original
idea-cycle budget from the run unless `investigation.execution.max_idea_cycles` is
explicit:

```bash
agsci run examples/grpo preset=grpo \
  agent.name=codex agent.model=gpt-5.6-sol \
  paths.run_dir=results/grpo/codex/<run_id>
```

Export `AGENT_CHATGPT_SUBSCRIPTION=1` for a Pi/OpenCode GPT-5.6-Sol subscription run;
it removes paid credentials from the agent containers and uses the existing Codex
ChatGPT login (`codex login`). The final AI review remains a separate API-key-backed
pipeline phase.

Fresh runs write to `results/` directly; set `paths.results_root` to change the root.
Task snapshots use native atomic no-replace publication when the filesystem supports
it and a descriptor-safe, resumable publication protocol otherwise.

## Phase configuration

Each phase composes the downstream stage's own config, so every stage key is
addressable as `<phase>.<stage key>` and only values that differ from the stage's
default are forwarded:

```bash
agsci run examples/circpack \
  investigation.execution.max_idea_cycles=5 \
  investigation.execution.isolate_worker_memory=true \
  investigation.task_drafting.design_max_repairs=2 \
  ablations.execution.num_replications=1 ablations.execution.stages='0-3' \
  posthoc.execution.max_selected_analyses=2 \
  paper.paper_format=iclr paper.paper_review_rounds=1
```

Keys the pipeline derives itself (paths, run id, agent, container, logging,
versioning) are rejected under a phase with a message naming the pipeline key to
use instead. Unknown keys, including the pre-composition spellings such as
`investigation.max_idea_cycles` or `paper.format`, fail at load and name the new key.

## Run layout

Each run has one predictable envelope:

```text
results/<task>/<agent>/<run_id>/
├── task_drafting/ # drafting logs and validation audit
├── task/
├── investigation/
├── ablations/
├── final_evaluation/
├── posthoc/
└── paper/
```

Investigation plus the run-local task snapshot is sufficient for paper generation.
Ablations, final evaluation, post-hoc analysis, and final review add evidence when
available; their failures are reported as degraded and do not discard a publishable
paper. Contracted tasks that explicitly require final evaluation still enforce it.

`task.dir` is an immutable source for end-to-end runs. At startup, the pipeline
copies the complete task into `<run_dir>/task` (honoring
`.agentic_task_ignore`) and uses that run-local snapshot for task drafting,
investigation, ablations, and post-hoc analysis. Generated scaffold files therefore stay
with the run instead of being written back into `examples/`. Resumes reuse the existing
snapshot without refreshing it from the source.

Legacy investigation runs without a stored snapshot bootstrap it once from an
explicit `task.dir=<original task source>` override; dry runs validate and
preview that migration without writing files.

## Ablations phase (variants pipeline)

Before ablation agents start, the variants phase automatically provisions and
validates `<run_dir>/task/.venv` in the pinned evaluation
image. The matching `.agentic_runtime.json` binds that environment to the
image, evaluator, requirements, and runtime tooling. A missing runtime is
created; a compatible one is reused together with its validated baseline, so
only the first run pays for the baseline evaluation. Unsafe symlinks, unmanaged environments,
unpinned requirements, or a baseline that fails sealed validation stop the
phase before variant artifacts or LLM calls are created. After correcting the
reported task/runtime issue, rerun the printed
`execution.phases=ablations,posthoc,paper,review` resume command against the same
investigation results directory.

Run baselines and ablations against a completed scientist run as the `ablations`
phase; every variants key is reachable as `ablations.<key>`:

```bash
# Full pipeline (9 stages: extract → retrofit → generate → select → materialize
# → execute → cross-task → aggregate → README)
agsci run examples/circpack execution.phases=ablations \
  paths.run_dir=results/circpack/claude/20260331_100000

# Run specific stages only (0-8 or comma-separated subsets)
agsci run examples/circpack execution.phases=ablations \
  paths.run_dir=results/circpack/claude/20260331_100000 \
  ablations.execution.stages='0,1'

# Resume from stage 4 (using existing artifacts)
agsci run examples/circpack execution.phases=ablations \
  paths.run_dir=results/circpack/claude/20260331_100000 \
  paths.ablations_dir=results/.../variants/circpack_claude_20260331_100000 \
  ablations.execution.stages='4-8'
```

See `src/variants_pipeline/README.md` for the full stage reference, the standalone
`python -m src.variants_pipeline.cli` entry point, custom baselines, cross-task
generalization, and output layout. `configs/variants_config.yaml` lists every
variants option.

## Post-hoc analysis phase

Run post-hoc analyses against a completed scientist run as the `posthoc` phase;
every post-hoc key is reachable as `posthoc.<key>`:

```bash
# Full pipeline
# (0=extract, 1=bootstrap, 2=generate, 3=select, 4=materialize, 5=execute,
#  6=aggregate_and_publish_valid_analyses)
agsci run examples/circpack execution.phases=posthoc \
  paths.run_dir=results/circpack/claude/20260331_100000

# Rerun against an existing run-scoped output dir
agsci run examples/circpack execution.phases=posthoc \
  paths.run_dir=results/circpack/claude/20260331_100000 \
  paths.posthoc_dir=results/.../posthoc/circpack_claude_20260331_100000 \
  posthoc.execution.max_selected_analyses=2
```

Stage subsets (`--stages 0,1`) are a standalone-CLI feature:
`python -m src.posthoc_pipeline.cli --stages 5,6 paths.task_dir=... paths.results_dir=...`;
see [posthoc_pipeline.md](posthoc_pipeline.md).

The post-hoc pipeline writes to a run-scoped directory under `results_dir/posthoc/`
and publishes a success-only stable pointer at `results_dir/posthoc/latest_manifest.json`
containing all valid analyses for downstream paper-agent selection.

Integrated post-hoc analysis is artifact-only: it uses the completed investigation
outputs without repairing or retraining a missing checkpoint. The integrated flow also
stages a bounded, hash-manifested scientific allowlist from the completed ablations bundle
(`pipeline_summary.json`, variant proposals, and aggregate CSVs). It is best-effort;
failure or an empty valid-analysis set is recorded in `integration_status.json`
and paper writing continues without `--posthoc-dir`. Valid analyses are passed to
paper writing explicitly before drafting begins. Each integrated run uses a fresh
owner-controlled `<run_dir>/posthoc`; provider credentials
are agent/model allowlisted, and its container network defaults to `bridge` (override
`posthoc.container.network=host` only when the selected agent requires it).

## Paper phase outside the orchestrator

To run only the paper stage outside the orchestrator, use the generic
`scripts/write_paper_variants.sh` (auto-detects the task name from
`aggregated/<task>/`):

```bash
SRC_RESULTS_DIR=<variants_output_dir> ./scripts/write_paper_variants.sh
```

## Reading progress and output

The default terminal view follows the scientific cycle: current cycle and stage,
idea title, attempt score and best-score delta, correctness, every rubric check,
and the final outcome. Long-running steps update in place in an interactive
terminal with a restrained spinner and semantic colors; redirected output
receives a durable heartbeat every 60 seconds. Set `NO_COLOR=1` to disable ANSI
color without changing the content. Completed stages also print the output
tokens of API/Headless calls, as a stage delta and a running total; dollar
figures stay out of the terminal because subscription agents self-report prices
nobody is billed for. Accounting in `metrics/cost_events.jsonl`
and `metrics/cost_summary.json` distinguishes provider-reported charges from
Headless API-list-price estimates; both contribute to `total_cost_usd`.
Unpriced usage remains explicit. Hardware costs are not included. Treat these
files as run telemetry and use provider invoices for tamper-resistant billing
reconciliation.

Use JSON lines for automation, or human verbose mode for diagnostics:

```bash
agsci investigate --log-format=json
agsci investigate --verbose
```

Equivalent Hydra overrides are `logging.format=json` and
`logging.verbose=true`. Every terminal mode also writes the complete structured
event stream to `<results_dir>/events.jsonl`; parallel workers write their own
journals beneath their worker result directories.

The same controls apply to the end-to-end orchestrator, standalone variants,
post-hoc analysis, and paper writing:

```bash
agsci run examples/circpack logging.format=human
agsci run execution.phases=ablations paths.run_dir=results/... logging.verbose=true
agsci run execution.phases=posthoc paths.run_dir=results/... logging.heartbeat_seconds=30
agsci run execution.phases=paper paths.run_dir=results/... logging.format=json
```

The parent orchestrator writes `pipeline_events.jsonl` in the first selected
phase's existing run root. Investigation, variants, and post-hoc runs write their
own `events.jsonl`; paper writing writes a private paper-event journal. Default
human output shows phase/gate state, idea or analysis names, scores, and rubric
checks. Standard-library warnings use the same timestamp, severity, and module
prefix in the parent pipeline and child investigation CLI. Checkpoint exclusion notices
instead use the concise `[CHECKPOINT STORED]` label. Completion output reports
total wall-clock runtime as hours, minutes, and seconds; `pipeline_end` also
records the numeric `elapsed_seconds` value in `pipeline_events.jsonl`. Raw
diagnostics and paths stay in private journals unless `--verbose` is selected.
