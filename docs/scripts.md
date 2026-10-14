# Script Reference

## Common agent options

Agent-facing scripts accept `--agent-model <name>`, the Codex-only
`--agent-profile <name>`, and `--reasoning-effort low|medium|high|xhigh`.
Reasoning effort applies to every agent pass started by the script, including
paper review/repair passes and Autoillustrate concept-figure generation.

## `src/scripts/task.sh`

Drafts/updates task scaffold from `task.md`:
- `capture.py`
- `evaluate.py`
- `program_0/program_spec.toml`
- `program_0/src/program.py`

Basic:

```bash
bash src/scripts/task.sh \
  --agent claude \
  --task-dir examples/circpack \
  --output-dir /tmp/task_draft_run
```

Custom prompt:

```bash
bash src/scripts/task.sh \
  --agent codex \
  --task-dir /path/to/task \
  --instruct src/prompts/draft_task.md \
  --output-dir /tmp/task_draft_codex
```

Main args: `--agent`, `--model`, `--agent-profile`, `--reasoning-effort`, `--task-dir`, `--output-dir`, `--instruct`.

## `src/scripts/idea.sh`

Generates one idea bundle:
- `idea_bundle.json`
- `target_rubrics.json`

Basic:

```bash
bash src/scripts/idea.sh \
  --agent claude \
  --task-dir examples/circpack \
  --results-dir results/circpack/claude \
  --cycle-index 1
```

Custom output/prompt:

```bash
bash src/scripts/idea.sh \
  --agent codex \
  --task-dir examples/circpack \
  --results-dir results/circpack/codex \
  --cycle-index 3 \
  --output-dir /tmp/idea_cycle_3 \
  --instruct src/prompts/idea.md
```

Main args: `--agent`, `--model`, `--agent-profile`, `--reasoning-effort`, `--task-dir`, `--results-dir`, `--cycle-index`, `--output-dir`, `--instruct`.

## `src/scripts/run.sh`

Runs one idea execution under `results/.../ideas/<idea>/`.

Basic:

```bash
bash src/scripts/run.sh \
  --agent claude \
  --task-dir examples/circpack \
  --results-dir results/circpack/claude \
  --idea-file path/to/idea_bundle.json \
  --rubrics-file path/to/target_rubrics.json \
  --attempt-id idea_0001_attempt_01 \
  --max-program-evals-per-attempt 4
```

Custom seed/deterministic names:

```bash
bash src/scripts/run.sh \
  --agent codex \
  --task-dir examples/circpack \
  --results-dir /tmp/circpack_run \
  --seed-program examples/circpack/program_0 \
  --idea-file examples/circpack/idea_example.json \
  --rubrics-file examples/circpack/rubrics_example.json \
  --attempt-id circpack_attempt_01 \
  --idea-dir-name 01_circpack_idea \
  --max-program-evals-per-attempt 2
```

Main args:
- `--agent`
- `--task-dir`
- `--results-dir`
- `--instruct`
- `--seed-program`
- `--idea-file`
- `--rubrics-file`
- `--attempt-id`
- `--idea-dir-name`
- `--max-program-evals-per-attempt`

Manifest retrieval:

```bash
cat results/circpack/codex/latest_run_manifest.json
```

Inspect latest evaluated artifacts:

```bash
EXP_DIR="$(python - <<'PY'
import json
print(json.load(open('results/circpack/codex/latest_run_manifest.json'))['experiment_dir'])
PY
)"
find "$EXP_DIR" -maxdepth 2 -name 'metrics.json' -o -name 'correct.json' -o -name 'rubric_status.json'
```

## `src/scripts/paper.sh`

Generates final paper artifacts.

Basic:

```bash
bash src/scripts/paper.sh \
  --agent claude \
  --task-dir examples/circpack \
  --results-dir results/circpack/claude
```

Custom output/citation bounds/prompt:

```bash
bash src/scripts/paper.sh \
  --agent codex \
  --task-dir examples/circpack \
  --results-dir results/circpack/codex \
  --output-dir /tmp/final_paper_codex \
  --paper-min-refs 12 \
  --paper-max-refs 20 \
  --instruct src/prompts/paper.md
```

Standalone-paper profile example:

```bash
bash src/scripts/paper.sh \
  --agent claude \
  --task-dir examples/circpack \
  --results-dir results/circpack/claude \
  --paper-profile standalone
```

Standalone-paper convenience flag example:

```bash
bash src/scripts/paper.sh \
  --agent claude \
  --task-dir examples/circpack \
  --results-dir results/circpack/claude \
  --standalone-paper
```

Equivalent explicit standalone skill selection example:

```bash
bash src/scripts/paper.sh \
  --agent claude \
  --task-dir examples/circpack \
  --results-dir results/circpack/claude \
  --instruct src/prompts/paper_standalone.md \
  --skills retrieve-references,write-paper-standalone,validate-paper,memory-log
```

Two-stage standalone paper example:

```bash
bash src/scripts/paper.sh \
  --agent claude \
  --task-dir examples/circpack \
  --results-dir results/circpack/claude \
  --instruct src/prompts/paper_standalone.md \
  --skills retrieve-references,write-paper-standalone,validate-paper,memory-log \
  --two-stage
```

Notes:
- `--paper-profile default|standalone` selects the paper workflow profile.
- `--paper-format <venue>` selects a venue declared under `configs/venue/` (default `iclr`).
- `--standalone-paper` is shorthand for `--paper-profile standalone --two-stage`.
- In standalone profile, `paper.sh` defaults to `src/prompts/paper_standalone.md` and the standalone paper skill set.
- `--two-stage` runs a framing pass first, which must write `paper_strategy.md` and `paper_strategy.json`, then runs the final paper-writing pass.
- After the drafting pass, `paper.sh` runs validation and automatically re-enters a repair loop when hard validation errors remain. Each repair round writes `validation_feedback.json` and `validation_feedback.md` into the paper workspace.
- If `--paper-review-rounds <n>` is greater than zero, `paper.sh` also runs up to `<n>` review/repair cycles after each drafting/repair pass, even when validation feedback is still unresolved. Each review writes `paper_review.json`, `paper_review.md`, `paper_review_feedback.json`, and `paper_review_feedback.md`.
- Review feedback is now split into `major_comments` and `minor_comments`, not blocking/non-blocking issues.
- After every review cycle, the manuscript always re-enters a text-only repair pass. The writer must address major comments seriously, may apply minor comments selectively, and must not run new experiments or invent new quantitative results during review repair.
- Standalone validation also rejects visible pipeline language such as `eligible`, `correctness rate`, `worker_01`, `worker_02`, and similar internal terms.
- Every host-side validation/compile first inserts (or regenerates) a mandatory watermark block in `paper.tex` reading "Generated by The Agentic Scientist" on every page. It is owned by `src/orchestrator/paper_watermark.py`, has no CLI or config switch, applies to `paper.sh`, `paper_edit.sh`, and resume finalization alike. Its settings are applied from the `begindocument/end` hook, so they take precedence over any other `draftwatermark` setup in the manuscript. Skill staging additionally watermarks the staged writer-skill template `paper.tex` files (`watermark_staged_templates`), so agent drafts and intermediate compiles carry the block from the first draft; the checked-in templates remain free of generated code.
- `--paper-max-repair-rounds` bounds total manuscript repair attempts. Default: `10`.
- `--paper-review-rounds` bounds planned review/repair cycles for the manuscript. Default: `0`.
- Before each review-driven manuscript repair pass, `paper.sh` copies the current `paper.pdf` into `paper/final_<timestamp>/revisions/` so the draft history stays inspectable over time.
- Set `paper.ai_review=openai` on an `agsci investigate` run, or append `review` to `execution.phases` for the end-to-end orchestrator, to send the validated final `paper.pdf` to the standalone OpenAI-backed `src/aireview.py` reviewer. Both forms require explicit opt-in. This final assessment is separate from the optional review/repair cycles above and writes `ai_review.json` plus `ai_review.log` in the final paper directory.
- `agsci review <paper.pdf> [--output review.json]` runs the same reviewer standalone on any PDF, outside a pipeline run and without manifest bookkeeping; it reads `OPENAI_API_KEY` from the environment or the repo `.env` (`--env-file` overrides it) and needs `uv` on `PATH`.
- Final AI review failures are non-gating: the paper run remains successful and `paper/latest_manifest.json` records `ai_review.status: failed`. Successful reviews record `completed` and the absolute review/log paths.
- The reviewer receives a restricted host environment. When the host key is absent, only OpenAI settings are loaded from the configured container env file or repo `.env`; unrelated credentials and general interpreter controls are excluded, while ambient uv resolver settings are preserved.
- `src/prompts/paper_strategy.md` is the default stage-1 framing prompt.
ALife writeup example command:
```bash
bash src/scripts/paper.sh \
--agent claude \
--task-dir examples/circpack \
--results-dir results/circpack/claude/<run_id> \
--containerize --paper-format alife --instruct src/prompts/paper_standalone.md
```

The venue brief defaults to the `brief` declared in `configs/venue/<paper-format>.yaml`. Pass `--venue-file <path>` to override it, or `--venue-file ""` to write without one.

Each paper format is one venue file under `configs/venue/`: its brief, canonical author block (`null` for anonymous venues), template directory inside the writer skills, the LaTeX markers that identify a compiled manuscript, the enforced page budget (`null` keeps the brief's budget advisory), and the behavioural checks it opts into (`alife_framing`). Briefs state an enforced budget through the `{{PAPER_PAGE_BUDGET}}` placeholder, which the paper scripts render from the YAML when staging `VENUE.md`. To add a venue, add `configs/venue/<name>.yaml` plus the brief and the `templates/<name>/` directory it references under `src/agents/skills/write-paper/` (the other writer skills receive that tree at staging time); the format list, brief lookup, page limits, format detection, and the scripts' `--paper-format` check all read the catalog.

Main args: `--agent`, `--model`, `--agent-profile`, `--reasoning-effort`, `--task-dir`, `--results-dir`, `--output-dir`, `--instruct`, `--venue-file`, `--skills`, `--paper-min-refs`, `--paper-max-refs`, `--paper-profile`, `--standalone-paper`, `--paper-format <venue>`, `--paper-max-repair-rounds`, `--paper-review-rounds`, `--two-stage`.

`paper_edit.sh` also supports the pipeline-managed `--external-review <json>` mode. It stages a
bounded review as untrusted feedback, preserves references, skips literature enrichment, and limits
the editor to changes grounded in existing artifacts. `--manifest-dir` publishes the validated
edit manifest separately from the staging directory; `--skip-postdraft-reference-enrichment`
explicitly disables the normal post-draft retrieval pass.

## Output layout

Default root: `results/<task-name>/<agent>`.

Core files:
- `outer_state.json`
- `ideas_history.jsonl`
- `attempts_history.jsonl`
- `best_program_manifest.json`
- `memory.md`
- `memory.jsonl`

Parallel files (`execution.workers>1`):
- `parallel_summary.json`
- `worker_XX/parallel_worker.log`
- `worker_XX/outer_state.json`
- `worker_XX/ideas_history.jsonl`
- `worker_XX/attempts_history.jsonl`

Per-idea execution files:
- `ideas/<NN_idea-name>/program_<idx>/rubric_status.json`
- `ideas/<NN_idea-name>/rubric_status.json`
- `ideas/<NN_idea-name>/agent_<agent>_<timestamp>.log`

Paper files:
- `paper/final_<timestamp>/paper.tex`
- `paper/final_<timestamp>/references.bib`
- `paper/final_<timestamp>/paper.pdf`
- `paper/final_<timestamp>/paper_data.json`
- `paper/final_<timestamp>/ai_review.json` (successful review)
- `paper/final_<timestamp>/ai_review.log`
- `paper/final_<timestamp>/references_provenance.json`
- `paper/final_<timestamp>/claims_grounding.json`
- `paper/final_<timestamp>/revisions/review_round_<NN>_before_repair_round_<NN>.pdf`
- `paper/final_<timestamp>/figures/*.png`
- `paper/latest_manifest.json` (stable pointer used by the run inspector and downstream tools)

## Offline tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
