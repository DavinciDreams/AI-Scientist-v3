# Agent billing fallback

Headless owns billing selection for native Codex and Claude. `HEADLESS_BILLING=auto`
(the default) starts with subscription credentials when available and can retry an
eligible subscription quota or model-access failure once using paid credentials.
`HEADLESS_BILLING=subscription` disables paid fallback; `HEADLESS_BILLING=api` selects
the paid route directly. Other harnesses retain their native provider routing.

For Codex, provide `CODEX_API_KEY` or `OPENAI_API_KEY` in the repository `.env`.
An exported `CODEX_API_KEY` is also forwarded to Codex containers by variable name,
without placing its value in command arguments. For Claude, the paid route uses
Bedrock credentials and region from the existing provider environment allowlist.
Claude's native Bedrock mapping retains the requested model; unavailable mappings
fail rather than selecting another model. `HEADLESS_BILLING` may be exported or set in
the provider `.env`; an exported setting takes precedence for normal agent stages.

The `AGENT_CHATGPT_SUBSCRIPTION=1` bridge remains subscription-only: it removes
paid credentials, including `CODEX_API_KEY`, and forces subscription billing for
the bridged Pi/OpenCode command. Do not use that bridge to enable paid
fallback.

Headless exit status 78 is terminal for Claude/Codex billing failures. Task drafting
stops the enclosing workflow before its transient-provider or scaffold repair
retries. This decision uses the agent process exit status; agent log text and
`billing` metadata never authorize a billing change or a stop. Investigation already
stops after the configured number of failed idea-generation attempts.

The cost ledger retains a bounded `billing.attempts` array (at most five entries)
with route, optional fixed billing reason, optional `retryReason: "model-capacity"`,
and known usage/provenance fields. Unknown fields
are dropped; malformed billing metadata is omitted without discarding valid
aggregate usage. Each validated attempt receives stable ledger identities within its
invocation; reconciled Claude model breakdowns remain separate. The combined
aggregate is omitted to avoid double counting.
Known costs and tokens remain counted when another attempt has missing usage, and
unknown attempts remain marked unknown. Native reported costs and API price estimates
retain separate provenance and summary totals. Reported or estimated
API-equivalent cost is not a receipt for a subscription charge. This metadata is
operational telemetry, not an authenticated billing or spending-limit system.

Focused checks: `tests/investigation/test_agent_billing.py`, `tests/investigation/test_cost_tracking.py`,
and `tests/runtime/test_container_auth.py`.
Use a rebuilt image containing the matching Headless implementation before running
experiments; changing a sibling Headless checkout does not update pinned images.

## Capacity retries

Headless owns retries for Codex's `Selected model is at capacity` failure: three
retries by default, waiting 30/60/120 seconds with jitter inside the original
invocation deadline. Its one billing fallback has an independent budget, allowing
at most five native executions. Task drafting does not repeat the capacity retry
loop after Headless returns a failure; other transient provider failures retain
the existing drafting retry policy. Task-level artifact and repair budgets remain
separate.

Set `HEADLESS_CAPACITY_RETRIES=0` to disable capacity retries, or `1`, `2`, or `3`
to choose a limit. Codex host settings and provider environment files transport
this override through local, container, and concept-figure invocations; an explicit
host value takes precedence. Invalid values fail before agent launch. This
operational setting does not authorize paid credentials in subscription mode.

Capacity regression checks: `tests/task_drafting/test_task_drafting_publication.py`,
`tests/e2e/test_agent_execution_pipeline.py -k logged_agent_step`, and
`tests/investigation/test_cost_tracking.py`.
