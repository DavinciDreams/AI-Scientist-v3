import { useCallback, useEffect, useMemo, useState } from "react";
import { forceRefresh } from "../api/client";
import { fetchStudies, isActive, paperUrl, type ControlConfig, type Job, type Study } from "../api/control";
import { formatDuration, formatRelative, href, phaseLabel } from "../router";

interface StudiesViewProps {
  jobs: Job[];
  config: ControlConfig | null;
}

const POLL_MS = 10000;

/** The job that launched or last continued a study, if this server ran it. */
function jobForStudy(study: Study, jobs: Job[]): Job | undefined {
  return jobs.find(
    (job) =>
      job.resume_study_id === study.study_id ||
      (job.run_id === study.run_id && job.task_name === study.task_name),
  );
}

export function displayStatus(study: Study, job: Job | undefined): string {
  if (study.status !== "running") return study.status;
  // The journal has no pipeline_end yet: live if our job is still going.
  if (job) return isActive(job.status) ? "running" : job.status;
  return "in progress";
}

export function StatusBadge({ status }: { status: string }) {
  return <span className={`status-badge status-${status.replace(/[^a-z_]/g, "_")}`}>{status.replace(/_/g, " ")}</span>;
}

export function StudiesView({ jobs, config }: StudiesViewProps) {
  const [studies, setStudies] = useState<Study[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    try {
      setStudies(await fetchStudies());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    void load();
    const timer = window.setInterval(() => void load(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [load]);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!studies || !needle) return studies ?? [];
    return studies.filter((study) =>
      [study.task_name, study.agent_dir, study.run_id, study.model ?? "", study.status]
        .join(" ")
        .toLowerCase()
        .includes(needle),
    );
  }, [studies, query]);

  const counts = useMemo(() => {
    const all = studies ?? [];
    return {
      total: all.length,
      papers: all.filter((s) => s.has_paper).length,
      live: all.filter((s) => displayStatus(s, jobForStudy(s, jobs)) === "running").length,
    };
  }, [studies, jobs]);

  return (
    <section className="view-body">
      <header className="view-header">
        <div>
          <h1>Runs</h1>
          <p className="subtle">
            {counts.total} run{counts.total === 1 ? "" : "s"} · {counts.live} running · {counts.papers} paper
            {counts.papers === 1 ? "" : "s"}
          </p>
        </div>
        <div className="view-actions">
          <input
            className="search-input"
            type="search"
            placeholder="Filter by task, agent, status…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            aria-label="Filter runs"
          />
          <button
            className="button secondary"
            disabled={refreshing}
            onClick={() => {
              setRefreshing(true);
              // Rescan the inspector's run index too, so new investigations link up.
              void forceRefresh()
                .catch(() => undefined)
                .then(load)
                .finally(() => setRefreshing(false));
            }}
          >
            {refreshing ? "Refreshing…" : "Refresh"}
          </button>
          {config?.launch_enabled ? (
            <a className="button primary" href={href("launch")}>
              New run
            </a>
          ) : null}
        </div>
      </header>

      {error ? <div className="notice error-notice">{error}</div> : null}

      {studies === null && !error ? <div className="empty-state">Loading runs…</div> : null}

      {studies !== null && studies.length === 0 ? (
        <div className="empty-state">
          <h2>No runs yet</h2>
          <p className="subtle">
            Runs appear here as soon as the pipeline creates their directory under the results root.
          </p>
          {config?.launch_enabled ? (
            <a className="button primary" href={href("launch")}>
              Launch your first run
            </a>
          ) : (
            <p className="subtle">
              Launch one with <code>agsci run examples/circpack</code>, or start this server with{" "}
              <code>--enable-launch</code>.
            </p>
          )}
        </div>
      ) : null}

      {studies !== null && studies.length > 0 && filtered.length === 0 ? (
        <div className="empty-state subtle">No runs match “{query}”.</div>
      ) : null}

      <ul className="study-list">
        {filtered.map((study) => {
          const job = jobForStudy(study, jobs);
          const status = displayStatus(study, job);
          return (
            <li key={study.study_id} className="study-card">
              <div className="study-main">
                <div className="study-title-row">
                  <h2 className="study-task">{study.task_name}</h2>
                  <StatusBadge status={status} />
                  {study.dry_run ? <span className="tag">dry run</span> : null}
                  {study.outcome ? <span className="tag">{study.outcome.replace(/_/g, " ")}</span> : null}
                </div>
                <div className="study-meta subtle">
                  <span className="mono">{study.run_id}</span>
                  <span>{study.agent ?? study.agent_dir}{study.model ? ` · ${study.model}` : ""}</span>
                  <span title={study.started_at ?? study.modified_at}>
                    started {formatRelative(study.started_at ?? study.modified_at)}
                  </span>
                  {study.started_at ? <span>{formatDuration(study.started_at, study.ended_at)}</span> : null}
                </div>
                {study.phases.length > 0 ? (
                  <ol className="phase-track" aria-label="Phases">
                    {study.phases.map((phase) => (
                      <li
                        key={phase.phase}
                        className={`phase-step phase-${phase.status}`}
                        title={[
                          `${phaseLabel(phase.phase)}: ${phase.status}`,
                          phase.elapsed ? `(${phase.elapsed})` : "",
                          phase.reason ?? "",
                        ].join(" ").trim()}
                      >
                        <span className="phase-dot" aria-hidden="true" />
                        {phaseLabel(phase.phase)}
                      </li>
                    ))}
                  </ol>
                ) : (
                  <p className="subtle small">No pipeline journal (started outside the pipeline or still preparing).</p>
                )}
              </div>
              <div className="study-actions">
                {study.inspector_run_id ? (
                  <a className="button secondary" href={href("inspect", "", { run: study.inspector_run_id })}>
                    Inspect
                  </a>
                ) : null}
                {study.has_paper ? (
                  <a className="button secondary" href={paperUrl(study.study_id)} target="_blank" rel="noreferrer">
                    Paper PDF
                  </a>
                ) : null}
                {job ? (
                  <a className="button ghost" href={href("jobs", job.job_id)}>
                    Log
                  </a>
                ) : null}
                {config?.launch_enabled && !isActive(job?.status ?? "") ? (
                  <a className="button ghost" href={href("launch", "", { resume: study.study_id, task: study.task_name })}>
                    Continue…
                  </a>
                ) : null}
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
