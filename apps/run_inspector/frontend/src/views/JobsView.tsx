import { useEffect, useRef, useState } from "react";
import { cancelJob, fetchJobLog, isActive, type Job } from "../api/control";
import { formatDuration, formatRelative, href } from "../router";
import { StatusBadge } from "./StudiesView";

interface JobsViewProps {
  jobs: Job[];
  selectedJobId: string | null;
  onChanged: () => Promise<void>;
}

const LOG_POLL_MS = 1500;
const MAX_LOG_CHARS = 2_000_000;
// eslint-disable-next-line no-control-regex
const ANSI_RE = /\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07/g;

/** Strip colour codes and collapse carriage-return progress redraws. */
export function cleanLog(text: string): string {
  return text
    .replace(ANSI_RE, "")
    .split("\n")
    .map((line) => {
      const parts = line.split("\r").filter((part) => part.length > 0);
      return parts.length ? parts[parts.length - 1] : "";
    })
    .join("\n");
}

function kindLabel(job: Job): string {
  if (job.kind === "doctor") return "doctor";
  if (job.dry_run) return "dry run";
  return job.resume_study_id ? "continue" : "run";
}

function JobLog({ job }: { job: Job }) {
  const [text, setText] = useState("");
  const [follow, setFollow] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const offsetRef = useRef(0);
  const preRef = useRef<HTMLPreElement>(null);
  const active = isActive(job.status);

  useEffect(() => {
    offsetRef.current = 0;
    setText("");
    setError(null);
  }, [job.job_id]);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const poll = async () => {
      try {
        // Drain everything already written, then wait for more.
        for (;;) {
          const chunk = await fetchJobLog(job.job_id, offsetRef.current);
          if (cancelled) return;
          offsetRef.current = chunk.next_offset;
          if (chunk.text) {
            setText((prev) => {
              const next = prev + chunk.text;
              return next.length > MAX_LOG_CHARS ? next.slice(next.length - MAX_LOG_CHARS) : next;
            });
          }
          if (chunk.next_offset >= chunk.size) break;
        }
        setError(null);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      }
      if (!cancelled && active) timer = window.setTimeout(() => void poll(), LOG_POLL_MS);
    };
    void poll();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [job.job_id, active]);

  useEffect(() => {
    if (follow && preRef.current) preRef.current.scrollTop = preRef.current.scrollHeight;
  }, [text, follow]);

  return (
    <div className="job-log">
      <div className="job-log-toolbar">
        <span className="subtle small">output.log{active ? " · live" : ""}</span>
        <label className="small follow-toggle">
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> Follow
        </label>
      </div>
      {error ? <div className="inline-error">{error}</div> : null}
      <pre
        ref={preRef}
        className="job-log-body mono"
        onWheel={(e) => {
          if (e.deltaY < 0) setFollow(false);
        }}
      >
        {text ? cleanLog(text) : active ? "Waiting for output…" : "No output."}
      </pre>
    </div>
  );
}

export function JobsView({ jobs, selectedJobId, onChanged }: JobsViewProps) {
  const [actionError, setActionError] = useState<string | null>(null);
  const [now, setNow] = useState(Date.now());
  const selected = jobs.find((job) => job.job_id === selectedJobId) ?? jobs[0] ?? null;

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => setActionError(null), [selected?.job_id]);

  const cancel = async (force: boolean) => {
    if (!selected) return;
    const question = force
      ? "Kill this job immediately? Agent containers may be left behind."
      : "Interrupt this job? The pipeline gets a chance to clean up.";
    if (!window.confirm(question)) return;
    try {
      await cancelJob(selected.job_id, force);
      await onChanged();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    }
  };

  if (jobs.length === 0) {
    return (
      <section className="view-body">
        <header className="view-header">
          <h1>Jobs</h1>
        </header>
        <div className="empty-state">
          <h2>No jobs yet</h2>
          <p className="subtle">Jobs started from the Launch page show up here with their live output.</p>
          <a className="button primary" href={href("launch")}>
            Go to Launch
          </a>
        </div>
      </section>
    );
  }

  return (
    <section className="view-body jobs-layout">
      <aside className="card job-list" aria-label="Jobs">
        {jobs.map((job) => (
          <a
            key={job.job_id}
            href={href("jobs", job.job_id)}
            className={`job-item${selected?.job_id === job.job_id ? " selected" : ""}`}
          >
            <div className="job-item-top">
              <strong>{job.task_name}</strong>
              <StatusBadge status={job.status} />
            </div>
            <div className="subtle small">
              {kindLabel(job)} · {formatRelative(job.created_at, now)}
            </div>
          </a>
        ))}
      </aside>

      {selected ? (
        <section className="card job-detail">
          <header className="job-detail-header">
            <div>
              <div className="study-title-row">
                <h2>{selected.task_name}</h2>
                <StatusBadge status={selected.status} />
                <span className="tag">{kindLabel(selected)}</span>
              </div>
              <div className="study-meta subtle">
                <span className="mono">{selected.job_id}</span>
                <span>started {formatRelative(selected.started_at ?? selected.created_at, now)}</span>
                <span>{formatDuration(selected.started_at, selected.ended_at, now)}</span>
                {selected.return_code !== null ? <span>exit {selected.return_code}</span> : null}
              </div>
            </div>
            <div className="row-actions">
              {selected.run_id || selected.resume_study_id ? (
                <a className="button secondary" href={href("runs")}>
                  View run
                </a>
              ) : null}
              {isActive(selected.status) ? (
                selected.cancel_requested ? (
                  <button className="button danger" onClick={() => void cancel(true)}>
                    Force kill
                  </button>
                ) : (
                  <button className="button danger" onClick={() => void cancel(false)}>
                    Interrupt
                  </button>
                )
              ) : null}
            </div>
          </header>
          {selected.error ? <div className="notice error-notice">{selected.error}</div> : null}
          {actionError ? <div className="inline-error">{actionError}</div> : null}
          <details className="command-details">
            <summary>Command</summary>
            <code className="mono">{selected.argv.join(" ")}</code>
          </details>
          <JobLog job={selected} />
        </section>
      ) : null}
    </section>
  );
}
