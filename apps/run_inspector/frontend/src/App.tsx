import { useCallback, useEffect, useState } from "react";
import { fetchControlConfig, fetchJobs, isActive, type ControlConfig, type Job } from "./api/control";
import { href, navigate, useRoute, type ViewName } from "./router";
import { InspectorView } from "./views/InspectorView";
import { JobsView } from "./views/JobsView";
import { LaunchView } from "./views/LaunchView";
import { StudiesView } from "./views/StudiesView";

const NAV: { view: ViewName; label: string }[] = [
  { view: "runs", label: "Runs" },
  { view: "launch", label: "Launch" },
  { view: "jobs", label: "Jobs" },
  { view: "inspect", label: "Inspector" },
];

const JOB_POLL_MS = 4000;

export function App() {
  const route = useRoute();
  const [config, setConfig] = useState<ControlConfig | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);

  useEffect(() => {
    fetchControlConfig().then(setConfig).catch(() => setConfig(null));
  }, []);

  const refreshJobs = useCallback(async () => {
    try {
      setJobs(await fetchJobs());
    } catch {
      // Keep the last list; the views surface their own errors.
    }
  }, []);

  useEffect(() => {
    void refreshJobs();
    const timer = window.setInterval(() => void refreshJobs(), JOB_POLL_MS);
    return () => window.clearInterval(timer);
  }, [refreshJobs]);

  const activeJobs = jobs.filter((job) => isActive(job.status)).length;

  return (
    <main className={`app-shell view-${route.view}`}>
      <nav className="top-nav" aria-label="Sections">
        <a className="brand" href={href("runs")}>
          <span className="brand-mark" aria-hidden="true" />
          Agentic Scientist
        </a>
        <div className="nav-links">
          {NAV.map((item) => (
            <a
              key={item.view}
              href={href(item.view)}
              className={`nav-link${route.view === item.view ? " active" : ""}`}
              aria-current={route.view === item.view ? "page" : undefined}
            >
              {item.label}
              {item.view === "jobs" && activeJobs > 0 ? (
                <span className="nav-badge" title={`${activeJobs} running`}>{activeJobs}</span>
              ) : null}
            </a>
          ))}
        </div>
        <div className="nav-meta">
          {config && !config.launch_enabled ? (
            <span className="mode-pill" title="Start the server with --enable-launch to run jobs from the UI">
              read-only
            </span>
          ) : null}
          {config?.agent_locked ? <span className="mode-pill">{config.agent_locked}</span> : null}
        </div>
      </nav>

      {route.view === "runs" ? <StudiesView jobs={jobs} config={config} /> : null}
      {route.view === "launch" ? (
        <LaunchView
          // Remount when the link changes the requested task or run to continue.
          key={route.params.toString()}
          config={config}
          params={route.params}
          onLaunched={(job) => {
            void refreshJobs();
            navigate(href("jobs", job.job_id));
          }}
        />
      ) : null}
      {route.view === "jobs" ? (
        <JobsView jobs={jobs} selectedJobId={route.rest || null} onChanged={refreshJobs} />
      ) : null}
      {route.view === "inspect" ? (
        <InspectorView
          runId={route.params.get("run")}
          onRunChange={(runId) => navigate(href("inspect", "", { run: runId }), { replace: true })}
        />
      ) : null}
    </main>
  );
}
