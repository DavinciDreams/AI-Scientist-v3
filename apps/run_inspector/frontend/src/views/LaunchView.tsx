import { useEffect, useMemo, useState } from "react";
import {
  createTask,
  fetchTasks,
  previewJob,
  startJob,
  type ControlConfig,
  type Job,
  type JobKind,
  type JobRequest,
  type TaskInfo,
} from "../api/control";
import { href, phaseLabel } from "../router";

interface LaunchViewProps {
  config: ControlConfig | null;
  params: URLSearchParams;
  onLaunched: (job: Job) => void;
}

const BUDGETS = [
  { key: "investigation.execution.max_idea_cycles", label: "Idea cycles", hint: "default 3", quick: "2" },
  { key: "ablations.execution.num_replications", label: "Ablation replications", hint: "default 5", quick: "1" },
  { key: "ablations.execution.max_evals_per_variant", label: "Evals per variant", hint: "default 5", quick: "2" },
  { key: "posthoc.execution.max_selected_analyses", label: "Post-hoc analyses", hint: "default 3", quick: "1" },
] as const;

type Budgets = Record<(typeof BUDGETS)[number]["key"], string>;
const EMPTY_BUDGETS = Object.fromEntries(BUDGETS.map((b) => [b.key, ""])) as Budgets;
const QUICK_BUDGETS = Object.fromEntries(BUDGETS.map((b) => [b.key, b.quick])) as Budgets;

function quoteArg(arg: string): string {
  return /^[\w@%+=:,./<>-]+$/.test(arg) ? arg : `'${arg.replace(/'/g, `'\\''`)}'`;
}

export function LaunchView({ config, params, onLaunched }: LaunchViewProps) {
  const [tasks, setTasks] = useState<TaskInfo[] | null>(null);
  const [taskPath, setTaskPath] = useState<string>("");
  const [tasksError, setTasksError] = useState<string | null>(null);

  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [newBrief, setNewBrief] = useState("");
  const [createError, setCreateError] = useState<string | null>(null);

  const [agent, setAgent] = useState("");
  const [model, setModel] = useState("");
  const [provider, setProvider] = useState("");
  const [preset, setPreset] = useState("");
  const [phases, setPhases] = useState<string[]>([]);
  const [budgets, setBudgets] = useState<Budgets>(EMPTY_BUDGETS);
  const [extra, setExtra] = useState("");
  const [resume, setResume] = useState<string | null>(params.get("resume"));

  const [preview, setPreview] = useState<string[] | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => {
    if (config) setPhases(config.default_phases);
  }, [config]);

  useEffect(() => {
    let cancelled = false;
    fetchTasks()
      .then((list) => {
        if (cancelled) return;
        setTasks(list);
        const wanted = params.get("task");
        const match = list.find((t) => t.path === wanted || t.name === wanted) ?? list[0];
        if (match) setTaskPath(match.path);
      })
      .catch((err) => !cancelled && setTasksError(err instanceof Error ? err.message : String(err)));
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedTask = tasks?.find((t) => t.path === taskPath) ?? null;

  const overrides = useMemo(() => {
    const list: string[] = [];
    if (!config?.agent_locked) {
      if (provider) list.push(`provider=${provider}`);
      if (agent) list.push(`agent.name=${agent}`);
      if (model.trim()) list.push(`agent.model=${model.trim()}`);
    }
    if (preset) list.push(`preset=${preset}`);
    if (config) {
      const ordered = config.phases.filter((p) => phases.includes(p));
      if (ordered.join(",") !== config.default_phases.join(",")) {
        list.push(`execution.phases=${ordered.join(",")}`);
      }
    }
    for (const budget of BUDGETS) {
      const value = budgets[budget.key].trim();
      if (value) list.push(`${budget.key}=${value}`);
    }
    for (const line of extra.split("\n")) {
      if (line.trim() && !line.trim().startsWith("#")) list.push(line.trim());
    }
    return list;
  }, [config, provider, agent, model, preset, phases, budgets, extra]);

  const request = (kind: JobKind, dryRun = false): JobRequest => ({
    kind,
    task_path: taskPath,
    overrides,
    dry_run: dryRun,
    resume_study_id: kind === "run" ? resume : null,
  });

  useEffect(() => {
    if (!taskPath) {
      setPreview(null);
      return;
    }
    const timer = window.setTimeout(() => {
      previewJob(request("run"))
        .then((result) => {
          setPreview(result.argv);
          setPreviewError(null);
        })
        .catch((err) => setPreviewError(err instanceof Error ? err.message : String(err)));
    }, 250);
    return () => window.clearTimeout(timer);
  }, [taskPath, overrides, resume]);

  const submit = async (kind: JobKind, dryRun: boolean, label: string) => {
    setSubmitting(label);
    setSubmitError(null);
    try {
      onLaunched(await startJob(request(kind, dryRun)));
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(null);
    }
  };

  const saveTask = async () => {
    setCreateError(null);
    try {
      const task = await createTask(newName.trim(), newBrief);
      setTasks((prev) => [...(prev ?? []), task]);
      setTaskPath(task.path);
      setCreating(false);
      setNewName("");
      setNewBrief("");
      // A bare brief needs the drafting phase to write its evaluator.
      setPhases((prev) => (prev.includes("task_drafting") ? prev : ["task_drafting", ...prev]));
    } catch (err) {
      setCreateError(err instanceof Error ? err.message : String(err));
    }
  };

  const launchEnabled = Boolean(config?.launch_enabled);
  const needsDrafting = selectedTask !== null && !selectedTask.has_scaffold && !phases.includes("task_drafting");
  const busy = submitting !== null;

  return (
    <section className="view-body">
      <header className="view-header">
        <div>
          <h1>Launch</h1>
          <p className="subtle">Pick a task, set the budget, then check, preview, or launch the pipeline.</p>
        </div>
      </header>

      {!launchEnabled && config ? (
        <div className="notice">
          This server is read-only. Restart it with <code>agsci inspect start --enable-launch</code> to launch
          jobs from here. The command preview still works.
        </div>
      ) : null}

      {resume ? (
        <div className="notice resume-notice">
          <div>
            Continuing <span className="mono">{resume}</span>: the selected phases run against the existing
            run directory (<code>paths.run_dir</code>). Pass the same agent and model as the original run.
          </div>
          <button className="button ghost" onClick={() => setResume(null)}>
            Start a fresh run instead
          </button>
        </div>
      ) : null}

      <div className="launch-grid">
        <section className="card">
          <div className="card-header">
            <h2>1 · Task</h2>
            {config?.can_create_tasks && !creating ? (
              <button className="button ghost" onClick={() => setCreating(true)}>
                + New from brief
              </button>
            ) : null}
          </div>

          {creating ? (
            <div className="new-task">
              <label className="field">
                <span>Name</span>
                <input
                  value={newName}
                  onChange={(e) => setNewName(e.target.value.toLowerCase())}
                  placeholder="my_task"
                  pattern="[a-z0-9][a-z0-9_-]*"
                />
              </label>
              <label className="field">
                <span>Research brief (task.md)</span>
                <textarea
                  rows={10}
                  value={newBrief}
                  onChange={(e) => setNewBrief(e.target.value)}
                  placeholder={"# Task title\n\nDescribe the problem, the objective to optimise, and how a solution is scored."}
                />
              </label>
              <p className="subtle small">
                Drafting a task from a bare brief is experimental and works best for small tasks. The task-drafting
                phase writes the evaluator and contract.
              </p>
              {createError ? <div className="inline-error">{createError}</div> : null}
              <div className="row-actions">
                <button className="button ghost" onClick={() => setCreating(false)}>
                  Cancel
                </button>
                <button className="button primary" disabled={!newName.trim() || !newBrief.trim()} onClick={() => void saveTask()}>
                  Save task
                </button>
              </div>
            </div>
          ) : null}

          {tasksError ? <div className="inline-error">{tasksError}</div> : null}
          {tasks?.length === 0 ? <p className="subtle">No tasks found in the configured task directories.</p> : null}
          <div className="task-list" role="radiogroup" aria-label="Task">
            {tasks?.map((task) => (
              <label key={task.path} className={`task-option${task.path === taskPath ? " selected" : ""}`}>
                <input
                  type="radio"
                  name="task"
                  value={task.path}
                  checked={task.path === taskPath}
                  onChange={() => setTaskPath(task.path)}
                />
                <div className="task-option-body">
                  <div className="task-option-title">
                    <strong>{task.name}</strong>
                    <span className="tag">{task.source}</span>
                    {!task.has_scaffold ? <span className="tag warn">brief only</span> : null}
                  </div>
                  {task.summary ? <div className="task-option-subtitle">{task.summary}</div> : null}
                  {task.path === taskPath ? <pre className="task-excerpt">{task.excerpt}</pre> : null}
                </div>
              </label>
            ))}
          </div>
        </section>

        <section className="card">
          <div className="card-header">
            <h2>2 · Configure</h2>
          </div>

          <div className="field-grid">
            {config?.agent_locked ? (
              <div className="field">
                <span>Agent</span>
                <div className="locked-value">{config.agent_locked}</div>
              </div>
            ) : (
              <>
                {config && config.providers.length > 0 ? (
                  <label className="field">
                    <span>Provider</span>
                    <select value={provider} onChange={(e) => setProvider(e.target.value)}>
                      <option value="">none</option>
                      {config.providers.map((p) => (
                        <option key={p} value={p}>
                          {p}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
                <label className="field">
                  <span>Agent</span>
                  <select value={agent} onChange={(e) => setAgent(e.target.value)}>
                    <option value="">{provider ? "from provider" : "default (claude)"}</option>
                    {config?.agents.map((a) => (
                      <option key={a} value={a}>
                        {a}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span>Model</span>
                  <input value={model} onChange={(e) => setModel(e.target.value)} placeholder="agent default" />
                </label>
              </>
            )}
            <label className="field">
              <span>Preset</span>
              <select value={preset} onChange={(e) => setPreset(e.target.value)}>
                <option value="">none</option>
                {config?.presets.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <fieldset className="phase-picker">
            <legend>Phases</legend>
            {config?.phases.map((phase) => (
              <label key={phase} className={`phase-chip${phases.includes(phase) ? " on" : ""}`}>
                <input
                  type="checkbox"
                  checked={phases.includes(phase)}
                  onChange={(e) =>
                    setPhases((prev) => (e.target.checked ? [...prev, phase] : prev.filter((p) => p !== phase)))
                  }
                />
                {phaseLabel(phase)}
              </label>
            ))}
          </fieldset>
          {needsDrafting ? (
            <div className="inline-warning">This task has no evaluator yet; include the Drafting phase.</div>
          ) : null}

          <div className="budget-header">
            <h3>Budget</h3>
            <div className="row-actions">
              <button className="button ghost small" onClick={() => setBudgets(QUICK_BUDGETS)}>
                Quick first run
              </button>
              <button className="button ghost small" onClick={() => setBudgets(EMPTY_BUDGETS)}>
                Defaults
              </button>
            </div>
          </div>
          <div className="field-grid">
            {BUDGETS.map((budget) => (
              <label key={budget.key} className="field">
                <span>{budget.label}</span>
                <input
                  type="number"
                  min={1}
                  inputMode="numeric"
                  value={budgets[budget.key]}
                  placeholder={budget.hint}
                  onChange={(e) => setBudgets((prev) => ({ ...prev, [budget.key]: e.target.value }))}
                />
              </label>
            ))}
          </div>

          <label className="field">
            <span>Extra overrides (one key=value per line)</span>
            <textarea
              className="mono"
              rows={3}
              value={extra}
              onChange={(e) => setExtra(e.target.value)}
              placeholder={"paper.paper_format=iclr\ninvestigation.execution.workers=2"}
            />
          </label>

          <div className="command-preview">
            <div className="command-preview-label">Command</div>
            {previewError ? (
              <div className="inline-error">{previewError}</div>
            ) : (
              <code className="mono">{preview ? preview.map(quoteArg).join(" ") : "Select a task"}</code>
            )}
          </div>

          {submitError ? <div className="inline-error">{submitError}</div> : null}

          <div className="launch-actions">
            <button
              className="button secondary"
              disabled={!launchEnabled || !taskPath || busy || Boolean(previewError)}
              onClick={() => void submit("doctor", false, "doctor")}
              title="agsci doctor: check CLI, credentials, image, GPUs and task scaffold"
            >
              {submitting === "doctor" ? "Starting…" : "Check setup"}
            </button>
            <button
              className="button secondary"
              disabled={!launchEnabled || !taskPath || busy || Boolean(previewError)}
              onClick={() => void submit("run", true, "dry")}
              title="Validate the configuration and print every phase command without running anything"
            >
              {submitting === "dry" ? "Starting…" : "Dry run"}
            </button>
            <button
              className="button primary"
              disabled={!launchEnabled || !taskPath || busy || Boolean(previewError)}
              onClick={() => void submit("run", false, "run")}
            >
              {submitting === "run" ? "Launching…" : resume ? "Continue run" : "Launch run"}
            </button>
          </div>
          <p className="subtle small">
            Jobs keep running if you close this page. Follow them under <a href={href("jobs")}>Jobs</a>.
          </p>
        </section>
      </div>
    </section>
  );
}
