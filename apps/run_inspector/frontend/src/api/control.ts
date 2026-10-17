export type JobKind = "run" | "doctor";
export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled" | "interrupted";

export interface ControlConfig {
  launch_enabled: boolean;
  can_create_tasks: boolean;
  max_running_jobs: number;
  agents: string[];
  agent_locked: string | null;
  phases: string[];
  default_phases: string[];
  presets: string[];
  providers: string[];
  reserved_keys: string[];
}

export interface TaskInfo {
  name: string;
  path: string;
  source: string;
  has_scaffold: boolean;
  summary: string;
  excerpt: string;
}

export interface PhaseState {
  phase: string;
  status: string;
  started_at?: string | null;
  ended_at?: string | null;
  elapsed?: string | null;
  reason?: string | null;
  outcome?: string | null;
}

export interface Study {
  study_id: string;
  path: string;
  task_name: string;
  agent_dir: string;
  run_id: string;
  has_paper: boolean;
  inspector_run_id: string | null;
  modified_at: string;
  status: string;
  started_at: string | null;
  ended_at: string | null;
  agent: string | null;
  model: string | null;
  dry_run: boolean;
  outcome: string | null;
  phases: PhaseState[];
}

export interface Job {
  job_id: string;
  kind: JobKind;
  task_name: string;
  task_path: string;
  argv: string[];
  dry_run: boolean;
  run_id: string | null;
  resume_study_id: string | null;
  status: JobStatus;
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
  return_code: number | null;
  cancel_requested: boolean;
  error: string | null;
}

export interface JobLogChunk {
  offset: number;
  next_offset: number;
  size: number;
  text: string;
  status: JobStatus;
}

export interface JobRequest {
  kind: JobKind;
  task_path: string;
  overrides: string[];
  dry_run?: boolean;
  resume_study_id?: string | null;
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      // Non-JSON error body: keep the status line.
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

function post<T>(url: string, body?: unknown): Promise<T> {
  return request<T>(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

function encodePath(id: string): string {
  return id.split("/").filter(Boolean).map(encodeURIComponent).join("/");
}

export const fetchControlConfig = () => request<ControlConfig>("/api/control/config");
export const fetchTasks = () => request<TaskInfo[]>("/api/tasks");
export const createTask = (name: string, brief: string) => post<TaskInfo>("/api/tasks", { name, brief });
export const fetchStudies = () => request<Study[]>("/api/studies");
export const paperUrl = (studyId: string) => `/api/studies/${encodePath(studyId)}/paper.pdf`;

export const fetchJobs = () => request<Job[]>("/api/jobs");
export const fetchJob = (jobId: string) => request<Job>(`/api/jobs/${encodeURIComponent(jobId)}`);
export const previewJob = (body: JobRequest) => post<{ argv: string[] }>("/api/jobs/preview", body);
export const startJob = (body: JobRequest) => post<Job>("/api/jobs", body);
export const cancelJob = (jobId: string, force = false) =>
  post<Job>(`/api/jobs/${encodeURIComponent(jobId)}/cancel${force ? "?force=true" : ""}`);
export const fetchJobLog = (jobId: string, offset: number) =>
  request<JobLogChunk>(`/api/jobs/${encodeURIComponent(jobId)}/log?offset=${offset}`);

export const isActive = (status: string) => status === "queued" || status === "running";
