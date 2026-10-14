import type {
  FileContent,
  IdeaAncestors,
  IdeaTimeline,
  NodeDetail,
  ProgramDirDiff,
  ProgramDiff,
  RunGraph,
  RunPerformance,
  RunSummary,
} from "../types";

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${message}`);
  }
  return (await response.json()) as T;
}

function encodeRunIdPath(runId: string): string {
  return runId
    .split("/")
    .filter((segment) => segment.length > 0)
    .map((segment) => encodeURIComponent(segment))
    .join("/");
}

export async function forceRefresh(): Promise<void> {
  const response = await fetch("/api/refresh", { method: "POST" });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${message}`);
  }
}

export async function fetchRuns(): Promise<RunSummary[]> {
  return fetchJson<RunSummary[]>("/api/runs");
}

export async function fetchRunGraph(runId: string): Promise<RunGraph> {
  return fetchJson<RunGraph>(`/api/runs/${encodeRunIdPath(runId)}/graph`);
}

export async function fetchNodeDetail(runId: string, nodeId: string): Promise<NodeDetail> {
  return fetchJson<NodeDetail>(
    `/api/runs/${encodeRunIdPath(runId)}/nodes/${encodeURIComponent(nodeId)}`,
  );
}

export async function fetchFileContent(runId: string, path: string): Promise<FileContent> {
  const query = new URLSearchParams({ run_id: runId, path });
  return fetchJson<FileContent>(`/api/file?${query.toString()}`);
}

export async function fetchRunPerformance(runId: string): Promise<RunPerformance> {
  return fetchJson<RunPerformance>(`/api/runs/${encodeRunIdPath(runId)}/performance`);
}

export async function fetchIdeaTimeline(runId: string): Promise<IdeaTimeline> {
  return fetchJson<IdeaTimeline>(`/api/runs/${encodeRunIdPath(runId)}/ideas`);
}

export async function fetchIdeaAncestors(
  runId: string,
  ideaId: string,
  workerName?: string | null,
): Promise<IdeaAncestors> {
  const base = `/api/runs/${encodeRunIdPath(runId)}/idea-ancestors/${encodeURIComponent(ideaId)}`;
  const qs = workerName != null ? `?worker_name=${encodeURIComponent(workerName)}` : "";
  return fetchJson<IdeaAncestors>(`${base}${qs}`);
}

export async function fetchProgramDiff(
  runId: string,
  fromPath: string,
  toPath: string,
): Promise<ProgramDiff> {
  const query = new URLSearchParams({ from_path: fromPath, to_path: toPath });
  return fetchJson<ProgramDiff>(
    `/api/runs/${encodeRunIdPath(runId)}/program-diff?${query.toString()}`,
  );
}

export async function fetchProgramDirDiff(
  runId: string,
  fromDir: string,
  toDir: string,
): Promise<ProgramDirDiff> {
  const query = new URLSearchParams({ from_dir: fromDir, to_dir: toDir });
  return fetchJson<ProgramDirDiff>(
    `/api/runs/${encodeRunIdPath(runId)}/program-dir-diff?${query.toString()}`,
  );
}
