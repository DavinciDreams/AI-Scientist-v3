export type NodeType = "worker" | "idea" | "program_eval";

export interface RunSummary {
  run_id: string;
  path: string;
  task_name: string;
  agent: string;
  containerized: boolean | null;
  parallel: boolean | null;
  worker_count: number | null;
  idea_count: number | null;
  program_eval_count: number | null;
  best_combined_score: number | null;
}

export interface GraphNode {
  id: string;
  type: NodeType;
  label: string;
  data: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  kind: string;
}

export interface RunGraph {
  run: RunSummary;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface IdeaPerformance {
  worker_index: number;
  worker_name: string;
  idea_node_id: string;
  idea_id: string;
  idea_title: string;
  idea_dir_name: string;
  idea_path: string;
  idea_order_index: number | null;
  program_count: number;
  scored_program_count: number;
  correct_program_count: number;
  incorrect_program_count: number;
  best_combined_score: number | null;
  best_program_label: string | null;
  best_correct: boolean | null;
}

export interface ProgramPerformance {
  worker_index: number;
  worker_name: string;
  idea_node_id: string;
  idea_id: string;
  idea_title: string;
  idea_dir_name: string;
  idea_path: string;
  idea_order_index: number | null;
  program_label: string;
  program_idx: number;
  sequence_idx: number;
  attempt_name: string | null;
  program_code_path: string | null;
  timestamp_unix: number | null;
  combined_score: number | null;
  correct: boolean | null;
}

export interface RunPerformance {
  run_id: string;
  ideas: IdeaPerformance[];
  programs: ProgramPerformance[];
  max_idea_combined_score: number | null;
  max_program_combined_score: number | null;
}

export interface ArtifactRef {
  name: string;
  path: string;
  kind: "json" | "code" | "log" | "data" | "text" | "other";
  exists: boolean;
  size_bytes: number | null;
  agentlens_url?: string | null;
}

export interface NodeDetail {
  run_id: string;
  node_id: string;
  node: GraphNode;
  metadata: Record<string, unknown>;
  artifacts: ArtifactRef[];
  agentlens_url: string | null;
}

export interface FileContent {
  path: string;
  size_bytes: number;
  truncated: boolean;
  text: string;
}

// ---------------------------------------------------------------------------
// Idea timeline types (used by GraphPanel tree)
// ---------------------------------------------------------------------------

export interface IdeaAttemptSummary {
  attempt_id: string | null;
  attempt_number: number | null;
  attempt_status: string | null;
  rubric_overall_pass: boolean;
  best_updated: boolean;
  metrics: Record<string, unknown> | null;
  timestamp: string | null;
}

export interface IdeaTimelineEntry {
  node_id: string | null;
  idea_id: string;
  idea_path: string | null;
  cycle_index: number | null;
  title: string;
  timestamp: string | null;
  hypothesis: string | null;
  implementation_plan: string[] | null;
  notes_for_agent: string | null;
  target_rubrics: Record<string, unknown>[];
  num_attempts: number;
  best_score: number | null;
  any_pass: boolean;
  rubric_summary: { passed: number; total: number } | null;
  attempts: IdeaAttemptSummary[];
  worker_name: string | null;
  prune_chapter: number | null;
}

export interface PruneEvent {
  chapter: number;
  cycle_index: number;
  timestamp: string | null;
}

export interface IdeaTimeline {
  run_id: string;
  ideas: IdeaTimelineEntry[];
  prune_events: PruneEvent[];
}

// ---------------------------------------------------------------------------
// Tree / diff types
// ---------------------------------------------------------------------------

export interface ProgramLocation {
  path: string;
}

export interface AncestorEntry {
  idea_id: string;
  title: string;
  cycle_index: number | null;
  worker_name: string | null;
  program_info: ProgramLocation | null;
  programs: SiblingProgramEntry[];
}

export interface SiblingProgramEntry {
  label: string;
  attempt_name: string | null;
  program_info: ProgramLocation | null;
}

export interface IdeaAncestors {
  idea_id: string;
  target_program: ProgramLocation | null;
  ancestors: AncestorEntry[];
  sibling_programs: SiblingProgramEntry[];
}

export interface DiffLine {
  type: "context" | "addition" | "removal" | "hunk_header";
  content: string;
  from_line: number | null;
  to_line: number | null;
}

export interface DiffStats {
  additions: number;
  deletions: number;
  hunks: number;
}

export interface ProgramDiff {
  lines: DiffLine[];
  stats: DiffStats;
}

export type ProgramDirDiffStatus = "added" | "removed" | "modified" | "unchanged";

export interface ProgramDirDiffFile {
  relpath: string;
  status: ProgramDirDiffStatus;
  from_path: string;
  to_path: string;
  from_exists: boolean;
  to_exists: boolean;
  is_binary: boolean;
  from_size_bytes: number | null;
  to_size_bytes: number | null;
}

export interface ProgramDirDiffStats {
  added: number;
  removed: number;
  modified: number;
  unchanged: number;
}

export interface ProgramDirDiff {
  files: ProgramDirDiffFile[];
  stats: ProgramDirDiffStats;
}
