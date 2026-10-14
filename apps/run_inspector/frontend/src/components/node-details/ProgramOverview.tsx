import type { NodeDetail } from "../../types";

export function ProgramOverview({ detail }: { detail: NodeDetail }) {
  return (
    <div className="kv-grid">
      <div className="kv-label">Program</div>
      <div className="kv-value">{detail.node.label}</div>
      <div className="kv-label">Combined Score</div>
      <div className="kv-value">{String(detail.metadata.combined_score ?? "-")}</div>
      <div className="kv-label">Correct</div>
      <div className="kv-value">{String(detail.metadata.correct ?? "-")}</div>
      <div className="kv-label">Attempt</div>
      <div className="kv-value mono">{String(detail.metadata.attempt_name ?? "(none)")}</div>
      <div className="kv-label">Idea</div>
      <div className="kv-value mono">{String(detail.metadata.idea_id ?? "-")}</div>
      <div className="kv-label">Path</div>
      <div className="kv-value mono">{String(detail.metadata.program_dir ?? "-")}</div>
    </div>
  );
}
