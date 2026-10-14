import type { NodeDetail } from "../../types";

export function WorkerOverview({ detail }: { detail: NodeDetail }) {
  return (
    <div className="kv-grid">
      <div className="kv-label">Node</div>
      <div className="kv-value">{detail.node.label}</div>
      <div className="kv-label">Worker Index</div>
      <div className="kv-value">{String(detail.metadata.worker_index ?? "-")}</div>
      <div className="kv-label">Synthetic</div>
      <div className="kv-value">{String(detail.metadata.synthetic ?? false)}</div>
      <div className="kv-label">Idea Count</div>
      <div className="kv-value">{String(detail.metadata.idea_count ?? 0)}</div>
      <div className="kv-label">Path</div>
      <div className="kv-value mono">{String(detail.metadata.worker_path ?? "-")}</div>
    </div>
  );
}
