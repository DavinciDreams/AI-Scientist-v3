import type { NodeDetail } from "../../types";

interface ProgramSummaryRow {
  label: string;
  combined_score: number | null;
  correct: boolean | null;
}

function toProgramRows(value: unknown): ProgramSummaryRow[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const rows: ProgramSummaryRow[] = [];
  for (const item of value) {
    if (typeof item !== "object" || item === null) {
      continue;
    }
    const row = item as Record<string, unknown>;
    const label = typeof row.label === "string" ? row.label : "";
    if (!label) {
      continue;
    }
    rows.push({
      label,
      combined_score: typeof row.combined_score === "number" ? row.combined_score : null,
      correct: typeof row.correct === "boolean" ? row.correct : null,
    });
  }
  return rows;
}

export function IdeaOverview({ detail }: { detail: NodeDetail }) {
  const programs = toProgramRows(detail.metadata.programs);

  return (
    <div className="idea-overview">
      <div className="kv-grid">
        <div className="kv-label">Title</div>
        <div className="kv-value">{String(detail.metadata.title ?? detail.node.label)}</div>
        <div className="kv-label">Idea ID</div>
        <div className="kv-value mono">{String(detail.metadata.idea_id ?? "-")}</div>
        <div className="kv-label">Worker</div>
        <div className="kv-value">{String(detail.metadata.worker_index ?? "-")}</div>
        <div className="kv-label">Program Evals</div>
        <div className="kv-value">{String(detail.metadata.program_eval_count ?? 0)}</div>
        <div className="kv-label">Path</div>
        <div className="kv-value mono">{String(detail.metadata.idea_path ?? "-")}</div>
      </div>

      <div className="idea-programs">
        <h3>Programs</h3>
        {programs.length === 0 ? (
          <div className="subtle">No programs found for this idea.</div>
        ) : (
          <table className="program-table">
            <thead>
              <tr>
                <th>Program</th>
                <th>Score</th>
                <th>Correct</th>
              </tr>
            </thead>
            <tbody>
              {programs.map((program, index) => (
                <tr key={`${program.label}:${index}`}>
                  <td className="mono">{program.label}</td>
                  <td>{program.combined_score ?? "-"}</td>
                  <td>{program.correct === null ? "-" : String(program.correct)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
