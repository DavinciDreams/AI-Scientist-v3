import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchNodeDetail, fetchRunGraph, fetchRuns, forceRefresh } from "./api/client";
import { DetailPanel } from "./components/DetailPanel";
import { GraphPanel } from "./components/GraphPanel";
import type { NodeDetail, RunGraph, RunSummary } from "./types";

export function App() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [graph, setGraph] = useState<RunGraph | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [detail, setDetail] = useState<NodeDetail | null>(null);
  const [status, setStatus] = useState("Loading runs...");
  const [error, setError] = useState<string | null>(null);
  const [refreshCount, setRefreshCount] = useState(0);
  const [refreshing, setRefreshing] = useState(false);
  // Idea node ids revealed during the GraphPanel progressive-build animation; forwarded
  // to the performance plot so it reveals in lockstep. `null` = show everything.
  const [revealedIdeaNodeIds, setRevealedIdeaNodeIds] = useState<Set<string> | null>(null);

  const loadRuns = async () => {
    setError(null);
    setStatus("Refreshing runs...");
    try {
      const runList = await fetchRuns();
      setRuns(runList);
      if (runList.length === 0) {
        setSelectedRunId(null);
        setGraph(null);
        setDetail(null);
        setStatus("No runs found under results root.");
        return;
      }

      setSelectedRunId((prev) => {
        if (prev && runList.some((run) => run.run_id === prev)) {
          return prev;
        }
        return runList[0].run_id;
      });
      setStatus(`Loaded ${runList.length} run(s).`);
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : String(runError));
      setStatus("Failed to load runs.");
    }
  };

  useEffect(() => {
    void loadRuns();
  }, []);

  useEffect(() => {
    setRevealedIdeaNodeIds(null);
    if (!selectedRunId) {
      setGraph(null);
      setDetail(null);
      return;
    }

    let cancelled = false;
    const loadGraph = async () => {
      setStatus(`Loading graph for ${selectedRunId}...`);
      setError(null);
      try {
        const graphPayload = await fetchRunGraph(selectedRunId);
        if (cancelled) return;
        setGraph(graphPayload);

        // The graph response carries a fully-populated RunSummary (with counts).
        // Merge it back into the runs list so the header pills update.
        setRuns((prev) =>
          prev.map((r) => (r.run_id === graphPayload.run.run_id ? graphPayload.run : r)),
        );

        const existing = selectedNodeId;
        if (existing && graphPayload.nodes.some((node) => node.id === existing)) {
          setSelectedNodeId(existing);
        } else if (graphPayload.nodes.length > 0) {
          setSelectedNodeId(graphPayload.nodes[0].id);
        } else {
          setSelectedNodeId(null);
          setDetail(null);
        }
        setStatus(`Graph loaded: ${graphPayload.nodes.length} nodes.`);
      } catch (graphError) {
        if (cancelled) return;
        setError(graphError instanceof Error ? graphError.message : String(graphError));
        setGraph(null);
        setDetail(null);
        setStatus("Failed to load graph.");
      }
    };

    void loadGraph();
    return () => {
      cancelled = true;
    };
  }, [selectedRunId, refreshCount]);

  useEffect(() => {
    if (!selectedRunId || !selectedNodeId) {
      setDetail(null);
      return;
    }

    let cancelled = false;
    const loadDetail = async () => {
      setError(null);
      try {
        const detailPayload = await fetchNodeDetail(selectedRunId, selectedNodeId);
        if (cancelled) return;
        setDetail(detailPayload);
      } catch (detailError) {
        if (cancelled) return;
        setError(detailError instanceof Error ? detailError.message : String(detailError));
        setDetail(null);
      }
    };

    void loadDetail();
    return () => {
      cancelled = true;
    };
  }, [selectedRunId, selectedNodeId, refreshCount]);

  const handleSelectIdea = useCallback(
    (nodeId: string) => {
      setSelectedNodeId(nodeId);
    },
    [],
  );

  const selectedRunSummary = useMemo(
    () => runs.find((run) => run.run_id === selectedRunId) ?? null,
    [runs, selectedRunId],
  );

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="header-left">
          <h1>Agentic Scientist Inspector</h1>
          <section className="run-summary-row">
            {selectedRunSummary ? (
              <>
                <div className="summary-pill">agent: {selectedRunSummary.agent}</div>
                <div className="summary-pill">workers: {selectedRunSummary.worker_count ?? "..."}</div>
                <div className="summary-pill">ideas: {selectedRunSummary.idea_count ?? "..."}</div>
                <div className="summary-pill">program evals: {selectedRunSummary.program_eval_count ?? "..."}</div>
                <div className="summary-pill">containerized: {selectedRunSummary.containerized != null ? String(selectedRunSummary.containerized) : "..."}</div>
                <div className="summary-pill">
                  best score: {selectedRunSummary.best_combined_score ?? "n/a"}
                </div>
              </>
            ) : (
              <div className="subtle">No run selected.</div>
            )}
          </section>
        </div>
        <div className="controls">
          <select
            className="run-select"
            value={selectedRunId ?? ""}
            onChange={(event) => setSelectedRunId(event.target.value || null)}
          >
            {runs.map((run) => (
              <option key={run.run_id} value={run.run_id}>
                {run.run_id}
              </option>
            ))}
          </select>
          <button
            className="refresh-button"
            disabled={refreshing}
            onClick={() => {
              setRefreshing(true);
              void forceRefresh()
                .then(() => { setRefreshCount((c) => c + 1); return loadRuns(); })
                .finally(() => setRefreshing(false));
            }}
          >
            {refreshing ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </header>

      {refreshing && <div className="refresh-progress"><div className="refresh-progress-bar" /></div>}

      <section className="status-row">
        <span>{status}</span>
        {error ? <span className="error">{error}</span> : null}
      </section>

      <section className="main-panels">
        <div className="panel left-panel">
          <GraphPanel
            runId={selectedRunId}
            graph={graph}
            selectedNodeId={selectedNodeId}
            onSelectNode={setSelectedNodeId}
            onRevealChange={setRevealedIdeaNodeIds}
          />
        </div>
        <div className="panel right-panel">
          <DetailPanel
            runId={selectedRunId}
            detail={detail}
            onSelectIdea={handleSelectIdea}
            revealedIdeaNodeIds={revealedIdeaNodeIds}
          />
        </div>
      </section>
    </main>
  );
}
