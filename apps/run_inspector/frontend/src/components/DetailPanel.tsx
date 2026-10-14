import { useEffect, useMemo, useRef, useState } from "react";
import { fetchFileContent, fetchRunPerformance } from "../api/client";
import type { ArtifactRef, FileContent, NodeDetail, RunPerformance } from "../types";
import { PerformanceOverview } from "./node-details/PerformanceOverview";
import { ProgramDiffViewer } from "./ProgramDiffViewer";

type TabId = "performance" | "artifacts" | "code" | "logs" | "diff";

interface DetailPanelProps {
  runId: string | null;
  detail: NodeDetail | null;
  onSelectIdea?: (nodeId: string) => void;
  /** Idea node ids revealed by the progressive-build animation; `null` = show all. */
  revealedIdeaNodeIds?: Set<string> | null;
}

interface ArtifactBrowserProps {
  runId: string;
  artifacts: ArtifactRef[];
  loaded: FileContent | null;
  loading: boolean;
  error: string | null;
  selectedPath: string | null;
  onLoad: (path: string) => void;
  showAgentLensAction?: boolean;
}

function ArtifactBrowser({
  runId,
  artifacts,
  loaded,
  loading,
  error,
  selectedPath,
  onLoad,
  showAgentLensAction = false,
}: ArtifactBrowserProps) {
  const selectedArtifact = useMemo(
    () => artifacts.find((artifact) => artifact.path === selectedPath) ?? null,
    [artifacts, selectedPath],
  );

  return (
    <div className="artifact-browser">
      <div className="artifact-list">
        {artifacts.length === 0 ? <div className="subtle">No artifacts in this view.</div> : null}
        {artifacts.map((artifact) => (
          <button
            key={`${artifact.path}:${artifact.name}`}
            className={`artifact-item ${selectedPath === artifact.path ? "active" : ""}`}
            disabled={!artifact.exists || !runId}
            onClick={() => onLoad(artifact.path)}
          >
            <span className="artifact-name">{artifact.name}</span>
            <span className="artifact-meta">{artifact.exists ? artifact.kind : "missing"}</span>
          </button>
        ))}
      </div>
      <div className="artifact-content">
        {showAgentLensAction && selectedArtifact?.agentlens_url ? (
          <div className="artifact-content-actions">
            <a href={selectedArtifact.agentlens_url} target="_blank" rel="noreferrer" className="action-link">
              Open in AgentLens
            </a>
          </div>
        ) : null}
        {loading ? <div className="subtle">Loading file...</div> : null}
        {error ? <div className="error-box">{error}</div> : null}
        {!loading && !error && !loaded ? <div className="subtle">Select an artifact to inspect.</div> : null}
        {loaded ? (
          <>
            <div className="artifact-content-header">
              <span className="mono artifact-path">{loaded.path}</span>
              <span className="artifact-size">{loaded.size_bytes} bytes{loaded.truncated ? " (truncated)" : ""}</span>
            </div>
            <pre className="content-pre">{loaded.text}</pre>
          </>
        ) : null}
      </div>
    </div>
  );
}

export function DetailPanel({ runId, detail, onSelectIdea, revealedIdeaNodeIds = null }: DetailPanelProps) {
  const [activeTab, setActiveTab] = useState<TabId>("performance");
  const [selectedArtifactPath, setSelectedArtifactPath] = useState<string | null>(null);
  const [codeArtifactsOverride, setCodeArtifactsOverride] = useState<ArtifactRef[] | null>(null);
  const [fileContent, setFileContent] = useState<FileContent | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [loadingFile, setLoadingFile] = useState(false);
  const [performance, setPerformance] = useState<RunPerformance | null>(null);
  const [loadingPerformance, setLoadingPerformance] = useState(false);
  const [performanceError, setPerformanceError] = useState<string | null>(null);
  const latestRequestedArtifactPath = useRef<string | null>(null);

  useEffect(() => {
    setSelectedArtifactPath(null);
    setCodeArtifactsOverride(null);
    setFileContent(null);
    setFileError(null);
    latestRequestedArtifactPath.current = null;
  }, [detail?.node_id]);

  useEffect(() => {
    setPerformance(null);
    setLoadingPerformance(false);
    setPerformanceError(null);
    setCodeArtifactsOverride(null);
  }, [runId]);

  const codeArtifacts = useMemo(
    () => detail?.artifacts.filter((artifact) => artifact.exists && artifact.kind === "code") ?? [],
    [detail],
  );
  const activeCodeArtifacts = codeArtifactsOverride ?? codeArtifacts;

  const logArtifacts = useMemo(
    () =>
      detail?.artifacts.filter(
        (artifact) => artifact.exists && (artifact.kind === "log" || artifact.name.endsWith(".md")),
      ) ?? [],
    [detail],
  );

  useEffect(() => {
    if (activeTab !== "performance" || !runId) {
      return;
    }
    if (performance?.run_id === runId) {
      return;
    }

    let cancelled = false;
    const loadPerformance = async () => {
      setLoadingPerformance(true);
      setPerformanceError(null);
      try {
        const payload = await fetchRunPerformance(runId);
        if (cancelled) return;
        setPerformance(payload);
      } catch (error) {
        if (cancelled) return;
        setPerformance(null);
        setPerformanceError(error instanceof Error ? error.message : String(error));
      } finally {
        if (!cancelled) {
          setLoadingPerformance(false);
        }
      }
    };

    void loadPerformance();
    return () => {
      cancelled = true;
    };
  }, [activeTab, runId, performance?.run_id]);

  const selectedIdeaId = useMemo(() => {
    if (!detail || detail.node.type !== "idea") {
      return null;
    }
    const value = detail.metadata.idea_id;
    return typeof value === "string" ? value : null;
  }, [detail]);

  const selectedIdeaNodeId = useMemo(() => {
    if (!detail || detail.node.type !== "idea") {
      return null;
    }
    return detail.node_id;
  }, [detail]);

  const selectedWorkerName = useMemo(() => {
    if (!detail || detail.node.type !== "idea") {
      return null;
    }
    const value = detail.metadata.worker_name;
    return typeof value === "string" ? value : null;
  }, [detail]);

  const selectedIdeaPath = useMemo(() => {
    if (!detail || detail.node.type !== "idea") {
      return null;
    }
    const value = detail.metadata.idea_path;
    return typeof value === "string" ? value : null;
  }, [detail]);

  const selectedWorkerIndex = useMemo(() => {
    if (!detail || detail.node.type !== "worker") {
      return null;
    }
    const value = detail.metadata.worker_index;
    return typeof value === "number" ? value : null;
  }, [detail]);

  const loadArtifact = async (path: string) => {
    if (!runId) return;
    latestRequestedArtifactPath.current = path;
    setLoadingFile(true);
    setFileError(null);
    setSelectedArtifactPath(path);
    try {
      const content = await fetchFileContent(runId, path);
      if (latestRequestedArtifactPath.current === path) {
        setFileContent(content);
      }
    } catch (error) {
      if (latestRequestedArtifactPath.current === path) {
        setFileContent(null);
        setFileError(error instanceof Error ? error.message : String(error));
      }
    } finally {
      if (latestRequestedArtifactPath.current === path) {
        setLoadingFile(false);
      }
    }
  };

  if (!detail) {
    return <div className="empty-panel">Click a node in the graph to inspect metadata, code, and logs.</div>;
  }

  return (
    <div className="detail-panel">
      <div className="detail-header">
        <h2>{detail.node.label}</h2>
        <div className="pill">{detail.node.type}</div>
      </div>

      <div className="tab-row">
        {(["performance", "artifacts", "code", "diff", "logs"] as TabId[]).map((tab) => (
          <button
            key={tab}
            className={`tab ${activeTab === tab ? "active" : ""}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab}
          </button>
        ))}
      </div>

      <div className="detail-body">
        {activeTab === "performance" ? (
          <PerformanceOverview
            performance={performance}
            loading={loadingPerformance}
            error={performanceError}
            selectedIdeaNodeId={selectedIdeaNodeId}
            workerFilter={selectedWorkerIndex}
            onSelectIdea={onSelectIdea}
            revealedIdeaNodeIds={revealedIdeaNodeIds}
          />
        ) : null}

        {activeTab === "diff" ? (
          runId && selectedIdeaId ? (
            <ProgramDiffViewer
              runId={runId}
              ideaId={selectedIdeaId}
              ideaTitle={detail.node.label}
              workerName={selectedWorkerName}
              ideaPath={selectedIdeaPath}
            />
          ) : (
            <div className="subtle">Select an idea node to view its diff.</div>
          )
        ) : null}

        {activeTab === "artifacts" ? (
          <ArtifactBrowser
            runId={runId ?? ""}
            artifacts={detail.artifacts}
            loaded={fileContent}
            loading={loadingFile}
            error={fileError}
            selectedPath={selectedArtifactPath}
            onLoad={loadArtifact}
          />
        ) : null}

        {activeTab === "code" ? (
          <ArtifactBrowser
            runId={runId ?? ""}
            artifacts={activeCodeArtifacts}
            loaded={fileContent}
            loading={loadingFile}
            error={fileError}
            selectedPath={selectedArtifactPath}
            onLoad={loadArtifact}
          />
        ) : null}

        {activeTab === "logs" ? (
          <ArtifactBrowser
            runId={runId ?? ""}
            artifacts={logArtifacts}
            loaded={fileContent}
            loading={loadingFile}
            error={fileError}
            selectedPath={selectedArtifactPath}
            onLoad={loadArtifact}
            showAgentLensAction
          />
        ) : null}
      </div>
    </div>
  );
}
