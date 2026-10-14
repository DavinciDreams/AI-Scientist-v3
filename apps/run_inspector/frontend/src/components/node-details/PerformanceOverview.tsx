import { useCallback, useMemo, useRef, useState } from "react";
import type { ProgramPerformance, RunPerformance } from "../../types";
import { performanceTimelineEndX, stepAfterPath } from "./performanceAnimation.js";

interface PerformanceOverviewProps {
  performance: RunPerformance | null;
  loading: boolean;
  error: string | null;
  selectedIdeaNodeId?: string | null;
  workerFilter?: number | null;
  onSelectIdea?: (nodeId: string) => void;
  /**
   * Idea node ids revealed by the GraphPanel progressive-build animation. When non-null,
   * only programs belonging to these ideas are drawn (over fixed, full-data axes), so the
   * plot fills in node-by-node in lockstep with the tree. `null` = show everything.
   */
  revealedIdeaNodeIds?: Set<string> | null;
}

interface ScoredProgramPoint {
  idea_node_id: string;
  idea_id: string;
  idea_title: string;
  program_label: string;
  program_code_path: string | null;
  worker_name: string;
  score: number;
  timestamp_unix: number | null;
  resolved_time_unix: number;
  correct: boolean | null;
  order_index: number;
}

interface ChartPoint extends ScoredProgramPoint {
  x: number;
  y: number;
  clamped: boolean;
}

interface IdeaSeries {
  ideaNodeId: string;
  ideaTitle: string;
  color: string;
  points: ChartPoint[];
}

interface BestSoFarStep {
  x: number;
  y: number;
}

const FAILURE_SENTINEL_SCORE_FLOOR = -1e8;

interface PlotGeometry {
  width: number;
  height: number;
  marginTop: number;
  marginRight: number;
  marginBottom: number;
  marginLeft: number;
  plotWidth: number;
  plotHeight: number;
  yMin: number;
  yMax: number;
  xMin: number;
  xMax: number;
  hasRealTimestamps: boolean;
  chartPoints: ChartPoint[];
  byIdea: IdeaSeries[];
  bestIdeaPoints: ChartPoint[];
  bestSoFarLine: BestSoFarStep[];
}

function colorForIdea(ideaId: string): string {
  let hash = 0;
  for (let index = 0; index < ideaId.length; index += 1) {
    hash = (hash * 31 + ideaId.charCodeAt(index)) >>> 0;
  }
  const hue = hash % 360;
  return `hsl(${hue} 72% 43%)`;
}

function sortProgramsByTime(a: ProgramPerformance, b: ProgramPerformance): number {
  const timeA = a.timestamp_unix ?? Number.POSITIVE_INFINITY;
  const timeB = b.timestamp_unix ?? Number.POSITIVE_INFINITY;
  if (timeA !== timeB) return timeA - timeB;
  if (a.worker_index !== b.worker_index) return a.worker_index - b.worker_index;
  const ideaA = a.idea_order_index ?? Number.MAX_SAFE_INTEGER;
  const ideaB = b.idea_order_index ?? Number.MAX_SAFE_INTEGER;
  if (ideaA !== ideaB) return ideaA - ideaB;
  if (a.sequence_idx !== b.sequence_idx) return a.sequence_idx - b.sequence_idx;
  return a.program_label.localeCompare(b.program_label);
}

function fallbackProgramCodePath(program: ProgramPerformance): string | null {
  if (program.program_code_path && program.program_code_path.length > 0) {
    return program.program_code_path;
  }
  if (!program.worker_name || !program.idea_dir_name) {
    return null;
  }
  const base = `${program.worker_name}/ideas/${program.idea_dir_name}`;
  const programDir =
    program.attempt_name && program.attempt_name.length > 0
      ? `${base}/${program.attempt_name}/program_${program.program_idx}`
      : `${base}/program_${program.program_idx}`;
  return `${programDir}/src/program.py`;
}

function isCatastrophicFailureSentinel(point: ScoredProgramPoint): boolean {
  return point.correct === false && point.score <= FAILURE_SENTINEL_SCORE_FLOOR;
}

function buildPlotGeometry(points: ScoredProgramPoint[]): PlotGeometry | null {
  if (points.length === 0) return null;

  const width = 1000;
  const height = 320;
  const marginTop = 16;
  const marginRight = 18;
  const marginBottom = 32;
  const marginLeft = 52;
  const plotWidth = width - marginLeft - marginRight;
  const plotHeight = height - marginTop - marginBottom;

  const scores = points.map((point) => point.score);
  const sorted = [...scores].sort((a, b) => a - b);

  let yMin: number;
  let yMax: number;

  if (sorted.length < 2) {
    yMin = sorted[0] - 0.5;
    yMax = sorted[0] + 0.5;
  } else {
    const q1Idx = (sorted.length - 1) * 0.25;
    const q3Idx = (sorted.length - 1) * 0.75;
    const q1 = sorted[Math.floor(q1Idx)] + (q1Idx % 1) * (sorted[Math.ceil(q1Idx)] - sorted[Math.floor(q1Idx)]);
    const q3 = sorted[Math.floor(q3Idx)] + (q3Idx % 1) * (sorted[Math.ceil(q3Idx)] - sorted[Math.floor(q3Idx)]);
    const iqr = q3 - q1;
    let iqrLower = q1 - 1.5 * iqr;
    let iqrUpper = q3 + 1.5 * iqr;

    // Clamp to actual data range
    iqrLower = Math.max(iqrLower, sorted[0]);
    iqrUpper = Math.min(iqrUpper, sorted[sorted.length - 1]);

    const pad = (iqrUpper - iqrLower) * 0.08 || 0.1;
    yMin = iqrLower - pad;
    yMax = iqrUpper + pad;
  }

  const times = points.map((point) => point.resolved_time_unix);
  let xMin = Math.min(...times);
  let xMax = Math.max(...times);
  if (xMin === xMax) {
    xMin -= 1;
    xMax += 1;
  }

  const xDenominator = Math.max(1e-9, xMax - xMin);
  const yDenominator = Math.max(1e-9, yMax - yMin);

  const yBottom = marginTop + plotHeight;
  const chartPoints = points.map((point) => {
    const x = marginLeft + ((point.resolved_time_unix - xMin) / xDenominator) * plotWidth;
    const rawY = marginTop + ((yMax - point.score) / yDenominator) * plotHeight;
    const clamped = rawY < marginTop || rawY > yBottom;
    const y = Math.max(marginTop, Math.min(yBottom, rawY));
    return { ...point, x, y, clamped };
  });

  const grouped = new Map<string, ChartPoint[]>();
  for (const point of chartPoints) {
    const existing = grouped.get(point.idea_node_id) ?? [];
    existing.push(point);
    grouped.set(point.idea_node_id, existing);
  }

  const byIdea: IdeaSeries[] = [...grouped.entries()]
    .map(([ideaNodeId, ideaPoints]) => ({
      ideaNodeId,
      ideaTitle: ideaPoints[0]?.idea_title ?? ideaNodeId,
      color: colorForIdea(ideaNodeId),
      points: ideaPoints,
    }))
    .sort((left, right) => left.ideaTitle.localeCompare(right.ideaTitle));

  const bestIdeaPoints = byIdea
    .map((series) => {
      const ordered = [...series.points].sort((left, right) => {
        if (left.score !== right.score) return right.score - left.score;
        return left.order_index - right.order_index;
      });
      return ordered[0] ?? null;
    })
    .filter((point): point is ChartPoint => point !== null);

  // Cumulative best score step line (sorted by time)
  const sortedByTime = [...chartPoints].sort((a, b) => a.x - b.x);
  const bestSoFarLine: BestSoFarStep[] = [];
  let runningBest = -Infinity;
  for (const pt of sortedByTime) {
    if (pt.score > runningBest) {
      runningBest = pt.score;
      const rawY = marginTop + ((yMax - runningBest) / yDenominator) * plotHeight;
      const clampedY = Math.max(marginTop, Math.min(yBottom, rawY));
      bestSoFarLine.push({ x: pt.x, y: clampedY });
    }
  }

  return {
    width,
    height,
    marginTop,
    marginRight,
    marginBottom,
    marginLeft,
    plotWidth,
    plotHeight,
    yMin,
    yMax,
    xMin,
    xMax,
    hasRealTimestamps: points.some((point) => point.timestamp_unix !== null),
    chartPoints,
    byIdea,
    bestIdeaPoints,
    bestSoFarLine,
  };
}

function pointTitle(point: ChartPoint): string {
  const timeLabel =
    point.timestamp_unix !== null
      ? new Date(point.timestamp_unix * 1000).toLocaleString()
      : "time unknown";
  const correctness = point.correct === true ? "correct" : point.correct === false ? "incorrect" : "unknown";
  return `${point.idea_title}\n${point.program_label}\nscore: ${point.score.toFixed(4)}\n${correctness}\n${timeLabel}`;
}

function bestIdeaTitle(point: ChartPoint): string {
  return `${point.idea_title}\nidea best combined_score: ${point.score.toFixed(4)}`;
}

function yTickValues(minValue: number, maxValue: number, count: number): number[] {
  if (count <= 1) return [minValue];
  const values: number[] = [];
  for (let index = 0; index < count; index += 1) {
    const ratio = index / (count - 1);
    values.push(minValue + ratio * (maxValue - minValue));
  }
  return values;
}

function xTickValues(minValue: number, maxValue: number, count: number): number[] {
  if (count <= 1) return [minValue];
  const values: number[] = [];
  for (let index = 0; index < count; index += 1) {
    const ratio = index / (count - 1);
    values.push(minValue + ratio * (maxValue - minValue));
  }
  return values;
}

function formatTimeLabel(unixSeconds: number, totalSpanSeconds: number): string {
  const date = new Date(unixSeconds * 1000);
  if (totalSpanSeconds >= 24 * 60 * 60) {
    return date.toLocaleString([], {
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  }
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function linePath(points: ChartPoint[]): string {
  if (points.length === 0) return "";
  return points.map((point, index) => `${index === 0 ? "M" : "L"}${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(" ");
}

export function PerformanceOverview({
  performance,
  loading,
  error,
  selectedIdeaNodeId = null,
  workerFilter = null,
  onSelectIdea,
  revealedIdeaNodeIds = null,
}: PerformanceOverviewProps) {
  const [hoveredPoint, setHoveredPoint] = useState<ChartPoint | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const chartRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const container = chartRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    setMousePos({ x: e.clientX - rect.left + 12, y: e.clientY - rect.top - 10 });
  }, []);

  const scoredPrograms = useMemo(() => {
    if (!performance) return [] as ScoredProgramPoint[];
    const source = performance.programs.filter((program) => {
      if (program.combined_score === null) return false;
      if (workerFilter !== null && program.worker_index !== workerFilter) return false;
      return true;
    });
    source.sort(sortProgramsByTime);
    let resolvedTime = source.find((program) => program.timestamp_unix !== null)?.timestamp_unix ?? Date.now() / 1000;
    return source.map((program, index) => {
      if (program.timestamp_unix !== null) {
        resolvedTime = program.timestamp_unix;
      } else {
        resolvedTime += 1e-3;
      }
      return {
        idea_node_id: program.idea_node_id,
        idea_id: program.idea_id,
        idea_title: program.idea_title,
        program_label: program.program_label,
        program_code_path: fallbackProgramCodePath(program),
        worker_name: program.worker_name,
        score: program.combined_score as number,
        timestamp_unix: program.timestamp_unix,
        resolved_time_unix: resolvedTime,
        correct: program.correct,
        order_index: index + 1,
      };
    });
  }, [performance, workerFilter]);

  const hiddenCatastrophicFailures = useMemo(
    () => scoredPrograms.filter(isCatastrophicFailureSentinel).length,
    [scoredPrograms],
  );

  const plottedPrograms = useMemo(
    () => scoredPrograms.filter((point) => !isCatastrophicFailureSentinel(point)),
    [scoredPrograms],
  );

  const plot = useMemo(() => buildPlotGeometry(plottedPrograms), [plottedPrograms]);

  // Overlays actually drawn. Axes/coords come from the full-data `plot` so they stay
  // fixed; when an animation reveal set is active we filter to the revealed ideas and
  // recompute the cumulative-best line from just those points (reusing each point's
  // already-computed pixel position).
  const displayed = useMemo(() => {
    if (!plot) return null;
    if (!revealedIdeaNodeIds) {
      return {
        byIdea: plot.byIdea,
        bestSoFarLine: plot.bestSoFarLine,
        bestSoFarEndX: performanceTimelineEndX(
          null,
          plot.marginLeft,
          plot.marginLeft + plot.plotWidth,
        ),
        bestIdeaPoints: plot.bestIdeaPoints,
      };
    }
    const revealed = revealedIdeaNodeIds;
    const byIdea = plot.byIdea.filter((series) => revealed.has(series.ideaNodeId));
    const bestIdeaPoints = plot.bestIdeaPoints.filter((point) => revealed.has(point.idea_node_id));
    const revealedPoints = plot.chartPoints
      .filter((point) => revealed.has(point.idea_node_id))
      .sort((a, b) => a.x - b.x);
    const bestSoFarLine: BestSoFarStep[] = [];
    let runningBest = -Infinity;
    for (const point of revealedPoints) {
      if (point.score > runningBest) {
        runningBest = point.score;
        bestSoFarLine.push({ x: point.x, y: point.y });
      }
    }
    return {
      byIdea,
      bestSoFarLine,
      bestSoFarEndX: performanceTimelineEndX(
        revealedPoints,
        plot.marginLeft,
        plot.marginLeft + plot.plotWidth,
      ),
      bestIdeaPoints,
    };
  }, [plot, revealedIdeaNodeIds]);

  if (loading) {
    return <div className="subtle">Loading performance view...</div>;
  }

  if (error) {
    return <div className="error-box">{error}</div>;
  }

  if (!performance || !plot) {
    if (hiddenCatastrophicFailures > 0) {
      return (
        <div className="subtle">
          No plottable performance data available. Hidden {hiddenCatastrophicFailures} catastrophic
          failure sentinel score{hiddenCatastrophicFailures === 1 ? "" : "s"}.
        </div>
      );
    }
    return <div className="subtle">No performance data available for this run.</div>;
  }

  const yTicks = yTickValues(plot.yMin, plot.yMax, 5);
  const xTicks = xTickValues(plot.xMin, plot.xMax, 5);
  const totalSpanSeconds = Math.max(0, plot.xMax - plot.xMin);

  // After the guard above `plot` is non-null, so `displayed` is non-null too.
  const {
    byIdea: shownByIdea,
    bestSoFarLine: shownBestSoFar,
    bestSoFarEndX: shownBestSoFarEndX,
    bestIdeaPoints: shownBestIdea,
  } = displayed ?? {
    ...plot,
    bestSoFarEndX: plot.marginLeft + plot.plotWidth,
  };

  return (
    <div className="performance-overview">
      <div className="performance-summary">
        <div className="summary-pill">points: {plot.chartPoints.length}</div>
        <div className="summary-pill">idea-best points: {plot.bestIdeaPoints.length}</div>
        <div className="summary-pill">ideas: {plot.byIdea.length}</div>
        <div className="summary-pill">score min: {plot.yMin.toFixed(4)}</div>
        <div className="summary-pill">score max: {plot.yMax.toFixed(4)}</div>
        {hiddenCatastrophicFailures > 0 && (
          <div className="summary-pill">
            hidden sentinels: {hiddenCatastrophicFailures}
          </div>
        )}
      </div>

      <div className="performance-chart-scroll" ref={chartRef} style={{ position: "relative" }}>
        <svg
          className="performance-chart"
          viewBox={`0 0 ${plot.width} ${plot.height}`}
          width="100%"
          height={plot.height}
          role="img"
          aria-label="Combined score across programs over time"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoveredPoint(null)}
        >
          <line
            x1={plot.marginLeft}
            y1={plot.height - plot.marginBottom}
            x2={plot.width - plot.marginRight}
            y2={plot.height - plot.marginBottom}
            className="axis-line"
          />
          <line
            x1={plot.marginLeft}
            y1={plot.marginTop}
            x2={plot.marginLeft}
            y2={plot.height - plot.marginBottom}
            className="axis-line"
          />

          {yTicks.map((tickValue) => {
            const y =
              plot.marginTop + ((plot.yMax - tickValue) / Math.max(1e-9, plot.yMax - plot.yMin)) * plot.plotHeight;
            return (
              <g key={`tick:${tickValue.toFixed(6)}`}>
                <line
                  x1={plot.marginLeft}
                  y1={y}
                  x2={plot.width - plot.marginRight}
                  y2={y}
                  className="grid-line"
                />
                <text x={plot.marginLeft - 8} y={y + 4} className="axis-label" textAnchor="end">
                  {tickValue.toFixed(2)}
                </text>
              </g>
            );
          })}

          {xTicks.map((tickValue) => {
            const x =
              plot.marginLeft + ((tickValue - plot.xMin) / Math.max(1e-9, plot.xMax - plot.xMin)) * plot.plotWidth;
            return (
              <g key={`xtick:${tickValue.toFixed(6)}`}>
                <line
                  x1={x}
                  y1={plot.marginTop}
                  x2={x}
                  y2={plot.height - plot.marginBottom}
                  className="grid-line"
                />
                <text x={x} y={plot.height - plot.marginBottom + 14} className="axis-label" textAnchor="middle">
                  {formatTimeLabel(tickValue, totalSpanSeconds)}
                </text>
              </g>
            );
          })}

          {/* Cumulative best score step line */}
          {shownBestSoFar.length > 0 && (
            <path
              d={stepAfterPath(shownBestSoFar, shownBestSoFarEndX)}
              fill="none"
              stroke="#10233a"
              strokeWidth={2.2}
              opacity={0.55}
            />
          )}

          {shownByIdea.map((series) => (
            <path
              key={`line:${series.ideaNodeId}`}
              d={linePath(series.points)}
              fill="none"
              stroke={series.color}
              strokeWidth={selectedIdeaNodeId === series.ideaNodeId ? 3 : 2}
              opacity={selectedIdeaNodeId === null || selectedIdeaNodeId === series.ideaNodeId ? 1 : 0.35}
            />
          ))}

          {shownByIdea.flatMap((series) =>
            series.points.map((point) => {
              const baseOpacity =
                selectedIdeaNodeId === null || selectedIdeaNodeId === series.ideaNodeId ? 1 : 0.35;
              return (
                <circle
                  key={`point:${series.ideaNodeId}:${point.order_index}:${point.program_label}`}
                  cx={point.x}
                  cy={point.y}
                  r={selectedIdeaNodeId === series.ideaNodeId ? 4.5 : 3.6}
                  className="performance-point"
                  fill={point.clamped ? "none" : series.color}
                  stroke={point.clamped ? series.color : undefined}
                  strokeWidth={point.clamped ? 1.5 : undefined}
                  strokeDasharray={point.clamped ? "2,2" : undefined}
                  opacity={point.clamped ? baseOpacity * 0.6 : baseOpacity}
                  onMouseEnter={() => setHoveredPoint(point)}
                  onMouseLeave={() => setHoveredPoint(null)}
                  onClick={() => {
                    onSelectIdea?.(point.idea_node_id);
                  }}
                />
              );
            }),
          )}

          {shownBestIdea.map((point) => {
            const radius = selectedIdeaNodeId === point.idea_node_id ? 7 : 6;
            const strokeWidth = selectedIdeaNodeId === point.idea_node_id ? 2.2 : 1.6;
            const opacity =
              selectedIdeaNodeId === null || selectedIdeaNodeId === point.idea_node_id ? 1 : 0.35;
            const pointsAttr = [
              `${point.x},${point.y - radius}`,
              `${point.x + radius},${point.y}`,
              `${point.x},${point.y + radius}`,
              `${point.x - radius},${point.y}`,
            ].join(" ");
            return (
              <polygon
                key={`idea-best:${point.idea_node_id}:${point.order_index}`}
                points={pointsAttr}
                className="idea-best-marker"
                fill="#ffffff"
                stroke={colorForIdea(point.idea_node_id)}
                strokeWidth={strokeWidth}
                opacity={opacity}
                onMouseEnter={() => setHoveredPoint(point)}
                onMouseLeave={() => setHoveredPoint(null)}
                onClick={() => {
                  onSelectIdea?.(point.idea_node_id);
                }}
              />
            );
          })}

          <text x={plot.width / 2} y={plot.height - 8} textAnchor="middle" className="axis-label">
            time
          </text>
          <text
            x={14}
            y={plot.height / 2}
            textAnchor="middle"
            className="axis-label"
            transform={`rotate(-90 14 ${plot.height / 2})`}
          >
            combined_score
          </text>
        </svg>

        {hoveredPoint && (
          <div
            className="score-evolution-tooltip-floating"
            style={{ left: mousePos.x, top: mousePos.y }}
          >
            <strong>{hoveredPoint.idea_title}</strong>
            <span>{hoveredPoint.program_label}</span>
            <span>Score: {hoveredPoint.score.toFixed(4)}{hoveredPoint.clamped ? " (outlier)" : ""}</span>
            <span>{hoveredPoint.correct === true ? "correct" : hoveredPoint.correct === false ? "incorrect" : "unknown"}</span>
            {hoveredPoint.timestamp_unix !== null && (
              <span>{new Date(hoveredPoint.timestamp_unix * 1000).toLocaleString()}</span>
            )}
          </div>
        )}
      </div>

      <div className="performance-legend">
        <div className="legend-item" title="Cumulative best score">
          <span className="legend-swatch" style={{ background: "#10233a" }} />
          <span className="legend-label">Best Score</span>
        </div>
        {shownByIdea.map((series) => (
          <div
            key={`legend:${series.ideaNodeId}`}
            className={`legend-item ${selectedIdeaNodeId === series.ideaNodeId ? "active" : ""}`}
            title={series.ideaTitle}
          >
            <span className="legend-swatch" style={{ background: series.color }} />
            <span className="legend-label">{series.ideaTitle}</span>
          </div>
        ))}
      </div>

      {hiddenCatastrophicFailures > 0 && (
        <div className="subtle">
          Catastrophic evaluator failure sentinels (`combined_score &lt;= -1e8` with
          `correct=false`) are hidden from this chart so they do not swamp the visible score range.
        </div>
      )}
    </div>
  );
}
