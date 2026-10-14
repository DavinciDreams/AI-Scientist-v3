import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { fetchIdeaTimeline } from "../api/client";
import type { IdeaTimelineEntry, PruneEvent, RunGraph } from "../types";
import {
  rowSubtreeHeight,
  shouldSnapAnimationTransition,
  sortIdeasForReplay,
  subtreeSpanHeight,
  workerNamesForLayout,
} from "./graphAnimation.js";

const NODE_WIDTH = 220;
const NODE_HEIGHT = 78;
const ROOT_NODE_HEIGHT = 40;
const HORIZONTAL_GAP = 12;
const VERTICAL_GAP = 36;
const PADDING_X = 24;
const PADDING_Y = 24;
const BOUNDARY_EXTRA = 100;

/** Maximum number of children placed in a single row before wrapping to a new one. */
const MAX_NODES_PER_ROW = 8;

/** Delay between revealing successive nodes during the progressive-build animation. */
const REVEAL_INTERVAL_MS = 600;

/** Duration of the position/opacity tween when the layout reflows between steps. */
const TWEEN_MS = 450;

// ---------------------------------------------------------------------------
// Tree types
// ---------------------------------------------------------------------------

interface TreeNodeData extends IdeaTimelineEntry {
  x: number;
  y: number;
  nodeIdx: number;
  isRoot: boolean;
  updatedBest?: boolean;
  /** Stable tree key. For idea nodes this matches the backend node ID. */
  ideaKey: string;
  isSeedNode?: boolean;
  seedSourceKey?: string;
  seedRound?: number;
}

interface TreeEdgeData {
  from: number;
  to: number;
  isImprovement: boolean;
  isInheritance?: boolean;
}

interface PruneBoundaryData {
  y: number;
  chapter: number;
}

interface TreeLayout {
  nodes: TreeNodeData[];
  edges: TreeEdgeData[];
  pruneBoundaries: PruneBoundaryData[];
  totalWidth: number;
  totalHeight: number;
  /** Per-node subtree extents (left/right of the node center), keyed by ideaKey. Reused
   *  as fixed extents across animation steps so collision spacing between sibling trees
   *  stays stable instead of drifting as one tree grows. */
  extents: Record<string, { l: number; r: number }>;
  /** Nodes that own children in the full tree, used to preserve its spacing decisions. */
  branchKeys: Set<string>;
  /** Full subtree heights keep wrapped scaffold rows fixed during partial replay. */
  subtreeHeights: Record<string, number>;
}

interface TreeLayoutOptions {
  /** Pins the root center so it stays put while children grow. */
  fixedRootCenterX?: number;
  /** Reuses full-tree extents so partially revealed sibling trees do not drift. */
  fixedExtents?: Record<string, { l: number; r: number }>;
  /** Reuses full-tree branch membership so partial layouts make identical spacing choices. */
  fixedBranchKeys?: ReadonlySet<string>;
  /** Reuses full-tree subtree heights so wrapped rows do not move as branches grow. */
  fixedSubtreeHeights?: Readonly<Record<string, number>>;
  /** Keeps the full worker scaffold present throughout a partial replay. */
  fixedWorkerNames?: readonly (string | null)[];
}

// ---------------------------------------------------------------------------
// Tree building algorithm
// ---------------------------------------------------------------------------

function didUpdateBest(idea: IdeaTimelineEntry): boolean {
  return (idea.attempts || []).some((a) => a.best_updated);
}

function ideaKey(idea: IdeaTimelineEntry): string {
  return idea.node_id ?? idea.idea_path ?? `${idea.worker_name ?? ""}:${idea.idea_id}`;
}

/** Determine which round an idea belongs to based on prune_chapter. */
function getIdeaRound(idea: IdeaTimelineEntry): number {
  if (idea.prune_chapter !== null && idea.prune_chapter !== undefined) {
    return idea.prune_chapter;
  }
  return 0;
}

function buildTree(
  ideas: IdeaTimelineEntry[],
  pruneEvents?: PruneEvent[],
  options: TreeLayoutOptions = {},
): TreeLayout {
  const {
    fixedRootCenterX,
    fixedExtents,
    fixedBranchKeys,
    fixedSubtreeHeights,
    fixedWorkerNames,
  } = options;
  // Sort ideas by cycle_index
  const sorted = sortIdeasForReplay(ideas);

  // Group ideas by worker_name
  const workerNames = workerNamesForLayout(sorted, fixedWorkerNames);
  const isMultiWorker = workerNames.length > 1;

  // Assign unique internal keys — ideas from different rounds can share the
  // same worker_name:idea_id, so we append a suffix (#2, #3, …) to duplicates
  // to prevent cycles in the tree.
  const keyCounts: Record<string, number> = {};
  const uniqueKeyByIndex: string[] = [];
  for (const idea of sorted) {
    const base = ideaKey(idea);
    const count = (keyCounts[base] ?? 0) + 1;
    keyCounts[base] = count;
    uniqueKeyByIndex.push(count > 1 ? `${base}#${count}` : base);
  }

  // Build idea lookup by unique key
  const ideaMap: Record<string, IdeaTimelineEntry> = {};
  sorted.forEach((idea, idx) => {
    ideaMap[uniqueKeyByIndex[idx]] = idea;
  });

  // Track seed nodes and their metadata
  const seedNodeInfo: Record<string, { sourceKey: string; round: number; sourceIdea: IdeaTimelineEntry | null }> = {};
  const nodeRound: Record<string, number> = {};

  /** Resolve the actual idea for a key, tracing through seed node chains. */
  function getSourceIdea(key: string): IdeaTimelineEntry | null {
    if (ideaMap[key]) return ideaMap[key];
    if (seedNodeInfo[key]) return seedNodeInfo[key].sourceIdea;
    return null;
  }

  // Build parent map per worker — keyed by stable tree node key
  const parentOf: Record<string, string> = {};
  const childrenOf: Record<string, string[]> = { root: [] };
  const currentBestSource: Record<string, string> = {};

  nodeRound["root"] = 0;

  const hasPruneEvents = pruneEvents && pruneEvents.length > 0;

  if (isMultiWorker) {
    const currentWorkerRound: Record<string, number> = {};

    for (const wn of workerNames) {
      const wkey = `worker:${wn ?? "default"}`;
      childrenOf.root.push(wkey);
      childrenOf[wkey] = [];
      currentBestSource[wn ?? "default"] = wkey;
      currentWorkerRound[wn ?? "default"] = 0;
      nodeRound[wkey] = 0;
    }

    sorted.forEach((idea, idx) => {
      const key = uniqueKeyByIndex[idx];
      const workerKey = idea.worker_name ?? "default";
      const round = getIdeaRound(idea);
      nodeRound[key] = round;

      // Insert seed nodes at round boundary crossings
      if (hasPruneEvents && round > currentWorkerRound[workerKey]) {
        for (let r = currentWorkerRound[workerKey] + 1; r <= round; r++) {
          const seedKey = `seed:${r}:${workerKey}`;
          const sourceKey = currentBestSource[workerKey];
          const sourceIdea = getSourceIdea(sourceKey);

          parentOf[seedKey] = sourceKey;
          if (!childrenOf[sourceKey]) childrenOf[sourceKey] = [];
          childrenOf[sourceKey].push(seedKey);
          childrenOf[seedKey] = [];

          seedNodeInfo[seedKey] = { sourceKey, round: r, sourceIdea };
          nodeRound[seedKey] = r;

          currentBestSource[workerKey] = seedKey;
        }
        currentWorkerRound[workerKey] = round;
      }

      const parent = currentBestSource[workerKey];
      parentOf[key] = parent;
      if (!childrenOf[parent]) childrenOf[parent] = [];
      childrenOf[parent].push(key);

      if (didUpdateBest(idea)) {
        currentBestSource[workerKey] = key;
        if (!childrenOf[key]) childrenOf[key] = [];
      }
    });
  } else {
    let bestSource = "root";
    let curRound = 0;

    sorted.forEach((idea, idx) => {
      const key = uniqueKeyByIndex[idx];
      const round = getIdeaRound(idea);
      nodeRound[key] = round;

      // Insert seed nodes at round boundary crossings
      if (hasPruneEvents && round > curRound) {
        for (let r = curRound + 1; r <= round; r++) {
          const seedKey = `seed:${r}:default`;
          const sourceKey = bestSource;
          const sourceIdea = getSourceIdea(sourceKey);

          parentOf[seedKey] = sourceKey;
          if (!childrenOf[sourceKey]) childrenOf[sourceKey] = [];
          childrenOf[sourceKey].push(seedKey);
          childrenOf[seedKey] = [];

          seedNodeInfo[seedKey] = { sourceKey, round: r, sourceIdea };
          nodeRound[seedKey] = r;

          bestSource = seedKey;
        }
        curRound = round;
      }

      parentOf[key] = bestSource;
      if (!childrenOf[bestSource]) childrenOf[bestSource] = [];
      childrenOf[bestSource].push(key);

      if (didUpdateBest(idea)) {
        bestSource = key;
        if (!childrenOf[bestSource]) childrenOf[bestSource] = [];
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Top-down layout. Children are laid out left-to-right under their parent, wrapping
  // after MAX_NODES_PER_ROW, and the row is centered under the parent.
  //
  // Spacing rule between two consecutive children:
  //   - if BOTH carry their own subtree (e.g. the per-worker trees under the root), space
  //     them by their actual horizontal extents so the trees can never collide;
  //   - otherwise (a leaf is involved — the common case inside one worker) use a FIXED
  //     stride, keeping single-worker trees tight with no ballooning gaps.
  // A node's X therefore depends only on its sibling counts/extents, never on how deep a
  // leaf-only neighbour's column is, so revealing descendants doesn't shift ancestors.
  // ---------------------------------------------------------------------------

  const SIBLING_STRIDE = NODE_WIDTH + HORIZONTAL_GAP;

  function nodeHeight(id: string): number {
    if (id === "root" || id.startsWith("worker:") || id.startsWith("seed:")) return ROOT_NODE_HEIGHT;
    return NODE_HEIGHT;
  }
  function hasKids(id: string): boolean {
    return (childrenOf[id]?.length ?? 0) > 0;
  }
  function hasLayoutKids(id: string): boolean {
    return fixedBranchKeys?.has(id) ?? hasKids(id);
  }

  // Partition a node's children into rows of at most MAX_NODES_PER_ROW.
  function rowsOf(id: string): string[][] {
    const kids = childrenOf[id] || [];
    const rows: string[][] = [];
    for (let i = 0; i < kids.length; i += MAX_NODES_PER_ROW) {
      rows.push(kids.slice(i, i + MAX_NODES_PER_ROW));
    }
    return rows;
  }

  // Post-order pass: compute each subtree's horizontal extents (left/right of the node's
  // center), its height, and the per-child offsets (dx from center, dy from top). Extents
  // can be overridden by `fixedExtents` (the full-tree extents) so collision spacing stays
  // stable across animation steps instead of drifting as a neighbouring tree grows.
  const extL: Record<string, number> = {};
  const extR: Record<string, number> = {};
  const subtreeH: Record<string, number> = {};
  const childOffset: Record<string, Record<string, { dx: number; dy: number }>> = {};
  const layoutVisiting = new Set<string>();
  const leftExtent = (id: string): number => fixedExtents?.[id]?.l ?? extL[id];
  const rightExtent = (id: string): number => fixedExtents?.[id]?.r ?? extR[id];
  const visualSubtreeHeight = (id: string, parentId: string): number =>
    subtreeSpanHeight(
      fixedSubtreeHeights?.[id] ?? subtreeH[id] ?? 0,
      nodeRound[parentId] ?? 0,
      nodeRound[id] ?? 0,
      BOUNDARY_EXTRA,
    );

  function layoutSubtree(id: string): void {
    if (extL[id] !== undefined || layoutVisiting.has(id)) return;
    layoutVisiting.add(id);

    const rows = rowsOf(id);
    rows.flat().forEach(layoutSubtree);

    const selfH = nodeHeight(id);
    const offsets: Record<string, { dx: number; dy: number }> = {};
    childOffset[id] = offsets;

    if (rows.length === 0) {
      extL[id] = NODE_WIDTH / 2;
      extR[id] = NODE_WIDTH / 2;
      subtreeH[id] = selfH;
      layoutVisiting.delete(id);
      return;
    }

    let leftMost = -NODE_WIDTH / 2; // the node itself occupies [-NW/2, +NW/2]
    let rightMost = NODE_WIDTH / 2;
    let childrenH = 0;
    let rowTopY = selfH + VERTICAL_GAP;

    rows.forEach((row, r) => {
      // 1. Lay children out left-to-right (relative coords), then center the row on 0.
      const centers: number[] = [];
      let prevCenter = 0;
      let prevRight = 0;
      let prevHasLayoutKids = false;
      row.forEach((kid, i) => {
        const kl = leftExtent(kid);
        let c: number;
        if (i === 0) {
          c = kl; // first child's left edge at 0
        } else if (prevHasLayoutKids && hasLayoutKids(kid)) {
          c = prevCenter + prevRight + HORIZONTAL_GAP + kl; // extent-based: no collision
        } else {
          c = prevCenter + SIBLING_STRIDE; // fixed stride when a leaf is involved
        }
        centers.push(c);
        prevCenter = c;
        prevRight = rightExtent(kid);
        prevHasLayoutKids = hasLayoutKids(kid);
      });

      let rowLeft = Infinity;
      let rowRight = -Infinity;
      row.forEach((kid, i) => {
        rowLeft = Math.min(rowLeft, centers[i] - leftExtent(kid));
        rowRight = Math.max(rowRight, centers[i] + rightExtent(kid));
      });
      const rowCenter = (rowLeft + rowRight) / 2;

      // 2. Record centered offsets and accumulate this subtree's extents/height.
      const visualSubtreeHeights = Object.fromEntries(
        row.map((kid) => [kid, visualSubtreeHeight(kid, id)]),
      );
      const rowH = rowSubtreeHeight(row, visualSubtreeHeights);
      row.forEach((kid, i) => {
        const dx = centers[i] - rowCenter;
        offsets[kid] = { dx, dy: rowTopY };
        leftMost = Math.min(leftMost, dx - leftExtent(kid));
        rightMost = Math.max(rightMost, dx + rightExtent(kid));
      });
      childrenH += rowH;
      if (r < rows.length - 1) childrenH += VERTICAL_GAP;
      rowTopY += rowH + VERTICAL_GAP;
    });

    extL[id] = -leftMost;
    extR[id] = rightMost;
    subtreeH[id] = selfH + VERTICAL_GAP + childrenH;
    layoutVisiting.delete(id);
  }
  layoutSubtree("root");

  // Pre-order placement: drop each child at its recorded offset from the parent.
  const nodeX: Record<string, number> = {};
  const nodeY: Record<string, number> = {};
  const placeVisiting = new Set<string>();
  function place(id: string, centerX: number, topY: number): void {
    if (placeVisiting.has(id)) return;
    placeVisiting.add(id);

    nodeX[id] = centerX - NODE_WIDTH / 2;
    nodeY[id] = topY;

    const offsets = childOffset[id] || {};
    for (const kid of childrenOf[id] || []) {
      const off = offsets[kid];
      if (off) place(kid, centerX + off.dx, topY + off.dy);
    }
    placeVisiting.delete(id);
  }
  // The root is pinned to `fixedRootCenterX` when given (so it stays put across animation
  // steps); otherwise it starts at a nominal center and the whole tree is normalized below.
  place("root", fixedRootCenterX ?? PADDING_X, PADDING_Y);

  // Per-round vertical offsets keep pruned rounds visually banded.
  for (const id of Object.keys(nodeY)) {
    const round = nodeRound[id] || 0;
    if (round) nodeY[id] += round * BOUNDARY_EXTRA;
  }

  // Normalize so the leftmost node sits at PADDING_X — but only for the unpinned (full)
  // layout. When the root is pinned we must NOT shift, or it would drift between steps;
  // a subset is always narrower than the full tree, so it never crosses PADDING_X anyway.
  if (fixedRootCenterX === undefined) {
    let minX = Infinity;
    for (const x of Object.values(nodeX)) minX = Math.min(minX, x);
    const shift = PADDING_X - minX;
    if (shift !== 0) {
      for (const id of Object.keys(nodeX)) nodeX[id] += shift;
    }
  }

  // 4. Build positioned node list (pre-order for rendering)
  const nodes: TreeNodeData[] = [];
  const nodeIndexById: Record<string, number> = {};

  const emitted = new Set<string>();
  function emitNode(id: string) {
    if (emitted.has(id)) {
      return;
    }
    emitted.add(id);
    const x = nodeX[id];
    const y = nodeY[id];
    const nodeIdx = nodes.length;
    nodeIndexById[id] = nodeIdx;

    if (id === "root" || id.startsWith("worker:")) {
      const displayName = id === "root" ? "Seed Program" : id.replace("worker:", "");
      nodes.push({
        node_id: null,
        idea_id: id,
        idea_path: null,
        ideaKey: id,
        title: displayName,
        x,
        y,
        nodeIdx,
        isRoot: true,
        cycle_index: null,
        timestamp: null,
        hypothesis: null,
        implementation_plan: null,
        notes_for_agent: null,
        target_rubrics: [],
        num_attempts: 0,
        best_score: null,
        any_pass: false,
        rubric_summary: null,
        attempts: [],
        worker_name: id === "root" ? null : id.replace("worker:", ""),
        prune_chapter: null,
      } as TreeNodeData);
    } else if (id.startsWith("seed:")) {
      const info = seedNodeInfo[id];
      const sourceIdea = info?.sourceIdea;
      const parts: string[] = ["Seed"];
      if (sourceIdea) {
        if (sourceIdea.cycle_index !== null && sourceIdea.cycle_index !== undefined) {
          parts.push(`Cycle ${sourceIdea.cycle_index}`);
        }
        if (sourceIdea.best_score !== null && sourceIdea.best_score !== undefined) {
          parts.push(sourceIdea.best_score.toFixed(4));
        }
      }
      nodes.push({
        node_id: null,
        idea_id: id,
        idea_path: null,
        ideaKey: id,
        title: parts.join(" \u00b7 "),
        x,
        y,
        nodeIdx,
        isRoot: false,
        isSeedNode: true,
        seedSourceKey: info?.sourceKey,
        seedRound: info?.round,
        cycle_index: sourceIdea?.cycle_index ?? null,
        timestamp: null,
        hypothesis: null,
        implementation_plan: null,
        notes_for_agent: null,
        target_rubrics: [],
        num_attempts: 0,
        best_score: sourceIdea?.best_score ?? null,
        any_pass: sourceIdea?.any_pass ?? false,
        rubric_summary: null,
        attempts: [],
        worker_name: null,
        prune_chapter: null,
      } as TreeNodeData);
    } else {
      const idea = ideaMap[id];
      if (idea) {
        nodes.push({
          ...idea,
          ideaKey: id,
          x,
          y,
          nodeIdx,
          isRoot: false,
          updatedBest: didUpdateBest(idea),
        });
      }
    }

    (childrenOf[id] || []).forEach((kid) => emitNode(kid));
  }

  emitNode("root");

  // 5. Build edges
  const edges: TreeEdgeData[] = [];

  if (isMultiWorker) {
    for (const wn of workerNames) {
      const wkey = `worker:${wn ?? "default"}`;
      const fromIdx = nodeIndexById["root"];
      const toIdx = nodeIndexById[wkey];
      if (fromIdx !== undefined && toIdx !== undefined) {
        edges.push({ from: fromIdx, to: toIdx, isImprovement: false });
      }
    }
  }

  // Seed node inheritance edges
  for (const seedKey of Object.keys(seedNodeInfo)) {
    const parentId = parentOf[seedKey];
    if (parentId !== undefined) {
      const fromIdx = nodeIndexById[parentId];
      const toIdx = nodeIndexById[seedKey];
      if (fromIdx !== undefined && toIdx !== undefined) {
        edges.push({ from: fromIdx, to: toIdx, isImprovement: false, isInheritance: true });
      }
    }
  }

  // Regular idea edges
  sorted.forEach((idea, idx) => {
    const key = uniqueKeyByIndex[idx];
    const parentId = parentOf[key];
    if (parentId !== undefined) {
      const fromIdx = nodeIndexById[parentId];
      const toIdx = nodeIndexById[key];
      if (fromIdx !== undefined && toIdx !== undefined) {
        edges.push({
          from: fromIdx,
          to: toIdx,
          isImprovement: didUpdateBest(idea),
        });
      }
    }
  });

  // 6. Compute total dimensions
  let maxX = 0;
  let maxY = 0;
  nodes.forEach((n) => {
    const right = n.x + NODE_WIDTH;
    const bottom = n.y + nodeHeight(n.ideaKey);
    if (right > maxX) maxX = right;
    if (bottom > maxY) maxY = bottom;
  });

  // 7. Compute prune boundary lines between rounds
  const pruneBoundaries: PruneBoundaryData[] = [];
  if (hasPruneEvents) {
    for (const pe of pruneEvents) {
      const roundAbove = pe.chapter;
      const roundBelow = pe.chapter + 1;

      const aboveNodes = nodes.filter((n) => {
        const r = nodeRound[n.ideaKey];
        return r !== undefined && r <= roundAbove && !n.isRoot;
      });
      const belowNodes = nodes.filter((n) => {
        const r = nodeRound[n.ideaKey];
        return r !== undefined && r >= roundBelow;
      });

      if (aboveNodes.length > 0 && belowNodes.length > 0) {
        const maxAboveBottom = Math.max(
          ...aboveNodes.map((n) => n.y + nodeHeight(n.ideaKey)),
        );
        const minBelowTop = Math.min(...belowNodes.map((n) => n.y));
        pruneBoundaries.push({
          y: (maxAboveBottom + minBelowTop) / 2,
          chapter: pe.chapter,
        });
      }
    }
  }

  const extents: Record<string, { l: number; r: number }> = {};
  for (const id of Object.keys(extL)) extents[id] = { l: extL[id], r: extR[id] };

  return {
    nodes,
    edges,
    pruneBoundaries,
    totalWidth: maxX + PADDING_X,
    totalHeight: maxY + PADDING_Y,
    extents,
    branchKeys: new Set(Object.keys(childrenOf).filter(hasKids)),
    subtreeHeights: { ...subtreeH },
  };
}

// ---------------------------------------------------------------------------
// SVG sub-components
// ---------------------------------------------------------------------------

function TreeEdge({
  fromNode,
  toNode,
  isImprovement,
  isInheritance,
}: {
  fromNode: TreeNodeData;
  toNode: TreeNodeData;
  isImprovement: boolean;
  isInheritance?: boolean;
}) {
  const fromHeight = fromNode.isRoot || fromNode.isSeedNode ? ROOT_NODE_HEIGHT : NODE_HEIGHT;
  const x1 = fromNode.x + NODE_WIDTH / 2;
  const y1 = fromNode.y + fromHeight;
  const x2 = toNode.x + NODE_WIDTH / 2;
  const y2 = toNode.y;

  // Use control points that produce a natural tangent at the endpoint.
  // Blend toward x2 so the arrow follows the curve direction on angled edges.
  const dy = y2 - y1;
  const cp1x = x1 + (x2 - x1) * 0.15;
  const cp1y = y1 + dy * 0.5;
  const cp2x = x2 - (x2 - x1) * 0.15;
  const cp2y = y2 - dy * 0.3;

  const stroke = isInheritance
    ? "var(--tree-seed-border, #8b5cf6)"
    : isImprovement
      ? "var(--tree-green)"
      : "var(--tree-border)";
  const strokeWidth = isInheritance ? 2 : isImprovement ? 2 : 1.2;
  const opacity = isInheritance ? 0.8 : isImprovement ? 0.9 : 0.5;
  const markerEnd = isInheritance
    ? "url(#arrowhead-seed)"
    : isImprovement
      ? "url(#arrowhead-green)"
      : "url(#arrowhead)";

  return (
    <path
      d={`M ${x1} ${y1} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${x2} ${y2}`}
      fill="none"
      stroke={stroke}
      strokeWidth={strokeWidth}
      opacity={opacity}
      markerEnd={markerEnd}
      strokeDasharray={isInheritance ? "6 3" : undefined}
    />
  );
}

function PruneBoundary({
  y,
  chapter,
  totalWidth,
}: {
  y: number;
  chapter: number;
  totalWidth: number;
}) {
  return (
    <g>
      <line
        x1={PADDING_X}
        y1={y}
        x2={totalWidth - PADDING_X}
        y2={y}
        stroke="var(--tree-prune-border, #f59e0b)"
        strokeWidth={2}
        strokeDasharray="8 4"
        opacity={0.7}
      />
      <text
        x={PADDING_X + 4}
        y={y - 6}
        fill="var(--tree-prune-border, #f59e0b)"
        fontSize={11}
        fontWeight={600}
        opacity={0.8}
      >
        Pruned — Round {chapter + 2}
      </text>
    </g>
  );
}

function RootNode({ node }: { node: TreeNodeData }) {
  return (
    <foreignObject x={node.x} y={node.y} width={NODE_WIDTH} height={ROOT_NODE_HEIGHT}>
      <div className="tree-node root">
        <div className="tree-node-title" style={{ marginBottom: 0, textAlign: "center" }}>
          {node.title}
        </div>
      </div>
    </foreignObject>
  );
}

function SeedNode({ node }: { node: TreeNodeData }) {
  return (
    <foreignObject x={node.x} y={node.y} width={NODE_WIDTH} height={ROOT_NODE_HEIGHT}>
      <div className="tree-node seed">
        <div className="tree-node-title" style={{ marginBottom: 0, textAlign: "center" }}>
          {node.title}
        </div>
      </div>
    </foreignObject>
  );
}

function IdeaNode({
  node,
  isSelected,
  onSelect,
}: {
  node: TreeNodeData;
  isSelected: boolean;
  onSelect: (node: TreeNodeData) => void;
}) {
  const scoreText =
    node.best_score !== null && node.best_score !== undefined
      ? node.best_score.toFixed(4)
      : "\u2014";

  return (
    <foreignObject x={node.x} y={node.y} width={NODE_WIDTH} height={NODE_HEIGHT}>
      <div
        className={`tree-node ${node.any_pass ? "pass" : "fail"} ${isSelected ? "selected" : ""} ${node.updatedBest ? "best-updated" : ""}`}
        onClick={() => onSelect(node)}
      >
        <div className="tree-node-header">
          <span className="tree-node-cycle">Cycle {node.cycle_index}</span>
          {node.updatedBest && <span className="tree-node-best-badge">NEW BEST</span>}
        </div>
        <div className="tree-node-title">{node.title}</div>
        <div className="tree-node-meta">
          <span>
            {node.num_attempts} attempt{node.num_attempts !== 1 ? "s" : ""}
          </span>
          <span>Score: {scoreText}</span>
          <span className={`badge ${node.any_pass ? "pass" : "fail"}`}>
            {node.any_pass ? "PASS" : "FAIL"}
          </span>
          {node.rubric_summary && (
            <span>
              {node.rubric_summary.passed}/{node.rubric_summary.total}
            </span>
          )}
        </div>
      </div>
    </foreignObject>
  );
}

// ---------------------------------------------------------------------------
// Main panel component
// ---------------------------------------------------------------------------

interface IdeaTreePanelProps {
  runId: string | null;
  graph: RunGraph | null;
  selectedNodeId: string | null;
  onSelectNode: (nodeId: string) => void;
  /**
   * Reports the set of idea node ids revealed so far during the progressive-build
   * animation, so sibling panels (e.g. the performance plot) can reveal in lockstep.
   * `null` means "not animating — show everything".
   */
  onRevealChange?: (revealed: Set<string> | null) => void;
}

export function GraphPanel({ runId, graph, selectedNodeId, onSelectNode, onRevealChange }: IdeaTreePanelProps) {
  const [ideas, setIdeas] = useState<IdeaTimelineEntry[] | null>(null);
  const [pruneEvents, setPruneEvents] = useState<PruneEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selectedIdeaKey, setSelectedIdeaKey] = useState<string | null>(null);
  // Progressive-build animation: `null` = not animating (show everything); otherwise
  // the number of idea nodes revealed so far (0 = only scaffolding visible).
  const [animStep, setAnimStep] = useState<number | null>(null);
  // Animated layout state. The tree is re-laid-out from the revealed subset on every
  // step; node positions/opacities are tweened toward the new layout each frame so
  // parents re-center and new nodes slide in (instead of teleporting).
  const positionsRef = useRef<Map<string, { x: number; y: number }>>(new Map());
  const opacitiesRef = useRef<Map<string, number>>(new Map());
  const rafRef = useRef<number | null>(null);
  const prevAnimStepRef = useRef<number | null>(null);
  const prevIdeasRef = useRef<IdeaTimelineEntry[] | null>(null);
  // Bumped on each tween frame to force a re-render that reads the mutable refs above.
  const [, setTweenTick] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // viewBox state
  const [viewBox, setViewBox] = useState<{ x: number; y: number; w: number; h: number } | null>(
    null,
  );
  const treeSize = useRef({ width: 0, height: 0 });
  const isPanning = useRef(false);
  const panStart = useRef({ x: 0, y: 0, vbx: 0, vby: 0 });
  const treeSizeInitialized = useRef(false);

  // Build tree idea key -> graph nodeId lookup.
  const ideaKeyToNodeId = useMemo(() => {
    const map: Record<string, string> = {};
    if (!graph) return map;
    for (const node of graph.nodes) {
      if (node.type !== "idea") continue;

      map[node.id] = node.id;

      const nIdeaPath = node.data.idea_path;
      if (typeof nIdeaPath === "string") {
        map[nIdeaPath] = node.id;
      }

      const nIdeaId = node.data.idea_id;
      const nWorker = node.data.worker_name;
      if (typeof nIdeaId === "string") {
        const key = `${typeof nWorker === "string" ? nWorker : ""}:${nIdeaId}`;
        map[key] ??= node.id;
      }
    }
    return map;
  }, [graph]);

  // Reverse map: graph nodeId -> tree ideaKey
  const nodeIdToIdeaKey = useMemo(() => {
    const map: Record<string, string> = {};
    if (!graph) return map;
    for (const node of graph.nodes) {
      if (node.type !== "idea") continue;

      const nIdeaPath = node.data.idea_path;
      if (typeof nIdeaPath === "string") {
        map[node.id] = nIdeaPath;
        continue;
      }

      const nIdeaId = node.data.idea_id;
      const nWorker = node.data.worker_name;
      map[node.id] =
        typeof nIdeaId === "string"
          ? `${typeof nWorker === "string" ? nWorker : ""}:${nIdeaId}`
          : node.id;
    }
    return map;
  }, [graph]);

  // The full-tree layout (all ideas). Used for the static view, viewbox sizing, and to
  // pin the root's X so it stays put while the animation reflows the revealed subset.
  const fullLayout = useMemo(
    () => (ideas && ideas.length > 0 ? buildTree(ideas, pruneEvents) : null),
    [ideas, pruneEvents],
  );
  const fullWorkerNames = useMemo(
    () => workerNamesForLayout(ideas ?? []),
    [ideas],
  );
  const fullRootCenterX = useMemo(() => {
    const root = fullLayout?.nodes.find((n) => n.ideaKey === "root");
    return root ? root.x + NODE_WIDTH / 2 : undefined;
  }, [fullLayout]);

  // Track whether the last selection was from an internal click (vs external e.g. performance plot)
  const lastInternalSelect = useRef(false);

  // Fetch idea timeline data when runId changes or graph is refreshed
  useEffect(() => {
    if (!runId) {
      setIdeas(null);
      setAnimStep(null);
      return;
    }
    treeSizeInitialized.current = false;
    setViewBox(null);
    setError(null);
    setAnimStep(null);
    fetchIdeaTimeline(runId)
      .then((data) => {
        setIdeas(data.ideas);
        setPruneEvents(data.prune_events ?? []);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, [runId, graph]);

  // Initialize viewBox once tree is built
  useEffect(() => {
    if (!fullLayout) return;
    if (treeSizeInitialized.current) return;
    treeSize.current = { width: fullLayout.totalWidth, height: fullLayout.totalHeight };
    treeSizeInitialized.current = true;
    setViewBox({ x: 0, y: 0, w: fullLayout.totalWidth, h: fullLayout.totalHeight });
  }, [fullLayout]);

  const handleSelect = useCallback(
    (node: TreeNodeData) => {
      lastInternalSelect.current = true;
      setSelectedIdeaKey(node.ideaKey);
      const graphLookupKey = `${node.worker_name ?? ""}:${node.idea_id}`;
      const graphNodeId =
        node.node_id ?? ideaKeyToNodeId[node.ideaKey] ?? ideaKeyToNodeId[graphLookupKey];
      if (graphNodeId) {
        onSelectNode(graphNodeId);
      }
    },
    [onSelectNode, ideaKeyToNodeId],
  );

  // Sync external selectedNodeId changes (e.g. from performance plot) to tree
  useEffect(() => {
    if (!selectedNodeId) {
      setSelectedIdeaKey(null);
      return;
    }
    if (!ideas || ideas.length === 0) return;

    // If this change came from an internal click, skip zoom — just clear the flag
    if (lastInternalSelect.current) {
      lastInternalSelect.current = false;
      return;
    }

    const key = nodeIdToIdeaKey[selectedNodeId];
    if (!key) return;

    // Find the tree node position and zoom to it.
    const tree = fullLayout ?? buildTree(ideas, pruneEvents);
    const treeNode = tree.nodes.find(
      (n) =>
        n.node_id === selectedNodeId ||
        n.ideaKey === key ||
        n.ideaKey.startsWith(key + "#"),
    );

    setSelectedIdeaKey(treeNode?.ideaKey ?? key);
    if (!treeNode) return;

    const centerX = treeNode.x + NODE_WIDTH / 2;
    const centerY = treeNode.y + NODE_HEIGHT / 2;
    // Zoom to a comfortable window around the node
    const zoomW = NODE_WIDTH * 4;
    const zoomH = NODE_HEIGHT * 5;
    setViewBox({
      x: centerX - zoomW / 2,
      y: centerY - zoomH / 2,
      w: zoomW,
      h: zoomH,
    });
  }, [selectedNodeId, ideas, pruneEvents, nodeIdToIdeaKey]);

  // Fit tree to container
  const fitToView = useCallback(() => {
    const { width, height } = treeSize.current;
    setViewBox({ x: 0, y: 0, w: width, h: height });
  }, []);

  // Progressive-build animation timer: reveal one more node every REVEAL_INTERVAL_MS,
  // then auto-reset to the full view (animStep -> null) once everything is shown.
  useEffect(() => {
    if (animStep === null) return;
    const total = ideas?.length ?? 0;
    if (animStep >= total) {
      const done = setTimeout(() => setAnimStep(null), REVEAL_INTERVAL_MS);
      return () => clearTimeout(done);
    }
    const tick = setTimeout(
      () => setAnimStep((s) => (s === null ? null : s + 1)),
      REVEAL_INTERVAL_MS,
    );
    return () => clearTimeout(tick);
  }, [animStep, ideas]);

  // Report the revealed idea node ids to the parent so the performance plot can reveal
  // in lockstep. Revealed = the first `animStep` ideas in cycle-sorted order (the same
  // subset that builds the tree layout). `null` = not animating -> show everything.
  useEffect(() => {
    if (!onRevealChange) return;
    if (animStep === null || !ideas) {
      onRevealChange(null);
      return;
    }
    const sorted = sortIdeasForReplay(ideas);
    const revealed = new Set<string>();
    for (const idea of sorted.slice(0, animStep)) {
      if (idea.node_id) revealed.add(idea.node_id);
    }
    onRevealChange(revealed);
  }, [animStep, ideas, onRevealChange]);

  // Start the progressive build, or stop it (snapping back to the full view).
  const toggleAnimation = useCallback(() => {
    setAnimStep((s) => (s === null ? 0 : null));
  }, []);

  // Target layout for the current reveal step: re-laid-out from only the revealed ideas
  // (first `animStep` in cycle order) so the tree genuinely grows, with the root pinned
  // to its full-tree position. `null` step = the full tree.
  const targetLayout = useMemo(() => {
    if (!fullLayout) return null;
    if (animStep === null) return fullLayout;
    const sorted = sortIdeasForReplay(ideas ?? []);
    // Pin the root X and reuse full-tree extents so the layout (and the spacing between
    // sibling worker-trees) stays put as deeper nodes are revealed.
    return buildTree(sorted.slice(0, animStep), pruneEvents, {
      fixedRootCenterX: fullRootCenterX,
      fixedExtents: fullLayout.extents,
      fixedBranchKeys: fullLayout.branchKeys,
      fixedSubtreeHeights: fullLayout.subtreeHeights,
      fixedWorkerNames: fullWorkerNames,
    });
  }, [fullLayout, ideas, pruneEvents, animStep, fullRootCenterX, fullWorkerNames]);

  // Tween node positions/opacities toward `targetLayout`. Existing nodes slide to their
  // new spot; nodes appearing this step emerge from their parent's current position and
  // fade in. Snaps instantly on initial load, run change, and animation start/stop.
  useLayoutEffect(() => {
    const cancelRaf = () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
    if (!targetLayout) {
      positionsRef.current = new Map();
      opacitiesRef.current = new Map();
      cancelRaf();
      return;
    }

    const toPos = new Map<string, { x: number; y: number }>();
    targetLayout.nodes.forEach((n) => toPos.set(n.ideaKey, { x: n.x, y: n.y }));
    const parentKeyOf = new Map<string, string>();
    for (const edge of targetLayout.edges) {
      const child = targetLayout.nodes[edge.to]?.ideaKey;
      const parent = targetLayout.nodes[edge.from]?.ideaKey;
      if (child && parent) parentKeyOf.set(child, parent);
    }

    const prevPos = positionsRef.current;
    const ideasChanged = prevIdeasRef.current !== ideas;
    const animationReset = shouldSnapAnimationTransition(prevAnimStepRef.current, animStep);
    const snap = prevPos.size === 0 || ideasChanged || animationReset;
    prevIdeasRef.current = ideas;
    prevAnimStepRef.current = animStep;

    if (snap) {
      cancelRaf();
      positionsRef.current = new Map(toPos);
      const op = new Map<string, number>();
      toPos.forEach((_value, key) => op.set(key, 1));
      opacitiesRef.current = op;
      setTweenTick((t) => t + 1);
      return;
    }

    // Capture start positions/opacities: existing nodes from their current displayed
    // state; new nodes from their parent's current position, transparent.
    const fromPos = new Map<string, { x: number; y: number }>();
    const fromOp = new Map<string, number>();
    toPos.forEach((tp, key) => {
      const cur = prevPos.get(key);
      if (cur) {
        fromPos.set(key, cur);
        fromOp.set(key, opacitiesRef.current.get(key) ?? 1);
      } else {
        const parentKey = parentKeyOf.get(key);
        const parentPos = parentKey ? prevPos.get(parentKey) : undefined;
        fromPos.set(key, parentPos ?? tp);
        fromOp.set(key, 0);
      }
    });

    cancelRaf();
    positionsRef.current = fromPos;
    opacitiesRef.current = fromOp;
    setTweenTick((count) => count + 1);
    const startTime = performance.now();
    const frame = (now: number) => {
      const t = Math.min(1, (now - startTime) / TWEEN_MS);
      // easeInOutQuad
      const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      const nextPos = new Map<string, { x: number; y: number }>();
      const nextOp = new Map<string, number>();
      toPos.forEach((tp, key) => {
        const f = fromPos.get(key) ?? tp;
        nextPos.set(key, { x: f.x + (tp.x - f.x) * e, y: f.y + (tp.y - f.y) * e });
        const fo = fromOp.get(key) ?? 1;
        nextOp.set(key, fo + (1 - fo) * e);
      });
      positionsRef.current = nextPos;
      opacitiesRef.current = nextOp;
      setTweenTick((c) => c + 1);
      rafRef.current = t < 1 ? requestAnimationFrame(frame) : null;
    };
    rafRef.current = requestAnimationFrame(frame);
    return cancelRaf;
  }, [targetLayout, ideas, animStep]);

  // Wheel handler: scroll → zoom toward cursor, pinch → zoom toward cursor
  // Uses a window-level listener so events from foreignObject HTML are captured.
  useEffect(() => {
    const onWheel = (e: WheelEvent) => {
      const container = containerRef.current;
      const svg = svgRef.current;
      if (!container || !svg) return;
      if (!container.contains(e.target as Node)) return;

      e.preventDefault();
      const rect = svg.getBoundingClientRect();
      const mx = (e.clientX - rect.left) / rect.width;
      const my = (e.clientY - rect.top) / rect.height;

      // Both regular scroll and pinch (ctrlKey) zoom toward the cursor.
      // Pinch gestures report smaller deltaY so use a larger multiplier.
      const speed = (e.ctrlKey || e.metaKey) ? 0.02 : 0.003;
      const zoomFactor = Math.exp(-e.deltaY * speed);

      setViewBox((vb) => {
        if (!vb) return vb;
        const newW = vb.w / zoomFactor;
        const newH = vb.h / zoomFactor;
        return {
          x: vb.x - (newW - vb.w) * mx,
          y: vb.y - (newH - vb.h) * my,
          w: newW,
          h: newH,
        };
      });
    };

    window.addEventListener("wheel", onWheel, { passive: false });
    return () => window.removeEventListener("wheel", onWheel);
  }, []); // stable — refs are checked inside the handler

  // Pan via click-drag
  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0 || !viewBox) return;
      isPanning.current = true;
      panStart.current = { x: e.clientX, y: e.clientY, vbx: viewBox.x, vby: viewBox.y };
    },
    [viewBox],
  );

  const onMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isPanning.current) return;
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      setViewBox((vb) => {
        if (!vb) return vb;
        // Use a single uniform scale matching preserveAspectRatio="xMidYMid meet"
        const scale = Math.min(rect.width / vb.w, rect.height / vb.h);
        const dx = (e.clientX - panStart.current.x) / scale;
        const dy = (e.clientY - panStart.current.y) / scale;
        return {
          ...vb,
          x: panStart.current.vbx - dx,
          y: panStart.current.vby - dy,
        };
      });
    },
    [],
  );

  const onMouseUp = useCallback(() => {
    isPanning.current = false;
  }, []);

  const zoomIn = useCallback(() => {
    setViewBox((vb) => {
      if (!vb) return vb;
      const factor = 0.8;
      const newW = vb.w * factor;
      const newH = vb.h * factor;
      return {
        x: vb.x + (vb.w - newW) / 2,
        y: vb.y + (vb.h - newH) / 2,
        w: newW,
        h: newH,
      };
    });
  }, []);

  const zoomOut = useCallback(() => {
    setViewBox((vb) => {
      if (!vb) return vb;
      const factor = 1.25;
      const newW = vb.w * factor;
      const newH = vb.h * factor;
      return {
        x: vb.x + (vb.w - newW) / 2,
        y: vb.y + (vb.h - newH) / 2,
        w: newW,
        h: newH,
      };
    });
  }, []);

  if (!runId || !graph) {
    return <div className="empty-panel">Pick a run to render idea tree.</div>;
  }
  if (error) return <div className="empty-panel">Error: {error}</div>;
  if (!ideas) return <div className="empty-panel">Loading ideas...</div>;
  if (ideas.length === 0) return <div className="empty-panel">No ideas found.</div>;

  if (!targetLayout) return <div className="empty-panel">No ideas found.</div>;

  // Render the current target structure, but read each node's position/opacity from the
  // tween refs so nodes slide/fade rather than jump as the layout reflows between steps.
  const layout = targetLayout;
  const positions = positionsRef.current;
  const displayNodes = layout.nodes.map((node) => {
    const p = positions.get(node.ideaKey);
    return p ? { ...node, x: p.x, y: p.y } : node;
  });
  const nodeOpacity = (key: string): number => opacitiesRef.current.get(key) ?? 1;
  const vb = viewBox || { x: 0, y: 0, w: layout.totalWidth, h: layout.totalHeight };

  return (
    <div className="tree-container" ref={containerRef}>
      <div className="tree-controls">
        <button
          onClick={toggleAnimation}
          title={animStep === null ? "Build tree progressively" : "Stop animation"}
        >
          {animStep === null ? "▶" : "■"}
        </button>
        <button onClick={zoomIn} title="Zoom in">
          +
        </button>
        <button onClick={zoomOut} title="Zoom out">
          &minus;
        </button>
        <button onClick={fitToView} title="Fit to view">
          &#x2922;
        </button>
      </div>
      <svg
        ref={svgRef}
        viewBox={`${vb.x} ${vb.y} ${vb.w} ${vb.h}`}
        preserveAspectRatio="xMidYMid meet"
        className="tree-svg"
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      >
        <defs>
          <marker id="arrowhead" markerWidth="5" markerHeight="4" refX="5" refY="2" orient="auto">
            <polygon points="0 0, 5 2, 0 4" fill="var(--tree-border)" opacity="0.6" />
          </marker>
          <marker
            id="arrowhead-green"
            markerWidth="5"
            markerHeight="4"
            refX="5"
            refY="2"
            orient="auto"
          >
            <polygon points="0 0, 5 2, 0 4" fill="var(--tree-green)" opacity="0.9" />
          </marker>
          <marker
            id="arrowhead-seed"
            markerWidth="5"
            markerHeight="4"
            refX="5"
            refY="2"
            orient="auto"
          >
            <polygon points="0 0, 5 2, 0 4" fill="var(--tree-seed-border, #8b5cf6)" opacity="0.8" />
          </marker>
        </defs>
        {layout.edges.map((edge, i) => (
          <g key={i} opacity={nodeOpacity(displayNodes[edge.to].ideaKey)}>
            <TreeEdge
              fromNode={displayNodes[edge.from]}
              toNode={displayNodes[edge.to]}
              isImprovement={edge.isImprovement}
              isInheritance={edge.isInheritance}
            />
          </g>
        ))}
        {layout.pruneBoundaries.map((pb) => (
          <PruneBoundary
            key={`prune-${pb.chapter}`}
            y={pb.y}
            chapter={pb.chapter}
            totalWidth={layout.totalWidth}
          />
        ))}
        {displayNodes.map((node) => (
          <g key={node.ideaKey} opacity={nodeOpacity(node.ideaKey)}>
            {node.isRoot ? (
              <RootNode node={node} />
            ) : node.isSeedNode ? (
              <SeedNode node={node} />
            ) : (
              <IdeaNode
                node={node}
                isSelected={selectedIdeaKey === node.ideaKey}
                onSelect={handleSelect}
              />
            )}
          </g>
        ))}
      </svg>
    </div>
  );
}
