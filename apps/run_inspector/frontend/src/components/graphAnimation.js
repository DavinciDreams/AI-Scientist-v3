export function sortIdeasForReplay(ideas) {
  return [...ideas].sort((left, right) => {
    const leftCycle = left.cycle_index;
    const rightCycle = right.cycle_index;
    if (leftCycle == null) return rightCycle == null ? 0 : 1;
    if (rightCycle == null) return -1;
    return leftCycle - rightCycle;
  });
}

export function workerNamesForLayout(ideas, fullWorkerNames) {
  if (fullWorkerNames !== undefined) return [...fullWorkerNames];
  return [...new Set(sortIdeasForReplay(ideas).map((idea) => idea.worker_name))];
}

export function rowSubtreeHeight(nodeIds, currentHeights, fixedHeights) {
  return nodeIds.reduce(
    (rowHeight, nodeId) =>
      Math.max(rowHeight, fixedHeights?.[nodeId] ?? currentHeights[nodeId] ?? 0),
    0,
  );
}

export function subtreeSpanHeight(subtreeHeight, parentRound, childRound, roundGap) {
  const roundDisplacement = Math.max(0, childRound - parentRound) * roundGap;
  return subtreeHeight + roundDisplacement;
}

export function shouldSnapAnimationTransition(previousStep, currentStep) {
  const isStarting = previousStep === null && currentStep === 0;
  const isStopping = previousStep !== null && currentStep === null;
  return isStarting || isStopping;
}
