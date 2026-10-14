interface TimelinePoint {
  x: number;
}

interface BestSoFarStep extends TimelinePoint {
  y: number;
}

export function performanceTimelineEndX(
  revealedPoints: readonly TimelinePoint[] | null,
  plotStartX: number,
  fullTimelineEndX: number,
): number;

export function stepAfterPath(
  steps: readonly BestSoFarStep[],
  timelineEndX: number,
): string;
