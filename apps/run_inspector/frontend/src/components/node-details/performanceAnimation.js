export function performanceTimelineEndX(revealedPoints, plotStartX, fullTimelineEndX) {
  if (revealedPoints === null) return fullTimelineEndX;
  if (revealedPoints.length === 0) return plotStartX;
  return Math.max(...revealedPoints.map((point) => point.x));
}

export function stepAfterPath(steps, timelineEndX) {
  if (steps.length === 0) return "";
  const segments = [];
  for (let index = 0; index < steps.length; index++) {
    const step = steps[index];
    if (index === 0) {
      segments.push(`M${step.x.toFixed(2)},${step.y.toFixed(2)}`);
    } else {
      segments.push(`H${step.x.toFixed(2)}`);
      segments.push(`V${step.y.toFixed(2)}`);
    }
  }
  segments.push(`H${timelineEndX.toFixed(2)}`);
  return segments.join(" ");
}
