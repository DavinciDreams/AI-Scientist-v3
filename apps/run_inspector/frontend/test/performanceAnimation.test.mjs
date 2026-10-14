import assert from "node:assert/strict";
import test from "node:test";
import {
  performanceTimelineEndX,
  stepAfterPath,
} from "../src/components/node-details/performanceAnimation.js";

test("progressive best line ends at the latest revealed point", () => {
  const revealedPoints = [{ x: 80 }, { x: 140 }];
  const timelineEndX = performanceTimelineEndX(revealedPoints, 40, 300);

  assert.equal(stepAfterPath([{ x: 80, y: 20 }], timelineEndX), "M80.00,20.00 H140.00");
});

test("normal best line spans the full timeline", () => {
  assert.equal(performanceTimelineEndX(null, 40, 300), 300);
});

test("empty progressive timeline ends at its plot origin", () => {
  assert.equal(performanceTimelineEndX([], 40, 300), 40);
});
