import assert from "node:assert/strict";
import test from "node:test";
import {
  rowSubtreeHeight,
  shouldSnapAnimationTransition,
  sortIdeasForReplay,
  subtreeSpanHeight,
  workerNamesForLayout,
} from "../src/components/graphAnimation.js";

test("replay sorts unknown cycles after numbered cycles", () => {
  const ideas = [
    { idea_id: "pending-a", cycle_index: null },
    { idea_id: "second", cycle_index: 2 },
    { idea_id: "first", cycle_index: 1 },
    { idea_id: "pending-b", cycle_index: null },
  ];

  assert.deepEqual(
    sortIdeasForReplay(ideas).map((idea) => idea.idea_id),
    ["first", "second", "pending-a", "pending-b"],
  );
});

test("partial replay keeps the full worker scaffold", () => {
  const revealedIdeas = [{ worker_name: "worker-1", cycle_index: 1 }];
  const fullWorkerNames = ["worker-1", "worker-2", "worker-3"];

  assert.deepEqual(workerNamesForLayout(revealedIdeas, fullWorkerNames), fullWorkerNames);
});

test("wrapped scaffold rows reserve full worker subtree heights", () => {
  const currentHeights = { "worker-1": 40, "worker-2": 40 };
  const fullHeights = { "worker-1": 500, "worker-2": 300 };

  assert.equal(
    rowSubtreeHeight(["worker-1", "worker-2"], currentHeights, fullHeights),
    500,
  );
});

test("pruned descendants extend their parent row span", () => {
  assert.equal(subtreeSpanHeight(306, 0, 2, 100), 506);
});

test("same-round descendants preserve their subtree height", () => {
  assert.equal(subtreeSpanHeight(306, 2, 2, 100), 306);
});

test("starting replay snaps to the collapsed scaffold", () => {
  assert.equal(shouldSnapAnimationTransition(null, 0), true);
});

test("stopping replay snaps to the full tree", () => {
  assert.equal(shouldSnapAnimationTransition(1, null), true);
});

test("successive reveal steps keep tweening", () => {
  assert.equal(shouldSnapAnimationTransition(1, 2), false);
});
