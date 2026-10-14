import type { IdeaTimelineEntry } from "../types";

export function sortIdeasForReplay(
  ideas: readonly IdeaTimelineEntry[],
): IdeaTimelineEntry[];

export function workerNamesForLayout(
  ideas: readonly IdeaTimelineEntry[],
  fullWorkerNames?: readonly (string | null)[],
): (string | null)[];

export function rowSubtreeHeight(
  nodeIds: readonly string[],
  currentHeights: Readonly<Record<string, number>>,
  fixedHeights?: Readonly<Record<string, number>>,
): number;

export function subtreeSpanHeight(
  subtreeHeight: number,
  parentRound: number,
  childRound: number,
  roundGap: number,
): number;

export function shouldSnapAnimationTransition(
  previousStep: number | null,
  currentStep: number | null,
): boolean;
