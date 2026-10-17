import { useEffect, useState } from "react";

export type ViewName = "runs" | "launch" | "jobs" | "inspect";

export interface Route {
  view: ViewName;
  /** Path segments after the view, e.g. the job id in `#/jobs/<id>`. */
  rest: string;
  params: URLSearchParams;
}

const VIEWS: ViewName[] = ["runs", "launch", "jobs", "inspect"];

export function parseHash(hash: string): Route {
  const raw = hash.replace(/^#\/?/, "");
  const [path, query = ""] = raw.split("?", 2);
  const [head = "", ...rest] = path.split("/");
  const view = (VIEWS as string[]).includes(head) ? (head as ViewName) : "runs";
  return { view, rest: rest.map(decodeURIComponent).join("/"), params: new URLSearchParams(query) };
}

export function href(view: ViewName, rest = "", params?: Record<string, string | null | undefined>): string {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value) query.set(key, value);
  }
  const suffix = rest ? `/${encodeURIComponent(rest)}` : "";
  const qs = query.toString();
  return `#/${view}${suffix}${qs ? `?${qs}` : ""}`;
}

export function navigate(target: string, { replace = false } = {}): void {
  if (replace) {
    history.replaceState(null, "", target);
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  } else {
    window.location.hash = target;
  }
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseHash(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

export function formatRelative(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return "—";
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return iso;
  const seconds = Math.round((now - then) / 1000);
  if (seconds < 45) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 36) return `${hours} h ago`;
  return new Date(then).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

export function formatDuration(start: string | null | undefined, end: string | null | undefined, now = Date.now()): string {
  if (!start) return "—";
  const from = Date.parse(start);
  const to = end ? Date.parse(end) : now;
  if (Number.isNaN(from) || Number.isNaN(to)) return "—";
  let seconds = Math.max(0, Math.round((to - from) / 1000));
  const hours = Math.floor(seconds / 3600);
  seconds -= hours * 3600;
  const minutes = Math.floor(seconds / 60);
  seconds -= minutes * 60;
  if (hours) return `${hours}h ${minutes}m`;
  if (minutes) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

export function phaseLabel(phase: string): string {
  const labels: Record<string, string> = {
    task_drafting: "Drafting",
    investigation: "Investigation",
    ablations: "Ablations",
    final_evaluation: "Final eval",
    posthoc: "Post-hoc",
    paper: "Paper",
    review: "Review",
    revision: "Revision",
  };
  return labels[phase] ?? phase;
}
