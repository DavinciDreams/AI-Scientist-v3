"""Semantic Scholar retrieval helpers for paper references."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

SEMANTIC_SCHOLAR_GRAPH_API = "https://api.semanticscholar.org/graph/v1"
CITATION_KEY_RE = re.compile(r"@\w+\{([^,]+),")
BUILTIN_REFERENCE_CATALOG: list[dict[str, Any]] = [
    {
        "slug": "nocedal_wright_2006",
        "citation_key": "NocedalWright2006",
        "title": "Numerical Optimization",
        "year": 2006,
        "venue": "Springer",
        "source_url": "https://link.springer.com/book/10.1007/978-0-387-40065-5",
        "bibtex": """@book{NocedalWright2006,
  title = {Numerical Optimization},
  author = {Nocedal, Jorge and Wright, Stephen J.},
  year = {2006},
  edition = {2},
  publisher = {Springer}
}""",
    },
    {
        "slug": "hansen_2016_cmaes",
        "citation_key": "Hansen2016CMAES",
        "title": "The CMA Evolution Strategy: A Tutorial",
        "year": 2016,
        "venue": "arXiv",
        "source_url": "https://arxiv.org/abs/1604.00772",
        "bibtex": """@article{Hansen2016CMAES,
  title = {The {CMA} Evolution Strategy: A Tutorial},
  author = {Hansen, Nikolaus},
  journal = {arXiv preprint arXiv:1604.00772},
  year = {2016}
}""",
    },
    {
        "slug": "storn_price_1997_de",
        "citation_key": "StornPrice1997DE",
        "title": "Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces",
        "year": 1997,
        "venue": "Journal of Global Optimization",
        "source_url": "https://doi.org/10.1023/A:1008202821328",
        "bibtex": """@article{StornPrice1997DE,
  title = {Differential Evolution --- A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces},
  author = {Storn, Rainer and Price, Kenneth},
  journal = {Journal of Global Optimization},
  volume = {11},
  number = {4},
  pages = {341--359},
  year = {1997}
}""",
    },
    {
        "slug": "nelder_mead_1965",
        "citation_key": "NelderMead1965Simplex",
        "title": "A Simplex Method for Function Minimization",
        "year": 1965,
        "venue": "The Computer Journal",
        "source_url": "https://doi.org/10.1093/comjnl/7.4.308",
        "bibtex": """@article{NelderMead1965Simplex,
  title = {A Simplex Method for Function Minimization},
  author = {Nelder, John A. and Mead, Roger},
  journal = {The Computer Journal},
  volume = {7},
  number = {4},
  pages = {308--313},
  year = {1965}
}""",
    },
    {
        "slug": "byrd_1995_lbfgsb",
        "citation_key": "Byrd1995LBFGSB",
        "title": "A Limited Memory Algorithm for Bound Constrained Optimization",
        "year": 1995,
        "venue": "SIAM Journal on Scientific Computing",
        "source_url": "https://doi.org/10.1137/0916069",
        "bibtex": """@article{Byrd1995LBFGSB,
  title = {A Limited Memory Algorithm for Bound Constrained Optimization},
  author = {Byrd, Richard H. and Lu, Peihuang and Nocedal, Jorge and Zhu, Ciyou},
  journal = {SIAM Journal on Scientific Computing},
  volume = {16},
  number = {5},
  pages = {1190--1208},
  year = {1995}
}""",
    },
    {
        "slug": "jones_1998_ego",
        "citation_key": "Jones1998EGO",
        "title": "Efficient Global Optimization of Expensive Black-Box Functions",
        "year": 1998,
        "venue": "Journal of Global Optimization",
        "source_url": "https://doi.org/10.1023/A:1008306431147",
        "bibtex": """@article{Jones1998EGO,
  title = {Efficient Global Optimization of Expensive Black-Box Functions},
  author = {Jones, Donald R. and Schonlau, Matthias and Welch, William J.},
  journal = {Journal of Global Optimization},
  volume = {13},
  number = {4},
  pages = {455--492},
  year = {1998}
}""",
    },
    {
        "slug": "snoek_2012_bayesopt",
        "citation_key": "Snoek2012BayesOpt",
        "title": "Practical Bayesian Optimization of Machine Learning Algorithms",
        "year": 2012,
        "venue": "NeurIPS",
        "source_url": "https://proceedings.neurips.cc/paper/2012/hash/05311655a15b75fab86956663e1819cd-Abstract.html",
        "bibtex": """@inproceedings{Snoek2012BayesOpt,
  title = {Practical Bayesian Optimization of Machine Learning Algorithms},
  author = {Snoek, Jasper and Larochelle, Hugo and Adams, Ryan P.},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2012}
}""",
    },
    {
        "slug": "audet_hare_2017_dfo",
        "citation_key": "AudetHare2017DFO",
        "title": "Derivative-Free and Blackbox Optimization",
        "year": 2017,
        "venue": "Springer",
        "source_url": "https://link.springer.com/book/10.1007/978-3-319-68913-5",
        "bibtex": """@book{AudetHare2017DFO,
  title = {Derivative-Free and Blackbox Optimization},
  author = {Audet, Charles and Hare, Warren},
  year = {2017},
  publisher = {Springer}
}""",
    },
    {
        "slug": "hansen_2021_coco",
        "citation_key": "Hansen2021COCO",
        "title": "{COCO}: A Platform for Comparing Continuous Optimizers in a Black-Box Setting",
        "year": 2021,
        "venue": "Optimization Methods and Software",
        "source_url": "https://doi.org/10.1080/10556788.2020.1808977",
        "bibtex": """@article{Hansen2021COCO,
  title = {{COCO}: A Platform for Comparing Continuous Optimizers in a Black-Box Setting},
  author = {Hansen, Nikolaus and Auger, Anne and Finck, Steffen and Ros, Raymond},
  journal = {Optimization Methods and Software},
  volume = {36},
  number = {1},
  pages = {114--144},
  year = {2021}
}""",
    },
    {
        "slug": "kirkpatrick_1983_annealing",
        "citation_key": "Kirkpatrick1983Annealing",
        "title": "Optimization by Simulated Annealing",
        "year": 1983,
        "venue": "Science",
        "source_url": "https://doi.org/10.1126/science.220.4598.671",
        "bibtex": """@article{Kirkpatrick1983Annealing,
  title = {Optimization by Simulated Annealing},
  author = {Kirkpatrick, Scott and Gelatt, C. Daniel and Vecchi, Mario P.},
  journal = {Science},
  volume = {220},
  number = {4598},
  pages = {671--680},
  year = {1983}
}""",
    },
]


class SemanticScholarError(RuntimeError):
    """Raised when Semantic Scholar retrieval fails hard."""


def utc_now() -> str:
    """Return the current UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def resolve_s2_api_key(explicit_key: str | None = None) -> str | None:
    """Resolve s2 api key."""
    if explicit_key:
        return explicit_key.strip() or None
    env_key = os.environ.get("S2_API_KEY")
    if env_key is None:
        return None
    env_key = env_key.strip()
    return env_key or None


def semantic_scholar_key_forbidden_error(
    sample_query: str,
    *,
    api_key: str | None = None,
    timeout_seconds: int = 20,
) -> str | None:
    """Return the 403 error text when a keyed probe is forbidden, else None."""
    resolved_api_key = resolve_s2_api_key(api_key)
    if resolved_api_key is None:
        return None
    probe_query = " ".join(sample_query.split()).strip()
    if not probe_query:
        return None
    try:
        search_papers(
            probe_query,
            limit=1,
            api_key=resolved_api_key,
            timeout_seconds=timeout_seconds,
        )
    except SemanticScholarError as exc:
        message = str(exc)
        if "HTTP 403" in message:
            return message
    return None


def _as_int(value: Any) -> int | None:
    """Convert value to int form when possible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _render_progress_bar(completed: int, total: int, *, width: int = 12) -> str:
    """Render a compact ASCII progress bar."""
    safe_total = max(total, 1)
    clamped = max(0, min(completed, safe_total))
    if clamped <= 0:
        filled = 0
    else:
        filled = max(1, round(width * clamped / safe_total))
    filled = min(width, filled)
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def _format_reference_progress(
    *,
    accepted_refs: int,
    target_refs: int,
    processed_candidates: int | None = None,
    total_candidates: int | None = None,
    query_index: int | None = None,
    total_queries: int | None = None,
    stage: str | None = None,
) -> str:
    """Format a console-friendly progress line for reference retrieval."""
    remaining_refs = max(target_refs - accepted_refs, 0)
    visible_target = max(target_refs, 1)
    parts = [
        f"{_render_progress_bar(min(accepted_refs, visible_target), visible_target)}",
        f"refs {accepted_refs}/{target_refs} ({remaining_refs} left)",
    ]
    if processed_candidates is not None and total_candidates is not None:
        parts.append(f"candidates {processed_candidates}/{total_candidates}")
    if query_index is not None and total_queries is not None:
        parts.append(f"query {query_index}/{total_queries}")
    body = " | ".join(parts)
    if stage:
        return f"{stage} | {body}"
    return body


def _log_reference_progress(**kwargs: Any) -> None:
    """Emit reference retrieval progress to stderr."""
    print(
        f"[semantic_scholar] {_format_reference_progress(**kwargs)}",
        file=sys.stderr,
        flush=True,
    )


def _read_json(url: str, *, api_key: str | None, timeout_seconds: int) -> dict[str, Any]:
    """Read JSON data from disk when available."""
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(
            url=url,
            headers={
                "Accept": "application/json",
                **({"x-api-key": api_key} if api_key else {}),
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise SemanticScholarError(f"Unexpected Semantic Scholar response type for {url}")
            return payload
        except urllib.error.HTTPError as exc:
            if exc.code in {403, 429} and attempt < max_attempts:
                retry_after = exc.headers.get("Retry-After")
                try:
                    wait_seconds = float(retry_after) if retry_after is not None else 1.5 * attempt
                except ValueError:
                    wait_seconds = 1.5 * attempt
                time.sleep(max(0.5, min(wait_seconds, 20.0)))
                continue
            raise SemanticScholarError(f"Semantic Scholar HTTP {exc.code} for {url}") from exc
        except urllib.error.URLError as exc:
            raise SemanticScholarError(f"Semantic Scholar request failed for {url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise SemanticScholarError(f"Invalid JSON from Semantic Scholar for {url}: {exc}") from exc

    raise SemanticScholarError(f"Semantic Scholar request failed after retries for {url}")


def _build_search_url(query: str, *, limit: int) -> str:
    """Build search url."""
    params = urllib.parse.urlencode(
        {
            "query": query,
            "limit": str(limit),
            "fields": "paperId,title,year,venue,citationCount",
        }
    )
    return f"{SEMANTIC_SCHOLAR_GRAPH_API}/paper/search?{params}"


def _build_paper_url(paper_id: str) -> str:
    """Build paper url."""
    safe_paper_id = urllib.parse.quote(paper_id, safe="")
    params = urllib.parse.urlencode(
        {
            "fields": "paperId,title,year,venue,citationCount,citationStyles,url",
        }
    )
    return f"{SEMANTIC_SCHOLAR_GRAPH_API}/paper/{safe_paper_id}?{params}"


def search_papers(
    query: str,
    *,
    limit: int,
    api_key: str | None,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    """Handle search papers logic for this module."""
    payload = _read_json(_build_search_url(query, limit=limit), api_key=api_key, timeout_seconds=timeout_seconds)
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def fetch_paper(
    paper_id: str,
    *,
    api_key: str | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Handle fetch paper logic for this module."""
    return _read_json(_build_paper_url(paper_id), api_key=api_key, timeout_seconds=timeout_seconds)


def extract_bibtex_key(bibtex: str) -> str | None:
    """Extract bibtex key."""
    match = CITATION_KEY_RE.search(bibtex)
    if not match:
        return None
    citation_key = match.group(1).strip()
    return citation_key or None


def _normalize_queries(queries: list[str]) -> list[str]:
    """Normalize queries."""
    seen: set[str] = set()
    out: list[str] = []
    for query in queries:
        candidate = query.strip()
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _builtin_reference_rows(
    *,
    queries: list[str],
    used_citation_keys: set[str],
    max_refs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return bundled references for offline/rate-limited fallback."""
    references: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    fallback_query = queries[0] if queries else "fallback_catalog"
    for row in BUILTIN_REFERENCE_CATALOG:
        if len(references) >= max_refs:
            break
        citation_key = str(row["citation_key"])
        if citation_key in used_citation_keys:
            continue
        used_citation_keys.add(citation_key)
        ref_id = f"ref.builtin.{row['slug']}"
        ref_row = {
            "id": ref_id,
            "paper_id": f"builtin:{row['slug']}",
            "citation_key": citation_key,
            "title": row["title"],
            "year": row["year"],
            "venue": row["venue"],
            "citation_count": 0,
            "query": fallback_query,
            "source_url": row["source_url"],
            "retrieved_at": utc_now(),
            "bibtex": row["bibtex"],
        }
        references.append(ref_row)
        provenance.append({key: value for key, value in ref_row.items() if key != "bibtex"})
    return references, provenance


def is_builtin_reference_id(ref_id: object) -> bool:
    """Return whether a reference id belongs to the builtin fallback catalog."""
    return str(ref_id).startswith("ref.builtin.")


def reference_outputs_exist(output_dir: str | os.PathLike[str]) -> bool:
    """Return whether both paper reference artifacts exist in an output directory."""
    output_root = os.fspath(output_dir)
    return os.path.isfile(os.path.join(output_root, "references.bib")) and os.path.isfile(
        os.path.join(output_root, "references_provenance.json")
    )


def retrieve_bibtex_entries(
    *,
    queries: list[str],
    min_refs: int,
    max_refs: int,
    api_key: str | None = None,
    search_limit_per_query: int = 25,
    timeout_seconds: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Handle retrieve bibtex entries logic for this module."""
    if min_refs <= 0:
        raise ValueError("min_refs must be > 0")
    if max_refs < min_refs:
        raise ValueError("max_refs must be >= min_refs")
    if search_limit_per_query <= 0:
        raise ValueError("search_limit_per_query must be > 0")

    resolved_api_key = resolve_s2_api_key(api_key)
    normalized_queries = _normalize_queries(queries)
    target_refs = min_refs
    total_queries = len(normalized_queries)
    seen_paper_ids: set[str] = set()
    retrieval_errors: list[str] = []

    references: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    used_citation_keys: set[str] = set()

    for query_index, query in enumerate(normalized_queries, start=1):
        if len(references) >= target_refs:
            break
        print(
            f"[semantic_scholar] search start | query {query_index}/{total_queries} | {query}",
            file=sys.stderr,
            flush=True,
        )
        try:
            search_rows = search_papers(
                query,
                limit=search_limit_per_query,
                api_key=resolved_api_key,
                timeout_seconds=timeout_seconds,
            )
        except SemanticScholarError as exc:
            retrieval_errors.append(str(exc))
            print(
                f"[semantic_scholar] search failed | query {query_index}/{total_queries} | {query} | {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue

        query_candidates: list[dict[str, Any]] = []
        for row in search_rows:
            paper_id = row.get("paperId")
            if not isinstance(paper_id, str) or not paper_id.strip():
                continue
            paper_id = paper_id.strip()
            if paper_id in seen_paper_ids:
                continue
            query_candidates.append(
                {
                    "paper_id": paper_id,
                    "query": query,
                    "citation_count": _as_int(row.get("citationCount")) or 0,
                }
            )

        ranked_candidates = sorted(
            query_candidates,
            key=lambda item: (-int(item.get("citation_count", 0)), str(item.get("paper_id"))),
        )
        _log_reference_progress(
            accepted_refs=len(references),
            target_refs=target_refs,
            processed_candidates=0,
            total_candidates=len(ranked_candidates),
            query_index=query_index,
            total_queries=total_queries,
            stage="search complete",
        )

        processed_candidates = 0
        for candidate in ranked_candidates:
            if len(references) >= target_refs:
                break

            paper_id = str(candidate["paper_id"])
            seen_paper_ids.add(paper_id)
            try:
                details = fetch_paper(
                    paper_id,
                    api_key=resolved_api_key,
                    timeout_seconds=timeout_seconds,
                )
            except SemanticScholarError as exc:
                retrieval_errors.append(str(exc))
            else:
                citation_styles = details.get("citationStyles")
                bibtex = None
                if isinstance(citation_styles, dict):
                    maybe_bibtex = citation_styles.get("bibtex")
                    if isinstance(maybe_bibtex, str) and maybe_bibtex.strip():
                        bibtex = maybe_bibtex.strip()
                if bibtex is not None:
                    citation_key = extract_bibtex_key(bibtex)
                    if citation_key is not None and citation_key not in used_citation_keys:
                        used_citation_keys.add(citation_key)

                        detail_citation_count = _as_int(details.get("citationCount"))
                        citation_count = (
                            detail_citation_count
                            if detail_citation_count is not None
                            else int(candidate.get("citation_count", 0))
                        )

                        ref_id = f"ref.s2.{paper_id}"
                        ref_row = {
                            "id": ref_id,
                            "paper_id": paper_id,
                            "citation_key": citation_key,
                            "title": details.get("title"),
                            "year": _as_int(details.get("year")),
                            "venue": details.get("venue"),
                            "citation_count": citation_count,
                            "query": candidate["query"],
                            "source_url": _build_paper_url(paper_id),
                            "retrieved_at": utc_now(),
                            "bibtex": bibtex,
                        }
                        references.append(ref_row)
                        provenance.append(
                            {
                                "id": ref_id,
                                "paper_id": paper_id,
                                "citation_key": citation_key,
                                "title": details.get("title"),
                                "year": _as_int(details.get("year")),
                                "venue": details.get("venue"),
                                "citation_count": citation_count,
                                "query": candidate["query"],
                                "source_url": _build_paper_url(paper_id),
                                "retrieved_at": ref_row["retrieved_at"],
                            }
                        )
            processed_candidates += 1
            _log_reference_progress(
                accepted_refs=len(references),
                target_refs=target_refs,
                processed_candidates=processed_candidates,
                total_candidates=len(ranked_candidates),
                query_index=query_index,
                total_queries=total_queries,
                stage="fetching",
            )

    if len(references) < min_refs:
        error_suffix = ""
        if retrieval_errors:
            error_suffix = f"; last_error={retrieval_errors[-1]}"
        raise SemanticScholarError(
            "Semantic Scholar retrieval did not meet minimum references: "
            f"{len(references)} < {min_refs}{error_suffix}"
        )

    return references, provenance


def write_bibtex_file(references: list[dict[str, Any]], path: str | os.PathLike[str]) -> None:
    """Write bibtex file."""
    entries = []
    for row in references:
        bibtex = row.get("bibtex")
        if isinstance(bibtex, str) and bibtex.strip():
            entries.append(bibtex.strip())
    if not entries:
        raise SemanticScholarError("No BibTeX entries available to write.")

    out_path = os.fspath(path)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(entries))
        handle.write("\n")
