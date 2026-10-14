"""Tests for reference planning, enrichment, and the Semantic Scholar / OpenAlex / scholarly fetchers."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import time
import urllib.error
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from src.orchestrator import paper_reference_enrichment as pre
from src.orchestrator import reference_query_plan as rqp
from src.orchestrator import semantic_scholar as s2

SKILL_SCRIPTS = Path(__file__).resolve().parents[2] / "src/agents/skills/retrieve-references/scripts"


def _load_script_module(script_name: str) -> Any:
    module_path = SKILL_SCRIPTS / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(f"tests._{script_name}_module", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rewrite_target(target_id: str, **overrides: Any) -> dict[str, Any]:
    target: dict[str, Any] = {
        "target_id": target_id,
        "section": "introduction",
        "priority": "high",
        "citation_intent": "positioning",
        "reason": "First target",
        "anchor_text": "Anchor one",
        "search_topics": ["local learning rules review"],
    }
    target.update(overrides)
    return target


def _query(query: str, category: str = "core_method", rationale: str = "a") -> dict[str, str]:
    return {"query": query, "category": category, "rationale": rationale}


# --- draft reference gaps -----------------------------------------------------

WELL_FORMED_GAP_PAYLOADS: list[tuple[str, dict[str, Any]]] = [
    (
        "ready-targeted-rewrite",
        {
            "status": "ready",
            "summary": "The draft needs stronger comparator coverage in related work.",
            "missing_literatures": ["direct feedback alignment comparisons"],
            "rewrite_targets": [
                _rewrite_target(
                    "related-work-comparators",
                    section="related work",
                    citation_intent="comparator",
                    reason="The draft names comparator families without citing them directly.",
                    anchor_text="Related work currently mentions biologically plausible alternatives.",
                    search_topics=[
                        "direct feedback alignment biologically plausible learning",
                        "feedback alignment comparator survey",
                    ],
                )
            ],
            "failure_reason": "",
        },
    ),
    (
        "no-gaps-empty-targets",
        {
            "status": "no_gaps",
            "summary": "The draft is already adequately covered.",
            "missing_literatures": [],
            "rewrite_targets": [],
            "failure_reason": "",
        },
    ),
]


def test_gap_payload_validation_and_fail_status_feedback(tmp_path: Path) -> None:
    # accepts well-formed payloads
    for name, payload in WELL_FORMED_GAP_PAYLOADS:
        pre.validate_draft_reference_gaps_payload(payload)  # must not raise for `name`

    # rejects duplicate target ids in a ready payload
    payload = {
        "status": "ready",
        "summary": "Needs more support.",
        "rewrite_targets": [
            _rewrite_target("intro-positioning"),
            _rewrite_target(
                "intro-positioning",
                section="related work",
                priority="medium",
                citation_intent="background",
                reason="Second target",
                anchor_text="Anchor two",
                search_topics=["alternative query"],
            ),
        ],
    }
    with pytest.raises(ValueError, match="duplicates target_id"):
        pre.validate_draft_reference_gaps_payload(payload)

    # write_draft_reference_gaps_validation_feedback reports fail status
    gaps_md = tmp_path / "draft_reference_gaps.md"
    gaps_json = tmp_path / "draft_reference_gaps.json"
    gaps_md.write_text("# Gaps\n", encoding="utf-8")
    gaps_json.write_text(
        json.dumps(
            {
                "status": "fail",
                "summary": "The current draft did not expose stable paragraph anchors.",
                "missing_literatures": [],
                "rewrite_targets": [],
                "failure_reason": "Could not identify stable paragraph anchors from the draft.",
            }
        ),
        encoding="utf-8",
    )

    feedback = pre.write_draft_reference_gaps_validation_feedback(
        output_dir=tmp_path,
        feedback_json_path=tmp_path / "feedback.json",
        feedback_md_path=tmp_path / "feedback.md",
        gaps_md_path=gaps_md,
        gaps_json_path=gaps_json,
    )

    assert feedback["ok"] is False
    assert "draft reference gap analysis failed" in feedback["hard_errors"][0]


def _provenance(generated_at: str, queries: list[str], references: list[dict[str, str]]) -> str:
    return json.dumps(
        {
            "generated_at": generated_at,
            "results_dir": "/tmp/results",
            "campaign_results_dir": "/tmp/results",
            "queries": queries,
            "reference_query_plan": {"status": "ready", "queries": [{"query": q} for q in queries]},
            "count": len(references),
            "references": references,
        }
    )


def test_merge_reference_artifacts_preserves_existing_and_appends_new(tmp_path: Path) -> None:
    existing_bib = tmp_path / "existing.bib"
    existing_json = tmp_path / "existing.json"
    incremental_bib = tmp_path / "incremental.bib"
    incremental_json = tmp_path / "incremental.json"
    merged_bib = tmp_path / "merged.bib"
    existing_bib.write_text("@article{SeedPaper2024,\n  title={Seed Paper}\n}\n", encoding="utf-8")
    existing_json.write_text(
        _provenance(
            "2026-04-12T10:00:00+00:00",
            ["seed query"],
            [{"id": "ref.seed.one", "citation_key": "SeedPaper2024", "query": "seed query"}],
        ),
        encoding="utf-8",
    )
    incremental_bib.write_text(
        "@article{SeedPaper2024,\n  title={Seed Paper Duplicate}\n}\n\n\n"
        "@article{GapPaper2025,\n  title={Gap Paper}\n}\n",
        encoding="utf-8",
    )
    incremental_json.write_text(
        _provenance(
            "2026-04-12T11:00:00+00:00",
            ["gap query"],
            [
                {"id": "ref.seed.dup", "citation_key": "SeedPaper2024", "query": "gap query"},
                {"id": "ref.gap.one", "citation_key": "GapPaper2025", "query": "gap query"},
            ],
        ),
        encoding="utf-8",
    )

    payload = pre.merge_reference_artifacts(
        existing_bib_path=existing_bib,
        existing_provenance_path=existing_json,
        incremental_bib_path=incremental_bib,
        incremental_provenance_path=incremental_json,
        output_bib_path=merged_bib,
        output_provenance_path=tmp_path / "merged.json",
        results_dir=Path("/tmp/results"),
        campaign_results_dir=Path("/tmp/results"),
    )

    merged_bib_text = merged_bib.read_text(encoding="utf-8")
    assert "SeedPaper2024" in merged_bib_text
    assert "GapPaper2025" in merged_bib_text
    assert merged_bib_text.count("@article{") == 2
    assert payload["count"] == 2
    assert [row["citation_key"] for row in payload["references"]] == ["SeedPaper2024", "GapPaper2025"]
    assert payload["queries"] == ["seed query", "gap query"]
    assert len(payload["reference_query_plans"]) == 2


# --- reference query plan -----------------------------------------------------

INVALID_QUERY_PLAN_CASES: list[tuple[str, dict[str, Any], str]] = [
    (
        "duplicate-queries",
        {
            "status": "ready",
            "central_topic": "Demo topic",
            "queries": [
                _query("Error diffusion CIFAR-10", "core_method", "One"),
                _query("error diffusion   cifar-10", "benchmark_context", "Two"),
            ],
        },
        "duplicates query",
    ),
    ("fail-without-reason", {"status": "fail", "failure_reason": ""}, "requires non-empty failure_reason"),
]


def test_query_plan_validation_query_texts_and_fail_status_feedback(tmp_path: Path) -> None:
    # accepts grounded queries
    payload = {
        "status": "ready",
        "central_topic": "Error diffusion transfer from fully connected to convolutional settings",
        "core_methods": ["Error Diffusion"],
        "comparators": ["Direct Feedback Alignment"],
        "benchmarks": ["MNIST", "CIFAR-10"],
        "architectural_shifts": ["fully connected to convolutional"],
        "result_driven_themes": ["auxiliary local losses"],
        "queries": [
            _query("error diffusion biologically plausible learning", "core_method", "Core method family of the run."),
            _query(
                "direct feedback alignment biologically plausible learning",
                "baseline_comparator",
                "Closest comparator implied by fixed feedback pathways.",
            ),
        ],
        "forbidden_query_patterns": ["optimization benchmarking"],
        "failure_reason": "",
    }
    rqp.validate_reference_query_plan_payload(payload)

    # rejects invalid payloads
    for name, invalid_payload, expected_error in INVALID_QUERY_PLAN_CASES:
        with pytest.raises(ValueError, match=expected_error):
            rqp.validate_reference_query_plan_payload(invalid_payload)
            pytest.fail(name)

    # reference_query_texts preserves query order
    payload = {"queries": [_query("first query"), _query("second query", "benchmark_context", "b")]}
    assert rqp.reference_query_texts(payload) == ["first query", "second query"]

    # write_reference_query_plan_validation_feedback reports fail status
    plan_md = tmp_path / "reference_query_plan.md"
    plan_json = tmp_path / "reference_query_plan.json"
    plan_md.write_text("# Plan\n", encoding="utf-8")
    plan_json.write_text(
        json.dumps(
            {
                "status": "fail",
                "central_topic": "",
                "queries": [],
                "failure_reason": "No coherent literature neighborhood",
            }
        ),
        encoding="utf-8",
    )

    feedback = rqp.write_reference_query_plan_validation_feedback(
        output_dir=tmp_path,
        feedback_json_path=tmp_path / "feedback.json",
        feedback_md_path=tmp_path / "feedback.md",
        plan_md_path=plan_md,
        plan_json_path=plan_json,
    )

    assert feedback["ok"] is False
    assert "reference query planning failed" in feedback["hard_errors"][0]


# --- semantic scholar ---------------------------------------------------------


def _s2_paper(paper_id: str, bibtex_key: str | None = None, **overrides: object) -> dict[str, object]:
    key = bibtex_key or f"{paper_id}Key"
    paper: dict[str, object] = {
        "paperId": paper_id,
        "title": f"Title {paper_id}",
        "year": 2024,
        "venue": "Venue",
        "citationCount": 100,
        "citationStyles": {"bibtex": f"@Article{{{key},\n title={{Title {paper_id}}}\n}}"},
    }
    paper.update(overrides)
    return paper


def _s2_fetch(paper_id: str, *, api_key: str | None, timeout_seconds: int) -> dict[str, object]:
    return _s2_paper(paper_id)


def test_semantic_scholar_helpers_progress_bibtex_key_forbidden_error_and_retry() -> None:
    # _format_reference_progress reports remaining work
    line = s2._format_reference_progress(
        accepted_refs=3,
        target_refs=8,
        processed_candidates=5,
        total_candidates=12,
        query_index=2,
        total_queries=4,
        stage="fetching",
    )
    for fragment in ("fetching", "refs 3/8", "5 left", "candidates 5/12", "query 2/4", "[", "]"):
        assert fragment in line

    # extract_bibtex_key
    assert s2.extract_bibtex_key("@Article{Smith2024Paper,\n author = {Alice Smith}\n}\n") == "Smith2024Paper"

    # semantic_scholar_key_forbidden_error returns the 403 message and ignores non-403 errors
    for message, expected in (
        ("Semantic Scholar HTTP 403 for https://example.test", "HTTP 403"),
        ("Semantic Scholar HTTP 429 for https://example.test", None),
    ):
        with patch.object(s2, "search_papers", side_effect=s2.SemanticScholarError(message)):
            error = s2.semantic_scholar_key_forbidden_error(
                "direct feedback alignment", api_key="demo-key", timeout_seconds=5
            )
        if expected is None:
            assert error is None, message
        else:
            assert error is not None, message
            assert expected in error, message

    # is_builtin_reference_id
    assert s2.is_builtin_reference_id("ref.builtin.catalog")
    assert not s2.is_builtin_reference_id("ref.s2.paper")
    assert not s2.is_builtin_reference_id(None)

    # _read_json retries a transient 403
    response = Mock()
    response.read.return_value = b'{"data": []}'
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)
    headers = Message()
    headers["Retry-After"] = "0"
    http_error = urllib.error.HTTPError(url="https://example.test", code=403, msg="forbidden", hdrs=headers, fp=None)

    with (
        patch("src.orchestrator.semantic_scholar.time.sleep"),
        patch(
            "src.orchestrator.semantic_scholar.urllib.request.urlopen", side_effect=[http_error, response]
        ) as urlopen_mock,
    ):
        payload = s2._read_json("https://example.test", api_key=None, timeout_seconds=5)

    assert payload == {"data": []}
    assert urlopen_mock.call_count == 2


def test_retrieve_bibtex_entries_dedupes_stops_early_logs_progress_and_fails_when_short() -> None:
    # dedupes shared citation keys and collects provenance
    def _search(query: str, *, limit: int, api_key: str | None, timeout_seconds: int) -> list[dict[str, object]]:
        assert limit > 0
        assert timeout_seconds > 0
        return [
            {"paperId": "paper_a", "citationCount": 100},
            {"paperId": "paper_b", "citationCount": 80},
            {"paperId": "paper_c", "citationCount": 60},
        ]

    def _fetch(paper_id: str, *, api_key: str | None, timeout_seconds: int) -> dict[str, object]:
        if paper_id == "paper_c":
            return {"paperId": paper_id, "citationStyles": {}}
        # paper_a and paper_b share a citation key; the duplicate should be dropped.
        return _s2_paper(
            paper_id,
            "DupKey",
            title="Paper A" if paper_id == "paper_a" else "Paper B",
            citationCount=100 if paper_id == "paper_a" else 80,
        )

    with patch.object(s2, "search_papers", side_effect=_search), patch.object(s2, "fetch_paper", side_effect=_fetch):
        refs, provenance = s2.retrieve_bibtex_entries(queries=["query one"], min_refs=1, max_refs=3)

    assert len(refs) == 1
    assert len(provenance) == 1
    assert refs[0]["citation_key"] == "DupKey"
    assert provenance[0]["id"] == refs[0]["id"]

    # stops after enough early queries
    search_calls: list[str] = []

    def _search_early(query: str, *, limit: int, api_key: str | None, timeout_seconds: int) -> list[dict[str, object]]:
        search_calls.append(query)
        assert limit == 25
        if query == "query one":
            return [{"paperId": "paper_a", "citationCount": 100}, {"paperId": "paper_b", "citationCount": 90}]
        pytest.fail(f"Unexpected later query: {query}")

    with patch.object(s2, "search_papers", side_effect=_search_early), patch.object(s2, "fetch_paper", side_effect=_s2_fetch):
        refs, provenance = s2.retrieve_bibtex_entries(queries=["query one", "query two"], min_refs=2, max_refs=2)

    assert search_calls == ["query one"]
    assert len(refs) == 2
    assert len(provenance) == 2

    # logs progress to stderr
    stderr = io.StringIO()
    results = [{"paperId": "paper_a", "citationCount": 100}, {"paperId": "paper_b", "citationCount": 90}]

    with (
        contextlib.redirect_stderr(stderr),
        patch.object(s2, "search_papers", return_value=results),
        patch.object(s2, "fetch_paper", side_effect=_s2_fetch),
    ):
        refs, provenance = s2.retrieve_bibtex_entries(queries=["query one"], min_refs=2, max_refs=2)

    assert len(refs) == 2
    assert len(provenance) == 2
    output = stderr.getvalue()
    for fragment in ("[semantic_scholar]", "refs 2/2", "candidates 2/2", "query 1/1"):
        assert fragment in output

    # fails when under the minimum
    with (
        patch.object(s2, "search_papers", return_value=[{"paperId": "paper_a", "citationCount": 10}]),
        patch.object(s2, "fetch_paper", return_value={"paperId": "paper_a", "citationStyles": {}}),
        patch.object(s2, "BUILTIN_REFERENCE_CATALOG", []),
        pytest.raises(s2.SemanticScholarError),
    ):
        s2.retrieve_bibtex_entries(queries=["one"], min_refs=1, max_refs=2)

    # fails on rate limit without injecting the builtin catalog
    with (
        patch.object(s2, "search_papers", side_effect=s2.SemanticScholarError("Semantic Scholar HTTP 429")),
        pytest.raises(s2.SemanticScholarError),
    ):
        s2.retrieve_bibtex_entries(queries=["rate limited query"], min_refs=3, max_refs=3)


def test_reference_outputs_exist_requires_both_files_and_write_bibtex_file_writes_entries(tmp_path: Path) -> None:
    # reference_outputs_exist requires both files
    assert not s2.reference_outputs_exist(tmp_path)

    (tmp_path / "references.bib").write_text("@article{RefKey}\n", encoding="utf-8")
    assert not s2.reference_outputs_exist(tmp_path)

    (tmp_path / "references_provenance.json").write_text('{"references":[]}', encoding="utf-8")
    assert s2.reference_outputs_exist(tmp_path)

    # write_bibtex_file writes entries
    out = tmp_path / "written" / "references.bib"
    out.parent.mkdir()

    s2.write_bibtex_file(
        [{"bibtex": "@Article{KeyOne,\n title={A}\n}"}, {"bibtex": "@Article{KeyTwo,\n title={B}\n}"}], out
    )

    text = out.read_text(encoding="utf-8")
    assert "KeyOne" in text
    assert "KeyTwo" in text


# --- openalex_fetch.py --------------------------------------------------------


@pytest.fixture(scope="module")
def openalex() -> Any:
    return _load_script_module("openalex_fetch")


AUTHOR_NAME_CASES = [
    (
        "falls-back-when-display-name-is-non-latin",
        [
            {"author": {"display_name": "Т. Romanova"}, "raw_author_name": "Tetyana Romanova"},
            {"author": {"display_name": "Т. Romanova"}},
            {"author": {"display_name": "Christian Köcher"}, "raw_author_name": "C. Kocher"},
        ],
        ["Tetyana Romanova", "Т. Romanova", "Christian Köcher"],
    ),
    (
        "prefers-display-names-in-authorship-order",
        [
            {"author": {"display_name": "Yann LeCun"}, "raw_author_name": "Y. LeCun"},
            {"author": {"display_name": "Yoshua Bengio"}, "raw_author_name": "Y. Bengio"},
            {"author": {}, "raw_author_name": "Geoffrey Hinton"},
        ],
        ["Yann LeCun", "Yoshua Bengio", "Geoffrey Hinton"],
    ),
]


def test_openalex_author_names_and_bibtex_author_upsert(openalex: Any) -> None:
    # author_names_from_work
    for name, authorships, expected in AUTHOR_NAME_CASES:
        assert openalex.author_names_from_work({"authorships": authorships}) == expected, name

    # upsert_bibtex_author_field replaces an existing author value
    bibtex = "@article{LeCun_2015, title={Deep learning}, journal={Nature}, author={Wrong Author}, year={2015} }"

    updated = openalex.upsert_bibtex_author_field(bibtex, ["Yann LeCun", "Yoshua Bengio", "Geoffrey E. Hinton"])

    assert "author={Yann LeCun and Yoshua Bengio and Geoffrey E. Hinton}" in updated
    assert "Wrong Author" not in updated


def _run_openalex_main(openalex: Any, tmp_path: Path, work: dict[str, object], bibtex: str) -> tuple[str, str]:
    out_bib = tmp_path / "references.bib"
    argv = [
        "openalex_fetch.py",
        "--query", "deep learning",
        "--min-refs", "1",
        "--max-refs", "1",
        "--out-bib", str(out_bib),
        "--out-json", str(tmp_path / "references.json"),
    ]
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    with (
        patch.object(openalex, "search_openalex", return_value=[work]),
        patch.object(openalex, "bibtex_for_doi", return_value=bibtex),
        patch.object(sys, "argv", argv),
        contextlib.redirect_stdout(stdout_buffer),
        contextlib.redirect_stderr(stderr_buffer),
    ):
        assert openalex.main() == 0

    return out_bib.read_text(encoding="utf-8"), stderr_buffer.getvalue()


def test_openalex_main_writes_authors_into_bibtex_and_logs_raw_results(openalex: Any, tmp_path: Path) -> None:
    # writes OpenAlex authors into the bibtex
    authored = tmp_path / "authored"
    authored.mkdir()
    work = {
        "id": "https://openalex.org/W1",
        "display_name": "Deep learning",
        "doi": "https://doi.org/10.1038/nature14539",
        "publication_year": 2015,
        "authorships": [
            {"author": {"display_name": "Yann LeCun"}},
            {"author": {"display_name": "Yoshua Bengio"}},
            {"author": {"display_name": "Geoffrey E. Hinton"}},
        ],
    }

    written_bib, _ = _run_openalex_main(
        openalex,
        authored,
        work,
        "@article{LeCun_2015, title={Deep learning}, author={Wrong Author}, journal={Nature}, year={2015} }",
    )

    assert "author={Yann LeCun and Yoshua Bengio and Geoffrey E. Hinton}" in written_bib

    # logs raw results to stderr
    logged = tmp_path / "logged"
    logged.mkdir()
    work = {
        "id": "https://openalex.org/W1",
        "display_name": "Deep learning",
        "doi": "https://doi.org/10.1038/nature14539",
        "authorships": [],
    }

    _, stderr_text = _run_openalex_main(
        openalex, logged, work, "@article{LeCun_2015, title={Deep learning}, year={2015} }"
    )

    assert "[openalex_fetch] raw results | query=deep learning" in stderr_text
    assert '"display_name": "Deep learning"' in stderr_text


# --- scholarly_fetch.py -------------------------------------------------------


def _fresh_scholarly() -> Any:
    """Fresh module per section: configure_proxy overwrites module globals."""
    module = _load_script_module("scholarly_fetch")
    module.ProxyGenerator = Mock()
    module.SCHOLARLY_CLIENT = Mock()
    return module


@pytest.fixture
def scholarly() -> Any:
    return _fresh_scholarly()


def _proxy_args(
    proxy_mode: str, http: str | None = None, https: str | None = None, timeout: float | None = 1.0
) -> SimpleNamespace:
    return SimpleNamespace(
        proxy_mode=proxy_mode, proxy_http=http, proxy_https=https, proxy_setup_timeout_seconds=timeout
    )


def _install_proxy_generator(module: Any, method: str, outcome: object) -> Mock:
    """Install a ProxyGenerator factory whose `method` returns `outcome` (or raises it)."""
    proxy_generator = Mock()
    if isinstance(outcome, Exception):
        getattr(proxy_generator, method).side_effect = outcome
    else:
        getattr(proxy_generator, method).return_value = outcome
    module.ProxyGenerator = Mock(return_value=proxy_generator)
    return proxy_generator


def test_installed_scholarly_supports_proxy_sessions() -> None:
    from scholarly import ProxyGenerator

    proxy_generator = ProxyGenerator()
    try:
        with patch.object(proxy_generator, "_check_proxy", return_value=True), pytest.warns(DeprecationWarning):
            configured = proxy_generator.SingleProxy(
                http="http://proxy.internal:8080", https="https://proxy.internal:8443"
            )
    finally:
        proxy_generator._close_session()

    assert configured


SINGLE_PROXY_SOURCE_CASES = [
    (
        "single-explicit-urls",
        _proxy_args("single", "http://proxy.internal:8080", "https://proxy.internal:8443", timeout=None),
        {},
        {"http": "http://proxy.internal:8080", "https": "https://proxy.internal:8443"},
    ),
    (
        "auto-environment-urls",
        _proxy_args("auto", timeout=None),
        {"SCHOLARLY_PROXY_HTTPS": "http://corp-proxy:8080"},
        {"http": "http://corp-proxy:8080", "https": "http://corp-proxy:8080"},
    ),
]


def test_configure_proxy_single_proxy_sources_and_free_proxy_generator() -> None:
    # single proxy sources (explicit flags or SCHOLARLY_PROXY_* environment), without a setup timeout
    for name, args, env, expected_call in SINGLE_PROXY_SOURCE_CASES:
        scholarly = _fresh_scholarly()
        proxy_generator = _install_proxy_generator(scholarly, "SingleProxy", True)

        assert scholarly.configure_proxy(args, env=env) == "single", name
        proxy_generator.SingleProxy.assert_called_once_with(**expected_call)
        scholarly.SCHOLARLY_CLIENT.use_proxy.assert_called_once_with(proxy_generator)

    # free proxy generator
    scholarly = _fresh_scholarly()
    proxy_generator = _install_proxy_generator(scholarly, "FreeProxies", True)

    assert scholarly.configure_proxy(_proxy_args("free"), env={}) == "free"
    proxy_generator.FreeProxies.assert_called_once_with()
    scholarly.SCHOLARLY_CLIENT.use_proxy.assert_called_once_with(proxy_generator)


NO_PROXY_CASES = [
    ("auto-ignores-ambient-environment", "auto", {"SCHOLARLY_PROXY_HTTPS": "http://ambient:8080"}),
    ("disabled", "none", {}),
    ("auto-without-proxy-config", "auto", {}),
]


def test_configure_proxy_returns_none_or_exits_without_usable_proxy() -> None:
    # returns None without proxy configuration
    for name, proxy_mode, ambient_env in NO_PROXY_CASES:
        scholarly = _fresh_scholarly()
        with patch.dict(scholarly.os.environ, ambient_env):
            mode = scholarly.configure_proxy(_proxy_args(proxy_mode), env={})

        assert mode is None, name
        scholarly.ProxyGenerator.assert_not_called()
        scholarly.SCHOLARLY_CLIENT.use_proxy.assert_not_called()

    # auto falls back to none when proxy setup fails
    scholarly = _fresh_scholarly()
    proxy_generator = _install_proxy_generator(scholarly, "SingleProxy", RuntimeError("proxy bootstrap failed"))

    mode = scholarly.configure_proxy(_proxy_args("auto"), env={"SCHOLARLY_PROXY_HTTPS": "http://corp-proxy:8080"})

    assert mode is None
    proxy_generator.SingleProxy.assert_called_once_with(http="http://corp-proxy:8080", https="http://corp-proxy:8080")
    scholarly.SCHOLARLY_CLIENT.use_proxy.assert_not_called()

    # single mode requires a proxy url
    scholarly = _fresh_scholarly()
    with pytest.raises(SystemExit) as exc_info:
        scholarly.configure_proxy(SimpleNamespace(proxy_mode="single", proxy_http=None, proxy_https=None), env={})

    assert "requires --proxy-http/--proxy-https" in str(exc_info.value)

    # redacts proxy details on failure
    scholarly = _fresh_scholarly()
    _install_proxy_generator(
        scholarly, "SingleProxy", RuntimeError("failed via http://user:secret@proxy.internal:8080")
    )

    with pytest.raises(SystemExit) as exc_info:
        scholarly.configure_proxy(_proxy_args("single", "http://user:secret@proxy.internal:8080"), env={})

    message = str(exc_info.value)
    assert "Failed to configure scholarly proxy" in message
    assert "secret" not in message
    assert "proxy.internal:8080" not in message


def test_scholarly_timeouts_and_consecutive_failure_abort(scholarly: Any) -> None:
    # _tightest_timeout uses the smallest active limit
    assert scholarly._tightest_timeout(None, 4.0, 1.5, None) == 1.5
    assert scholarly._tightest_timeout(None, None) is None

    # retrieve_references aborts after consecutive failures
    candidate = scholarly.Candidate(
        paper_id="paper-1",
        query="query one",
        title="Title",
        year=2024,
        venue="Venue",
        citation_count=10,
        source_url="https://example.com",
        rank_score=10.0,
        raw_publication={"id": "raw"},
    )

    with (
        patch.object(scholarly, "_search_candidates", return_value=[candidate]),
        patch.object(scholarly, "_bibtex_for_candidate", return_value=None),
        pytest.raises(SystemExit) as exc_info,
    ):
        scholarly.retrieve_references(
            queries=["query one"],
            max_refs=1,
            search_limit=10,
            sort_by="relevance",
            year_low=None,
            year_high=None,
            delay_seconds=0.0,
            query_timeout_seconds=1.0,
            candidate_timeout_seconds=1.0,
            total_timeout_seconds=10.0,
            max_consecutive_failures=1,
            progress_interval_seconds=0.0,
        )

    assert "Too many consecutive scholarly failures" in str(exc_info.value)

    # _run_with_timeout raises on a slow call
    if scholarly.os.name == "nt":
        pytest.skip("signal-based timeout not supported on Windows")

    with pytest.raises(scholarly.ScholarlyTimeoutError):
        scholarly._run_with_timeout(0.05, "slow op", time.sleep, 0.2)
