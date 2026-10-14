"""Paper pipeline scripts: scientific_evidence, validate_claims, validate_rendered_si, compile_latex,
validate_tex_refs, write_paper_variants.sh and install_autoillustrate.sh."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS = REPO_ROOT / "src" / "agents" / "skills"
SCIENTIFIC_EVIDENCE_SH = SKILLS / "scientific-evidence/scripts/scientific_evidence.sh"
SCIENTIFIC_EVIDENCE_PY = SKILLS / "scientific-evidence/scripts/scientific_evidence.py"
VALIDATE_CLAIMS = {
    "write-paper": SKILLS / "write-paper/scripts/validate_claims.py",
    "standalone": SKILLS / "write-paper-standalone/scripts/validate_claims.py",
}
VALIDATE_RENDERED_SI = SKILLS / "validate-paper/scripts/validate_rendered_si.py"
VALIDATE_TEX_REFS = SKILLS / "validate-paper/scripts/validate_tex_refs.py"
COMPILE_LATEX = SKILLS / "validate-paper/scripts/compile_latex.sh"
WRITE_PAPER_VARIANTS = REPO_ROOT / "scripts" / "write_paper_variants.sh"
INSTALL_AUTOILLUSTRATE = REPO_ROOT / "scripts" / "install_autoillustrate.sh"
DOCKERFILE = REPO_ROOT / "src" / "docker" / "agent-cli" / "Dockerfile"

AGENT_SKILL_PARTS = [
    (".claude", "skills"), (".codex", "skills"), (".cursor", "skills"),
    (".opencode", "skills"), (".gemini", "skills"), (".pi", "agent", "skills"),
]
CLAIM_TEX = ".codex/skills/payload/claim.tex"
MUST_NOT_IMPORT = "must not import agent skill files"


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_files(root: Path, files: dict[str, str], **substitutions: str) -> None:
    for relative, content in files.items():
        for token, value in substitutions.items():
            content = content.replace(token, value)
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


# =========================================================================== scientific_evidence.sh


def _evidence(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCIENTIFIC_EVIDENCE_SH), *args], check=False, capture_output=True, text=True
    )


def _report(root_dir: Path) -> dict:
    return json.loads((root_dir / "scientific-evidence" / "verification-report.json").read_text())


def _emit_workspace(root_dir: Path) -> Path:
    emit = _evidence("emit-bundle", "--root-dir", str(root_dir))
    assert emit.returncode == 0, emit.stderr
    return root_dir


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return _emit_workspace(tmp_path / "workspace")


def test_scientific_evidence_help_emit_bundle_and_verify_success(workspace: Path) -> None:
    # --help smoke
    result = _evidence("--help")
    assert result.returncode == 0, result.stderr
    for token in ("emit-bundle", "emit-schema", "verify", "--root-dir"):
        assert token in result.stdout

    # emit-bundle writes the schema bundle
    bundle_dir = workspace / "scientific-evidence"
    assert bundle_dir.is_dir()
    expected_files = {
        bundle_dir / "spec.md",
        bundle_dir / "bundle.manifest.json",
        bundle_dir / "ledger.jsonl",
        bundle_dir / "schemas" / "evidence-reference.schema.json",
        bundle_dir / "schemas" / "artifact-manifest.schema.json",
        bundle_dir / "schemas" / "ledger-entry.schema.json",
        bundle_dir / "schemas" / "verification-report.schema.json",
    }
    assert expected_files == {path for path in bundle_dir.rglob("*") if path.is_file()}
    manifest = json.loads((bundle_dir / "bundle.manifest.json").read_text())
    assert manifest["schema_name"] == "scientific-evidence.schema-bundle"
    assert manifest["ledger_path"] == "ledger.jsonl"
    spec_text = (bundle_dir / "spec.md").read_text()
    for token in (
        "@evidence[[TARGET]]", "scientific-evidence/", "### Python", "### YAML",
        "machine-checkable trail", "do not hardcode result numbers", "resolved",
    ):
        assert token in spec_text

    # verify succeeds for anchored references and a consistent manifest
    _write_files(
        workspace,
        {
            "generated/stats.json": '{"mnist": {"accuracy_mean": 96.7}}\n',
            "scripts/build_figure.py": "print('ok')\n",
            "paper.tex": "Accuracy reached 96.7\\%. % @evidence[[generated/stats.json#mnist.accuracy_mean]]\n",
            "figures/main.png.prov.json": json.dumps(
                {
                    "schema_name": "scientific-evidence.artifact-manifest",
                    "artifact": {"path": "figures/main.png", "kind": "figure"},
                    "derived_from": [
                        "@evidence[[generated/stats.json#mnist.accuracy_mean]]",
                        "@evidence[[scripts/build_figure.py]]",
                    ],
                    "generated_by": {"script_path": "scripts/build_figure.py"},
                }
            ),
        },
    )
    (workspace / "figures" / "main.png").write_bytes(b"fakepng")

    verify = _evidence("verify", "--root-dir", str(workspace))

    assert verify.returncode == 0, verify.stderr
    report = _report(workspace)
    assert report["schema_name"] == "scientific-evidence.verification-report"
    assert report["summary"]["dangling_references"] == 0
    assert report["summary"]["anchored_references"] >= 1
    assert report["summary"]["manifests_checked"] == 1


_CLEAN_PAPER = {"paper.md": "No evidence refs in the real paper.\n"}


@pytest.mark.parametrize(
    "files",
    [
        *[
            pytest.param(
                {
                    **_CLEAN_PAPER,
                    "/".join(parts) + "/validate-paper/SKILL.md":
                        "Helper docs mention @evidence[[generated/missing.json#accuracy]].\n",
                },
                id="agent-support-" + parts[0],
            )
            for parts in AGENT_SKILL_PARTS
        ],
        pytest.param(
            {
                **_CLEAN_PAPER,
                "paper_prompt.md": "Prompt boilerplate with @evidence[[...]].\n",
                "paper_repair_prompt_round_1.md":
                    "Another prompt with @evidence[[generated/missing.json#mnist.accuracy_mean]].\n",
                "validation_report.json": '{"hard_errors": ["Fix dangling `@evidence[[...]]` links."]}\n',
            },
            id="generated-prompt-files",
        ),
        pytest.param(
            {
                "paper.md": "No evidence refs in the candidate.\n",
                "previous_paper/paper.tex": "Historical @evidence[[scientific-evidence/inputs/missing.csv]].\n",
                "previous_paper/figure.png.prov.json": '{"generated_by":{"script_path":"missing.py"}}\n',
            },
            id="previous-paper-snapshot",
        ),
        pytest.param(
            {
                **_CLEAN_PAPER,
                "validation_feedback.md": "Stale feedback mentions @evidence[[generated/missing.json#acc]].\n",
                "paper_evidence_audit.md": "Old audit packet mentions @evidence[[generated/missing.json#acc]].\n",
                "paper_evidence_audit.json": '{"note": "@evidence[[generated/missing.json#acc]]"}\n',
                "validation_report.json": '{"prior_error": "@evidence[[generated/missing.json#acc]]"}\n',
                ".pi/agent/skills/validate-paper/SKILL.md": "Operational @evidence[[generated/missing.json#acc]].\n",
            },
            id="feedback-and-audit-files",
        ),
    ],
)
def test_scientific_evidence_verify_ignores_operational_files(workspace: Path, files: dict[str, str]) -> None:
    _write_files(workspace, files)

    verify = _evidence("verify", "--root-dir", str(workspace))

    assert verify.returncode == 0, verify.stderr
    summary = _report(workspace)["summary"]
    assert summary["total_references"] == 0
    assert summary["dangling_references"] == 0


@pytest.mark.parametrize(
    ("files", "expected_subjects"),
    [
        pytest.param(
            {"paper.md": "Main result 96.7. @evidence[[generated/missing.json#mnist.accuracy_mean]]\n"},
            {"paper.md"},
            id="dangling-reference",
        ),
        pytest.param(
            {
                ".codex/sections/results.tex": "Result @evidence[[generated/missing.json#accuracy]].\n",
                "paper.tex": "\\input{.codex/sections/results.tex}\n",
            },
            {".codex/sections/results.tex"},
            id="non-skill-codex-content",
        ),
        pytest.param(
            {
                "previous_paper/section.tex": "Imported @evidence[[missing.csv]].\n",
                "paper.tex": "\\input{previous_paper/section.tex}\n",
            },
            {"previous_paper/section.tex"},
            id="imported-previous-paper",
        ),
        pytest.param(
            {"results/validation_report.json": '{"source": "@evidence[[results/missing.json#accuracy]]"}\n'},
            {"results/validation_report.json"},
            id="nested-validation-report",
        ),
        pytest.param(
            {
                "results/validation_report.json": '{"claim": "@evidence[[generated/missing.json#acc]]"}\n',
                "results/.pi/data.md": "Claim @evidence[[generated/missing.json#acc]].\n",
            },
            {"results/validation_report.json", "results/.pi/data.md"},
            id="nested-operational-names",
        ),
    ],
)
def test_scientific_evidence_verify_reports_dangling_references(
    workspace: Path, files: dict[str, str], expected_subjects: set[str]
) -> None:
    _write_files(workspace, files)

    verify = _evidence("verify", "--root-dir", str(workspace))

    assert verify.returncode == 1
    report = _report(workspace)
    assert report["summary"]["total_references"] == len(expected_subjects)
    assert report["summary"]["dangling_references"] == len(expected_subjects)
    dangling = {item["subject_path"] for item in report["results"] if item["status"] == "dangling"}
    assert dangling == expected_subjects


@pytest.mark.parametrize(
    ("files", "expected_stderr"),
    [
        pytest.param(
            {
                CLAIM_TEX: "Unsupported @evidence[[generated/missing.json#accuracy]].\n",
                "paper.tex": f"\\input {CLAIM_TEX}\n",
            },
            MUST_NOT_IMPORT, id="imported-agent-skill-tex",
        ),
        pytest.param(
            {"paper.tex": "\\input{.codex/x/../skills/payload/claim.tex}\n"}, MUST_NOT_IMPORT, id="normalized-path"
        ),
        pytest.param(
            {"paper.tex": f"\\verb|%| \\input{{{CLAIM_TEX}}}\n"}, MUST_NOT_IMPORT, id="percent-in-verb"
        ),
        pytest.param(
            {"paper.tex": f"\\newcommand{{\\loadfile}}{{\\input}}\n\\loadfile{{{CLAIM_TEX}}}\n"},
            "dynamic import command", id="zero-argument-alias",
        ),
        pytest.param(
            {
                "paper.tex": "\\usepackage{loader}\n\\loadclaim\n",
                "loader.sty": f"\\newcommand{{\\loadclaim}}{{\\input{{{CLAIM_TEX}}}}}\n",
            },
            MUST_NOT_IMPORT, id="workspace-local-package",
        ),
        pytest.param(
            {"paper.tex": "\\newcommand{\\paperpkg}{loader}\n\\usepackage{\\paperpkg}\n"},
            "dynamic support import", id="dynamic-local-package",
        ),
        pytest.param(
            {"paper.tex": f"% \\begin{{verbatim}}\n\\input{{{CLAIM_TEX}}}\n% \\end{{verbatim}}\n"},
            MUST_NOT_IMPORT, id="commented-verbatim-delimiters",
        ),
        pytest.param(
            {
                "../external.tex": "Unsupported @evidence[[generated/missing.json#accuracy]].\n",
                "paper.tex": "\\input{../external.tex}\n",
            },
            "escapes the paper workspace", id="outside-workspace-import",
        ),
    ],
)
def test_scientific_evidence_verify_rejects_unsafe_tex_imports(
    workspace: Path, files: dict[str, str], expected_stderr: str
) -> None:
    _write_files(workspace, files)

    verify = _evidence("verify", "--root-dir", str(workspace))

    assert verify.returncode == 1
    assert expected_stderr in verify.stderr


def test_scientific_evidence_verify_rejects_absolute_file_and_outside_workspace_targets(
    workspace: Path, tmp_path: Path
) -> None:
    # absolute and file:// evidence targets are reported dangling without a resolved path
    (workspace / "paper.md").write_text(
        "Absolute @evidence[[/etc/passwd]] and file URI @evidence[[file:///etc/passwd]].\n", encoding="utf-8"
    )

    verify = _evidence("verify", "--root-dir", str(workspace))

    assert verify.returncode == 1
    report = _report(workspace)
    results = report["results"]
    assert report["summary"]["dangling_references"] == 2
    assert all(item["status"] == "dangling" for item in results)
    assert all("resolved_path" not in item for item in results)
    messages = "\n".join(message for item in results for message in item["messages"])
    assert "Absolute evidence paths are not allowed" in messages
    assert "file:// evidence targets are not allowed" in messages

    # manifest paths outside the workspace are rejected before hashing
    workspace = _emit_workspace(tmp_path / "manifest-workspace")
    outside_path = tmp_path / "outside.txt"
    outside_path.write_text("outside workspace\n", encoding="utf-8")
    _write_files(
        workspace,
        {
            "figures/main.png.prov.json": json.dumps(
                {
                    "schema_name": "scientific-evidence.artifact-manifest",
                    "artifact": {"path": "../outside.txt", "kind": "figure", "sha256": "0" * 64},
                    "derived_from": ["@evidence[[https://example.com/data]]"],
                    "generated_by": {"script_path": str(outside_path), "script_sha256": "0" * 64},
                }
            )
        },
    )

    verify = _evidence("verify", "--root-dir", str(workspace))

    assert verify.returncode == 1
    results = _report(workspace)["results"]
    artifact_results = [item for item in results if item["raw_reference"] == "<artifact.path>"]
    script_results = [item for item in results if item["raw_reference"] == "<generated_by.script_path>"]
    assert artifact_results and artifact_results[0]["status"] == "dangling"
    assert script_results and script_results[0]["status"] == "dangling"
    assert "outside the paper workspace" in artifact_results[0]["messages"][0]
    assert "Absolute evidence paths are not allowed" in script_results[0]["messages"][0]
    raw_references = {item.get("raw_reference") for item in results}
    assert "<artifact.sha256>" not in raw_references
    assert "<generated_by.script_sha256>" not in raw_references


# =========================================================================== validate_claims.py (both skills)


@pytest.fixture(params=["write-paper", "standalone"])
def validate_claims(request: pytest.FixtureRequest) -> ModuleType:
    return _load_module(VALIDATE_CLAIMS[request.param], f"validate_claims_{request.param}")


def _bundle_with_stale_report(paper_dir: Path) -> tuple[Path, dict]:
    bundle_dir = paper_dir / "scientific-evidence"
    bundle_dir.mkdir(parents=True)
    (paper_dir / "paper.tex").write_text("No evidence links.\n", encoding="utf-8")
    stale_report = {"stale": True}
    (bundle_dir / "verification-report.json").write_text(json.dumps(stale_report), encoding="utf-8")
    return bundle_dir / "verification-report.json", stale_report


def _fake_verifier(path: Path, reported_root: str | None, exit_code: int) -> Path:
    root_expression = "root_dir" if reported_root is None else repr(reported_root)
    path.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "root_dir = args[args.index('--root-dir') + 1]\n"
        "output_path = Path(args[args.index('--output') + 1])\n"
        "output_path.write_text(json.dumps({\n"
        "    'schema_name': 'scientific-evidence.verification-report',\n"
        "    'verified_at': '2026-01-01T00:00:00Z',\n"
        f"    'root_dir': {root_expression},\n"
        "    'summary': {'total_references': 0, 'resolved_references': 0, 'anchored_references': 0,\n"
        "                'dangling_references': 0, 'manifests_checked': 0},\n"
        "    'results': [],\n"
        "}), encoding='utf-8')\n"
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    return path


BINDING_CASES = [
    ("foreign-root", "/tmp/another-paper", 0, "schema-invalid report"),
    ("exit-status-mismatch", None, 1, "exit status disagrees"),
]


def test_validate_claims_durable_report_path_and_payload_validator(
    validate_claims: ModuleType, tmp_path: Path
) -> None:
    # uses the durable verification report path
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    subprocess.check_call([sys.executable, str(SCIENTIFIC_EVIDENCE_PY), "emit-bundle", "--root-dir", str(paper_dir)])
    (paper_dir / "paper.tex").write_text(
        "\n".join(f"Missing {idx}. @evidence[[results/missing_{idx}.json#acc]]" for idx in range(6)) + "\n",
        encoding="utf-8",
    )

    errors = validate_claims.validate_evidence_bundle(paper_dir)

    expected_report_path = paper_dir / "scientific-evidence" / "verification-report.json"
    assert expected_report_path.is_file()
    assert len(errors) == 1
    assert str(expected_report_path) in errors[0]

    # the payload validator rejects inconsistent results
    payload = {
        "schema_name": "scientific-evidence.verification-report",
        "verified_at": "2026-01-01T00:00:00Z",
        "root_dir": "/tmp/paper",
        "summary": {
            "total_references": 1, "resolved_references": 1, "anchored_references": 0,
            "dangling_references": 0, "manifests_checked": 0,
        },
        "results": [
            {
                "subject_path": "paper.tex", "subject_kind": "text-artifact",
                "raw_reference": "@evidence[[results/value.json#score]]", "target": "results/value.json",
                "status": "invalid", "messages": [],
            }
        ],
    }
    assert validate_claims.is_usable_verification_payload(payload, expected_root=Path("/tmp/paper")) is False


def test_validate_claims_keeps_stale_report_after_verifier_crash_or_unbound_report(
    validate_claims: ModuleType, tmp_path: Path
) -> None:
    # rejects a stale report after the verifier crashes
    paper_dir = tmp_path / "crash" / "paper"
    report_path, stale_report = _bundle_with_stale_report(paper_dir)
    crashing_script = tmp_path / "crashing_verifier.py"
    crashing_script.write_text("raise SystemExit(2)\n", encoding="utf-8")
    setattr(validate_claims, "SCIENTIFIC_EVIDENCE_SCRIPT", crashing_script)

    errors = validate_claims.validate_evidence_bundle(paper_dir)

    assert len(errors) == 1
    assert "rc=2" in errors[0]
    assert json.loads(report_path.read_text(encoding="utf-8")) == stale_report
    assert list(report_path.parent.glob(".verification-report.*.json")) == []

    # binds the report to the invocation (foreign root / exit-status mismatch)
    for name, reported_root, exit_code, expected_error in BINDING_CASES:
        paper_dir = tmp_path / name / "paper"
        report_path, stale_report = _bundle_with_stale_report(paper_dir)
        setattr(
            validate_claims,
            "SCIENTIFIC_EVIDENCE_SCRIPT",
            _fake_verifier(tmp_path / f"verifier_{name}.py", reported_root, exit_code),
        )

        errors = validate_claims.validate_evidence_bundle(paper_dir)

        assert len(errors) == 1, name
        assert expected_error in errors[0], name
        assert json.loads(report_path.read_text(encoding="utf-8")) == stale_report, name


NESTED_TOKEN_CASES = [
    (
        "nested-operational-names",
        {
            "results/validation_report.json": '{"claim":"@evidence[[generated/missing.json#acc]]"}\n',
            "results/.pi/data.md": "Claim @evidence[[generated/missing.json#acc]].\n",
        },
        "2 `@evidence[[...]]` token(s)",
    ),
    (
        "nested-validation-report",
        {
            "paper.tex": "No evidence refs.\n",
            "results/validation_report.json": '{"source": "@evidence[[results/missing.json#accuracy]]"}\n',
        },
        "scientific-evidence/ bundle missing",
    ),
]


def test_validate_claims_scans_nested_tokens_and_standalone_ignores_feedback_artifacts(
    validate_claims: ModuleType, tmp_path: Path
) -> None:
    # scans nested evidence tokens
    for name, files, expected_error in NESTED_TOKEN_CASES:
        paper_dir = tmp_path / name / "paper"
        _write_files(paper_dir, files)

        errors = validate_claims.validate_evidence_bundle(paper_dir)

        assert len(errors) == 1, name
        assert expected_error in errors[0], name

    # the standalone validator ignores generated feedback artifacts
    standalone = _load_module(VALIDATE_CLAIMS["standalone"], "validate_claims_standalone")
    paper_dir = tmp_path / "standalone" / "paper"
    _write_files(
        paper_dir,
        {
            "paper.tex": "Clean paper without evidence refs.\n",
            "validation_feedback.md": "Old feedback mentions @evidence[[results/missing.json#acc]].\n",
            "paper_evidence_audit_feedback.md": "Old audit mentions @evidence[[results/missing.json#acc]].\n",
            "validation_report.json": '{"hard_errors": ["Fix dangling `@evidence[[...]]` links."]}\n',
        },
    )

    assert standalone.validate_evidence_bundle(paper_dir) == []


_CLEAN_TEX = {"paper.tex": "Clean paper without evidence refs.\n"}


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        *[
            pytest.param(
                {
                    **_CLEAN_TEX,
                    "/".join(parts) + "/validate-paper/SKILL.md":
                        "Example @evidence[[results/missing.json#accuracy]].\n",
                },
                None, id="ignores-agent-support-" + parts[0],
            )
            for parts in AGENT_SKILL_PARTS
        ],
        pytest.param(
            {
                CLAIM_TEX: "Unsupported @evidence[[results/missing.json#accuracy]].\n",
                "paper.tex": f"\\input {CLAIM_TEX}\n",
            },
            MUST_NOT_IMPORT, id="imported-agent-skill-tex",
        ),
        *[
            pytest.param({"paper.tex": f"\\input{{{target}}}\n"}, MUST_NOT_IMPORT, id=f"normalized-{idx}")
            for idx, target in enumerate(
                [".codex/./skills/payload/claim.tex", ".codex//skills/payload/claim.tex",
                 ".codex/x/../skills/payload/claim.tex"]
            )
        ],
        pytest.param(
            {"paper.tex": f"\\newcommand{{\\loadfile}}[1]{{\\input{{#1}}}}\n\\loadfile{{{CLAIM_TEX}}}\n"},
            "dynamic import target", id="macro-wrapped-import",
        ),
        pytest.param(
            {"paper.tex": f"\\newcommand{{\\loadfile}}{{\\input}}\n\\loadfile{{{CLAIM_TEX}}}\n"},
            "dynamic import command", id="zero-argument-alias",
        ),
        pytest.param(
            {
                "paper.tex": "\\usepackage{loader}\n\\loadclaim\n",
                "loader.sty": f"\\newcommand{{\\loadclaim}}{{\\input{{{CLAIM_TEX}}}}}\n",
            },
            MUST_NOT_IMPORT, id="workspace-local-package",
        ),
        pytest.param(
            {"paper.tex": "\\newcommand{\\paperpkg}{loader}\n\\usepackage{\\paperpkg}\n"},
            "dynamic support import", id="dynamic-local-package",
        ),
        pytest.param({"paper.tex": f"\\verb|\\input{{{CLAIM_TEX}}}|\n"}, None, id="ignores-verb-literal"),
        pytest.param(
            {"paper.tex": f"\\verb|\\iffalse| \\input{{{CLAIM_TEX}}}\n"}, MUST_NOT_IMPORT, id="masked-false-literal"
        ),
        pytest.param(
            {"paper.tex": f"% \\begin{{verbatim}}\n\\input{{{CLAIM_TEX}}}\n% \\end{{verbatim}}\n"},
            MUST_NOT_IMPORT, id="commented-verbatim-delimiters",
        ),
        pytest.param({"paper.tex": f"\\iffalse\n\\input{{{CLAIM_TEX}}}\n\\fi\n"}, None, id="ignores-disabled-input"),
        pytest.param(
            {"paper.tex": f"\\iffalse\nDisabled.\n\\else\n\\input{{{CLAIM_TEX}}}\n\\fi\n"},
            MUST_NOT_IMPORT, id="input-in-else-branch",
        ),
        pytest.param({"paper.tex": "The staged path is \\verb|.codex/skills|.\n"}, None, id="allows-path-in-prose"),
        pytest.param(
            {"../external.tex": "External content.\n", "paper.tex": "\\input{../external.tex}\n"},
            "escapes the paper workspace", id="outside-workspace-relative",
        ),
        pytest.param(
            {"../external.tex": "External content.\n", "paper.tex": "\\input{__EXTERNAL__}\n"},
            "escapes the paper workspace", id="outside-workspace-absolute",
        ),
    ],
)
def test_validate_claims_tex_import_policy(
    validate_claims: ModuleType, tmp_path: Path, files: dict[str, str], expected: str | None
) -> None:
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    _write_files(paper_dir, files, __EXTERNAL__=str(tmp_path / "external.tex"))

    if expected is None:
        assert validate_claims.validate_evidence_bundle(paper_dir) == []
    else:
        with pytest.raises(RuntimeError, match=expected):
            validate_claims.validate_evidence_bundle(paper_dir)


# =========================================================================== validate_rendered_si.py + compile_latex.sh

SI_BODY = "Grounded robustness and reproducibility details. " * 5


@pytest.fixture(scope="module")
def rendered_si() -> ModuleType:
    return _load_module(VALIDATE_RENDERED_SI, "validate_rendered_si")


@pytest.mark.parametrize(
    ("pages", "expected_error"),
    [
        pytest.param(
            ["Main paper", "References\nReference entries", "A Supplementary Information\n" + SI_BODY],
            None, id="visible-supplement-after-references",
        ),
        pytest.param(
            ["Main paper", "R EFERENCES\nReference entries", "A     S UPPLEMENTARY I NFORMATION\n" + SI_BODY],
            None, id="letter-spaced-small-caps",
        ),
        pytest.param(
            ["References\nReference entries", "A S UPPLEMENTARY\nI NFORMATION\n" + SI_BODY],
            "missing the Supplementary Information heading", id="heading-split-across-lines",
        ),
        pytest.param(
            ["References\nReference entries", r"\section{Supplementary Information}" + " literal source code" * 12],
            "missing the Supplementary Information heading", id="literal-latex-heading",
        ),
        pytest.param(
            [
                "References\nMain bibliography",
                "A Supplementary Information\n" + "Extended grounded details. " * 6,
                "References\nSupplement-specific citations and more grounded details.",
            ],
            None, id="supplemental-references-after-si",
        ),
        pytest.param(
            [
                "References\nMain bibliography",
                "A Supplementary Information",
                "References\n" + "Long supplemental citation entry. " * 8,
            ],
            "at least 100 visible characters", id="empty-supplement-long-references",
        ),
        pytest.param(
            [
                "References\nMain bibliography starts",
                "Continuation of a bibliography entry\nA Supplementary Information\n" + SI_BODY,
            ],
            "must start on a separate page", id="bibliography-continuation-before-heading",
        ),
        pytest.param(
            ["Paper Title\nReferences\nMain bibliography", "Paper Title\nA Supplementary Information\n" + SI_BODY],
            None, id="repeated-running-header",
        ),
    ],
)
def test_validate_rendered_supplementary_information(
    rendered_si: ModuleType, pages: list[str], expected_error: str | None
) -> None:
    errors = rendered_si.validate_rendered_supplementary_information(pages)

    if expected_error is None:
        assert errors == []
    else:
        assert expected_error in "; ".join(errors)


def _run_compile_latex(root: Path, fakes: dict[str, str], env_overrides: dict[str, str] | None = None):
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    (root / "paper.tex").write_text("placeholder", encoding="utf-8")
    for name, body in fakes.items():
        _write_executable(bin_dir / name, body)
    env = os.environ | {"PATH": f"{bin_dir}:/usr/bin", **(env_overrides or {})}
    return subprocess.run(
        ["bash", str(COMPILE_LATEX), "paper.tex"], cwd=root, env=env, check=False, capture_output=True, text=True
    )


_WRITE_PLAIN_AUX = "printf '\\\\bibstyle{plain}\\n\\\\bibdata{references}\\n' > paper.aux\n"


def test_compile_script_rendered_gate_env_scrubbing_and_input_rejections(tmp_path: Path) -> None:
    # runs the rendered validation after the final pdflatex pass, without shell escape
    compile_script = COMPILE_LATEX.read_text(encoding="utf-8")
    assert compile_script.index("validate_rendered_si.py") > compile_script.rindex("run_pdflatex")
    assert "-no-shell-escape" in compile_script

    # executes the rendered gate with the system python and a scrubbed environment
    root = tmp_path / "gate"
    tex_tool = (
        "/usr/bin/env >> compile-environment.log\n"
        "if [ \"$(basename \"$0\")\" = pdflatex ]; then\n"
        "  printf 'PWD %s\\nINPUT paper.tex\\n' \"$PWD\" > paper.fls\n"
        "  " + _WRITE_PLAIN_AUX +
        "  : > paper.pdf\n"
        "else\n"
        "  [ -z \"${BIBINPUTS+x}\" ] || exit 9\n"
        "  [ -z \"${BSTINPUTS_bibtex+x}\" ] || exit 9\n"
        "  [ \"${TEXMFHOME:-}\" = /nonexistent ] || exit 9\n"
        "fi\n"
    )
    pdftotext = (
        "printf 'Main paper\\fReferences\\nEntries\\fA Supplementary Information\\n'\n"
        "printf 'Grounded robustness and reproducibility details. %.0s' 1 2 3 4 5\n"
    )
    leaked = (
        "BIBINPUTS", "BSTINPUTS_bibtex", "TEXINPUTS", "TEXINPUTS.pdflatex", "TEXFORMATS",
        "TEXMFLOCAL", "KPSE_DOT", "shell_escape", "openout_any", "UNRELATED_SECRET",
    )
    env = {name: "/tmp/untrusted" for name in leaked}
    env.update({"shell_escape": "t", "openout_any": "a", "UNRELATED_SECRET": "must-not-reach-tex"})

    result = _run_compile_latex(root, {"pdflatex": tex_tool, "bibtex": tex_tool, "pdftotext": pdftotext}, env)

    assert result.returncode == 0, result.stderr
    assert "Validated rendered Supplementary Information" in result.stdout
    compile_environment = (root / "compile-environment.log").read_text(encoding="utf-8")
    for name in leaked:
        assert f"{name}=" not in compile_environment
    assert "HOME=/nonexistent" in compile_environment

    # rejects staged skill files recorded as pdflatex inputs
    pdflatex = (
        "mkdir -p .codex/skills/payload\n"
        ": > .codex/skills/payload/claim.tex\n"
        "printf 'PWD %s\\nINPUT .codex/skills/payload/claim.tex\\n' \"$PWD\" > paper.fls\n"
        + _WRITE_PLAIN_AUX
    )

    result = _run_compile_latex(tmp_path / "skill", {"pdflatex": pdflatex, "bibtex": "exit 0", "pdftotext": "exit 0"})

    assert result.returncode != 0
    assert "must not read staged agent skill files" in result.stderr

    # rejects an external bibliography before bibtex runs
    root = tmp_path / "bib"
    pdflatex = (
        "printf 'PWD %s\\nINPUT paper.tex\\n' \"$PWD\" > paper.fls\n"
        "printf '\\\\bibstyle{plain}\\n\\\\bibdata{/tmp/evil}\\n' > paper.aux\n"
    )

    result = _run_compile_latex(root, {"pdflatex": pdflatex, "bibtex": ": > bibtex-ran", "pdftotext": "exit 0"})

    assert result.returncode != 0
    assert "canonical references.bib" in result.stderr
    assert not (root / "bibtex-ran").exists()


# =========================================================================== validate_tex_refs.py

SI_DETAILS = (
    "\\subsection{Extended evaluation details}\n"
    "Additional grounded implementation and evaluation details support reproducibility, "
    "including controls, limitations, robustness checks, and failure analysis for the study.\n"
)
REAL_SUPPLEMENT = "\\section{Supplementary Information}\n" + SI_DETAILS
FAKE_SUPPLEMENT = (
    "\\section{Supplementary Information}\n"
    "\\subsection{Fake}\n"
    "This literal example contains more than one hundred characters but is source code, "
    "not rendered supplementary material, and must never satisfy the structural gate.\n"
)
VALID_TEX = (
    "\\begin{document}\n"
    "\\begin{abstract}Summary.\\end{abstract}\n"
    "\\section{Introduction}Intro.\n"
    "\\section{Related Work}Prior work.\n"
    "\\section{Method}Method.\n"
    "\\section{Results}Results.\n"
    "\\section{Conclusion}Conclusion.\n"
    "\\clearpage\n"
    "\\bibliography{references}\n"
    "\\clearpage\n"
    "\\appendix\n"
    + REAL_SUPPLEMENT
    + "\\end{document}\n"
)
MISSING_SI_HEADING = "missing Supplementary Information heading"
UNRELATED_TAIL = "Unrelated visible prose after the supplement.\n\\end{document}\n"


def _after_supplement(extra: str) -> str:
    return VALID_TEX.replace("\\end{document}\n", extra + "\\end{document}\n")


def _preamble(extra: str) -> str:
    return VALID_TEX.replace("\\begin{document}\n", extra + "\\begin{document}\n")


def _wrap_tail(prefix: str, suffix: str = "\\fi\n\\end{document}\n") -> str:
    start = VALID_TEX.index("\\clearpage\n\\bibliography")
    tail = VALID_TEX[start : VALID_TEX.index("\\end{document}")]
    return VALID_TEX[:start] + prefix + tail + suffix


def _literal_environment(environment: str) -> str:
    begin = "\\begin{minted}{text}\n" if environment == "minted" else f"\\begin{{{environment}}}\n"
    return VALID_TEX.replace(REAL_SUPPLEMENT, f"{begin}{FAKE_SUPPLEMENT}\\end{{{environment}}}\n")


_NESTED_DISABLED_SUFFIX = "\\else\nNo supplementary material in the active branch.\n\\fi\n\\end{document}\n"
_NONPRIMITIVE_IF_MACROS = (
    "\\ifthenelse{a}{b}{c}", "\\iflanguage{english}{a}{b}", "\\ifnumcomp{1}{=}{1}{a}{b}", "\\ifdefempty{example}{a}{b}"
)

TEX_CASES = [
    pytest.param(VALID_TEX, (), id="accepts-substantive-si"),
    pytest.param(
        VALID_TEX.split("\\clearpage\n\\appendix\n", maxsplit=1)[0] + "\\end{document}\n",
        (MISSING_SI_HEADING,), id="missing-si",
    ),
    pytest.param(VALID_TEX.replace("\\clearpage\n\\appendix", "\\appendix"), (r"\clearpage",), id="missing-clearpage"),
    pytest.param(
        VALID_TEX.replace("Conclusion.\n\\clearpage\n\\bibliography", "Conclusion.\n\\bibliography"),
        ("before the bibliography",), id="missing-clearpage-before-bibliography",
    ),
    pytest.param(
        VALID_TEX.replace("\\bibliography{references}\n", "")
        .replace("\\clearpage\n\\appendix", "\\iffalse\\bibliography{references}\\fi\n\\clearpage\n\\appendix")
        .replace("\\end{document}", "\\bibliography{references}\n\\end{document}"),
        ("after the bibliography",), id="ignores-false-conditional-bibliography",
    ),
    pytest.param(
        VALID_TEX.replace(
            "\\bibliography{references}", "\\newcommand{\\fakebibliography}{\\bibliography{references}}", 1
        ).replace("\\end{document}", "\\bibliography{references}\n\\end{document}"),
        ("after the bibliography",), id="real-bibliography-after-si",
    ),
    pytest.param(
        VALID_TEX.replace(
            SI_DETAILS,
            "\\begin{figure}\n"
            "\\includegraphics{figures/a-very-long-filename-that-must-not-count-as-visible-"
            "supplementary-prose-or-grounded-scientific-content.png}\n"
            "\\end{figure}\n",
        ),
        ("100 visible characters",), id="empty-figure-filename-not-visible",
    ),
    pytest.param(
        VALID_TEX.replace(SI_DETAILS, "Add supplementary material only when grounded artifacts support it.\n"),
        ("placeholder text",), id="placeholder-si",
    ),
    pytest.param(
        VALID_TEX.replace("\\appendix\n", "").replace(
            "\\bibliography{references}\n", "\\appendix\n\\bibliography{references}\n"
        ),
        ("must appear after the bibliography",), id="appendix-before-references",
    ),
    pytest.param(
        VALID_TEX.replace(
            SI_DETAILS,
            "Empty.\n\\section{Unrelated Material}\n\\subsection{Long unrelated details}\n"
            "This later section has enough prose to exceed the threshold but cannot satisfy the "
            "Supplementary Information contract.\n",
        ),
        ("at least 100 visible characters", "final top-level section"), id="empty-si-before-unrelated-section",
    ),
    pytest.param(
        _after_supplement("\\section*{References}\nSupplement-specific citations.\n"),
        (), id="allows-supplement-local-references",
    ),
    pytest.param(
        _after_supplement("\\csname section\\endcsname{other material}\nUnrelated visible prose after the supplement.\n"),
        ("dynamic top-level sections",), id="dynamic-top-level-section",
    ),
    pytest.param(
        _after_supplement("\\section{Additional \\textbf{Results}}\nUnrelated visible prose after the supplement.\n"),
        ("final top-level section",), id="nested-title-section",
    ),
    pytest.param(
        _after_supplement("\\section[Additional]{Additional Results}\nUnrelated visible prose after the supplement.\n"),
        ("final top-level section",), id="short-title-section",
    ),
    pytest.param(
        _after_supplement(
            "\\section*{References}\nSupplement-specific citations.\n\\section{Other Material}\nUnrelated prose.\n"
        ),
        ("final top-level section",), id="section-after-local-references",
    ),
    pytest.param(
        _preamble("\\let\\latersection\\section\n").replace(
            "\\end{document}\n", "\\latersection*{other material}\nUnrelated prose.\n\\end{document}\n"
        ),
        ("dynamic top-level sections",), id="section-alias",
    ),
    pytest.param(
        _preamble("\\newcommand{\\latersection}[1]{%\n  \\section{#1}}\n").replace(
            "\\end{document}\n", "\\latersection{Other Material}\nUnrelated prose.\n\\end{document}\n"
        ),
        ("dynamic top-level sections",), id="multiline-section-wrapper",
    ),
    pytest.param(_preamble("\\newcommand{\\modelname}{Example}\n"), (), id="allows-unrelated-macro"),
    pytest.param(_after_supplement("\\input{later-section}\n"), ("unchecked TeX files",), id="tex-import-after-si"),
    pytest.param(
        VALID_TEX.replace("\\bibliography{references}\n", "\\section{References}\n").replace(
            "\\end{document}\n", "\\bibliography{references}\n\\end{document}\n"
        ),
        ("must appear after the bibliography",), id="si-before-actual-bibliography-command",
    ),
    *[
        pytest.param(_literal_environment(environment), (MISSING_SI_HEADING,), id=f"literal-{environment}")
        for environment in ("verbatim", "lstlisting", "minted")
    ],
    pytest.param(_wrap_tail("\\iffalse\nDisabled branch.\n\\else\n"), (), id="si-in-active-else-branch"),
    *[
        pytest.param(_wrap_tail(f"\\iffalse\n{macro}\n\\else\n"), (), id=f"nonprimitive-if-{idx}")
        for idx, macro in enumerate(_NONPRIMITIVE_IF_MACROS)
    ],
    pytest.param(
        VALID_TEX.replace(
            "\\section{Conclusion}Conclusion.\n",
            "\\begin{verbatim}\n\\iffalse\n\\end{verbatim}\n\\section{Conclusion}Conclusion.\n",
        ),
        (), id="ignores-false-conditional-in-verbatim",
    ),
    pytest.param(
        _wrap_tail("\\newif\\ifFoo\n\\iffalse\n\\ifFoo\nNested disabled branch.\n\\fi\n", _NESTED_DISABLED_SUFFIX),
        (MISSING_SI_HEADING,), id="fake-si-under-newif-conditional",
    ),
    pytest.param(
        _wrap_tail("\\iffalse\n\\ifExternal\n{Nested disabled branch.}\n\\fi\n", _NESTED_DISABLED_SUFFIX),
        (MISSING_SI_HEADING,), id="fake-si-under-external-conditional",
    ),
]


@pytest.mark.parametrize(("tex", "expected_errors"), TEX_CASES)
def test_validate_tex_refs(tmp_path: Path, tex: str, expected_errors: tuple[str, ...]) -> None:
    tex_path, bib_path = tmp_path / "paper.tex", tmp_path / "references.bib"
    tex_path.write_text(tex, encoding="utf-8")
    bib_path.write_text("@article{RefKey2024, title={Reference}}\n", encoding="utf-8")

    result = subprocess.run(
        [str(VALIDATE_TEX_REFS), "--tex", str(tex_path), "--bib", str(bib_path)],
        check=False, capture_output=True, text=True,
    )

    if not expected_errors:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        for fragment in expected_errors:
            assert fragment in result.stderr


# =========================================================================== scripts/write_paper_variants.sh


def _create_task_bundle(path: Path) -> None:
    program_dir = path / "program_0"
    (program_dir / "src").mkdir(parents=True)
    (program_dir / "program_spec.toml").write_text(
        'schema_version = 1\ncode_root = "src"\n\n'
        '[entrypoint]\npath = "program.py"\nlanguage = "python"\ntype = "script"\n\n'
        "[copy]\nexclude = []\ninclude = []\n\n"
        '[integration]\nkind = "generic"\n',
        encoding="utf-8",
    )
    (program_dir / "src" / "program.py").write_text("", encoding="utf-8")
    (path / "task.md").write_text("# Task\n", encoding="utf-8")
    (path / "evaluate.py").write_text("", encoding="utf-8")


_WRAPPER_ENV_VARS = ("TASK_DIR", "TASK_NAME", "CONTAINER_ENV_FILE", "AGENT", "AGENT_MODEL", "AGENT_PROFILE")


def _run_wrapper(
    tmp_path: Path,
    bundle_names: list[str],
    *,
    partial_bundle_names: list[str] | None = None,
    mutate_results: Callable[[Path], None] | None = None,
    **env_vars: str,
) -> tuple[subprocess.CompletedProcess[str], list[str], Path]:
    results_dir = tmp_path / "variants"
    aggregated = results_dir / "aggregated" / "example_task"
    aggregated.mkdir(parents=True)
    (aggregated / "run_summary.csv").write_text("score\n1\n", encoding="utf-8")
    for name in bundle_names:
        _create_task_bundle(results_dir / name)
    for name in partial_bundle_names or []:
        (results_dir / name).mkdir()
    if mutate_results is not None:
        mutate_results(results_dir)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured_args = tmp_path / "paper-args.txt"
    _write_executable(fake_bin / "bash", "printf '%s\\n' \"$@\" > \"$CAPTURE_ARGS\"")

    env = os.environ.copy()
    for name in _WRAPPER_ENV_VARS:
        env.pop(name, None)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "CAPTURE_ARGS": str(captured_args),
            "SRC_RESULTS_DIR": str(results_dir),
            "OUT_DIR": str(tmp_path / "paper"),
            **env_vars,
        }
    )
    completed = subprocess.run(
        ["/bin/bash", str(WRITE_PAPER_VARIANTS)], cwd=REPO_ROOT, env=env, check=False, capture_output=True, text=True
    )
    args = captured_args.read_text(encoding="utf-8").splitlines() if captured_args.exists() else []
    return completed, args, results_dir


def _argument_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def _add_extra_task(name: str) -> Callable[[Path], None]:
    def mutate(results_dir: Path) -> None:
        extra = results_dir / "aggregated" / name
        extra.mkdir()
        (extra / "run_summary.csv").write_text("score\n2\n", encoding="utf-8")

    return mutate


def _create_incomplete_retrofit(results_dir: Path) -> None:
    retrofit = results_dir / "_retrofitted_bundle"
    (retrofit / "task.md").write_text("# Task\n", encoding="utf-8")
    (retrofit / "evaluate.py").write_text("", encoding="utf-8")
    (retrofit / "program_0").mkdir()


def _symlink_retrofit_root_outside(results_dir: Path) -> None:
    outside = results_dir.parent / "outside"
    _create_task_bundle(outside)
    (results_dir / "_retrofitted_bundle").symlink_to(outside, target_is_directory=True)


def _replace_retrofit_task_with_escape(results_dir: Path) -> None:
    outside_task = results_dir.parent / "outside-task.md"
    outside_task.write_text("sensitive", encoding="utf-8")
    task_md = results_dir / "_retrofitted_bundle" / "task.md"
    task_md.unlink()
    task_md.symlink_to(outside_task)


BUNDLE_SELECTION_CASES = [
    (["_extracted_bundle"], "_extracted_bundle"),
    (["_extracted_bundle", "_retrofitted_bundle"], "_retrofitted_bundle"),
    (["task", "_retrofitted_bundle", "_extracted_bundle"], "task"),
]


def test_wrapper_selects_best_available_bundle_unless_task_dir_is_explicit(tmp_path: Path) -> None:
    # selects the best available task bundle
    for idx, (bundle_names, expected_bundle) in enumerate(BUNDLE_SELECTION_CASES):
        completed, args, results_dir = _run_wrapper(tmp_path / f"case{idx}", bundle_names)
        assert completed.returncode == 0, (expected_bundle, completed.stderr)
        assert _argument_value(args, "--task-dir") == str(results_dir / expected_bundle), expected_bundle

    # an explicit TASK_DIR takes precedence
    explicit_task = tmp_path / "explicit-task"
    explicit_task.mkdir()

    completed, args, _ = _run_wrapper(tmp_path / "explicit", ["task", "_retrofitted_bundle"], TASK_DIR=str(explicit_task))

    assert completed.returncode == 0, completed.stderr
    assert _argument_value(args, "--task-dir") == str(explicit_task)


@pytest.mark.parametrize(
    ("bundle_names", "partial_bundle_names", "mutate_results"),
    [
        pytest.param(["_extracted_bundle"], ["_retrofitted_bundle"], None, id="partial-retrofit"),
        pytest.param(["_extracted_bundle"], ["_retrofitted_bundle"], _create_incomplete_retrofit, id="empty-program"),
        pytest.param(["_extracted_bundle"], None, _symlink_retrofit_root_outside, id="symlink-bundle-root"),
        pytest.param(
            ["_extracted_bundle", "_retrofitted_bundle"], None, _replace_retrofit_task_with_escape,
            id="escaping-nested-symlink",
        ),
    ],
)
def test_wrapper_falls_back_to_complete_extracted_bundle(
    tmp_path: Path, bundle_names, partial_bundle_names, mutate_results
) -> None:
    completed, args, results_dir = _run_wrapper(
        tmp_path, bundle_names, partial_bundle_names=partial_bundle_names, mutate_results=mutate_results
    )

    assert completed.returncode == 0, completed.stderr
    assert _argument_value(args, "--task-dir") == str(results_dir / "_extracted_bundle")


def test_wrapper_aggregated_task_detection_and_symlinked_root_rejection(tmp_path: Path) -> None:
    # auto-detection rejects multiple aggregated tasks
    completed, args, _ = _run_wrapper(
        tmp_path / "multi", ["_extracted_bundle"], mutate_results=_add_extra_task("aaa_extra")
    )

    assert completed.returncode == 1
    assert args == []
    assert "multiple tasks" in completed.stderr
    assert "set TASK_NAME explicitly" in completed.stderr

    # an explicit TASK_NAME resolves multiple aggregated tasks
    completed, _, results_dir = _run_wrapper(
        tmp_path / "named", ["_extracted_bundle"], mutate_results=_add_extra_task("extra_task"), TASK_NAME="example_task"
    )

    assert completed.returncode == 0, completed.stderr
    assert os.readlink(results_dir / "aggregated" / "run_summary.csv") == "example_task/run_summary.csv"

    # rejects a symlinked aggregated root without writing outside it
    outside = tmp_path / "outside-aggregated"
    outside_task = outside / "example_task"
    outside_task.mkdir(parents=True)
    (outside_task / "run_summary.csv").write_text("score\n9\n", encoding="utf-8")
    outside_sentinel = outside / "run_summary.csv"
    outside_sentinel.write_text("do not replace\n", encoding="utf-8")

    def redirect_aggregated_root(results_dir: Path) -> None:
        shutil.rmtree(results_dir / "aggregated")
        (results_dir / "aggregated").symlink_to(outside, target_is_directory=True)

    completed, args, _ = _run_wrapper(
        tmp_path / "symlinked", ["_extracted_bundle"], mutate_results=redirect_aggregated_root
    )

    assert completed.returncode == 1
    assert args == []
    assert outside_sentinel.read_text(encoding="utf-8") == "do not replace\n"
    assert not outside_sentinel.is_symlink()


def test_wrapper_rejects_unselectable_task_name_or_missing_task_bundle(tmp_path: Path) -> None:
    # an explicit TASK_NAME must select an aggregated task
    for idx, task_name in enumerate(["typo", "../example_task", "/"]):
        completed, args, _ = _run_wrapper(tmp_path / f"name{idx}", ["_extracted_bundle"], TASK_NAME=task_name)

        assert completed.returncode == 1, task_name
        assert args == [], task_name
        assert "TASK_NAME" in completed.stderr, task_name

    # errors when no task bundle exists
    completed, args, _ = _run_wrapper(tmp_path / "empty", [])

    assert completed.returncode == 1
    assert args == []
    assert "could not find a usable task bundle" in completed.stderr


def test_wrapper_and_paper_script_container_env_file_handling(tmp_path: Path) -> None:
    # an empty CONTAINER_ENV_FILE forwards the disable value
    completed, args, _ = _run_wrapper(tmp_path / "empty", ["_extracted_bundle"], CONTAINER_ENV_FILE="")

    assert completed.returncode == 0, completed.stderr
    assert _argument_value(args, "--container-env-file") == ""

    # an explicit CONTAINER_ENV_FILE must exist
    completed, args, _ = _run_wrapper(
        tmp_path / "missing", ["_extracted_bundle"], CONTAINER_ENV_FILE=str(tmp_path / "missing.env")
    )

    assert completed.returncode == 1
    assert args == []
    assert "container env file" in completed.stderr

    # an explicit CONTAINER_ENV_FILE is forwarded
    env_file = tmp_path / "agent.env"
    env_file.write_text("TOKEN=value\n", encoding="utf-8")

    completed, args, _ = _run_wrapper(tmp_path / "forwarded", ["_extracted_bundle"], CONTAINER_ENV_FILE=str(env_file))

    assert completed.returncode == 0, completed.stderr
    assert _argument_value(args, "--container-env-file") == str(env_file)

    # paper.sh: an explicit empty env file disables the repo default
    env = os.environ.copy()
    env.pop("AGENT_CONTAINER_ENV_FILE", None)

    completed = subprocess.run(
        ["/bin/bash", "-x", "src/scripts/paper.sh", "--container-env-file", "", "--task-dir", str(tmp_path / "missing-task")],
        cwd=REPO_ROOT, env=env, check=False, capture_output=True, text=True,
    )

    assert completed.returncode == 1
    assert f"CONTAINER_ENV_FILE={REPO_ROOT}/.env" not in completed.stderr


def test_wrapper_forwards_agent_profile_and_model_only_when_explicit(tmp_path: Path) -> None:
    # AGENT_PROFILE is forwarded
    completed, args, _ = _run_wrapper(tmp_path / "profile", ["_extracted_bundle"], AGENT_PROFILE="custom")

    assert completed.returncode == 0, completed.stderr
    assert _argument_value(args, "--agent-profile") == "custom"

    # a non-claude agent does not inherit the claude model
    completed, args, _ = _run_wrapper(tmp_path / "codex", ["_extracted_bundle"], AGENT="codex")

    assert completed.returncode == 0, completed.stderr
    assert "--agent-model" not in args

    # a non-claude agent forwards an explicit model
    completed, args, _ = _run_wrapper(
        tmp_path / "codex-model", ["_extracted_bundle"], AGENT="codex", AGENT_MODEL="gpt-5.3-codex"
    )

    assert completed.returncode == 0, completed.stderr
    assert _argument_value(args, "--agent-model") == "gpt-5.3-codex"


# =========================================================================== scripts/install_autoillustrate.sh

_FAKE_NODE = 'if [ "${1:-}" = -p ]; then echo 22; else echo v22.0.0; fi'


def _run_install(tmp_path: Path, fakes: dict[str, str], env_overrides: dict[str, str], *args: str):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    for name, body in fakes.items():
        _write_executable(fake_bin / name, body)
    env = os.environ | {"PATH": f"{fake_bin}:/usr/bin:/bin", "AUTOILLUSTRATE_PREFIX": str(tmp_path / "prefix")}
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(INSTALL_AUTOILLUSTRATE), *args], cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False
    )


def test_install_autoillustrate_pins_release_and_verifies_installed_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # installs the pinned npm release into the requested user prefix
    match = re.search(r"^ARG AUTOILLUSTRATE_VERSION=(\S+)$", DOCKERFILE.read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None
    version = match.group(1)
    monkeypatch.delenv("AUTOILLUSTRATE_VERSION", raising=False)
    root = tmp_path / "pinned"
    install_prefix, npm_log = root / "prefix", root / "npm.log"
    npm = (
        "printf '%s\\n' \"$*\" >> \"$TEST_NPM_LOG\"\n"
        'if [ "${1:-}" = install ]; then\n'
        '  mkdir -p "$TEST_INSTALL_PREFIX/bin"\n'
        "  printf '#!/bin/sh\\necho %s\\n' \"$TEST_VERSION\" > \"$TEST_INSTALL_PREFIX/bin/autoillustrate\"\n"
        '  chmod +x "$TEST_INSTALL_PREFIX/bin/autoillustrate"\n'
        "fi"
    )

    result = _run_install(
        root,
        {"node": _FAKE_NODE, "npm": npm},
        {"TEST_INSTALL_PREFIX": str(install_prefix), "TEST_NPM_LOG": str(npm_log), "TEST_VERSION": version},
    )

    assert result.returncode == 0, result.stderr
    assert version in result.stdout
    assert npm_log.read_text(encoding="utf-8").splitlines() == [
        f"install --global --prefix {install_prefix} autoillustrate@{version}"
    ]
    assert (install_prefix / "bin" / "autoillustrate").is_file()

    # rejects a version mismatch after install
    root = tmp_path / "mismatch"
    npm = (
        'mkdir -p "$TEST_INSTALL_PREFIX/bin"\n'
        "printf '#!/bin/sh\\necho 0.0.1\\n' > \"$TEST_INSTALL_PREFIX/bin/autoillustrate\"\n"
        'chmod +x "$TEST_INSTALL_PREFIX/bin/autoillustrate"'
    )

    result = _run_install(
        root,
        {"node": _FAKE_NODE, "npm": npm},
        {"AUTOILLUSTRATE_VERSION": "0.4.8", "TEST_INSTALL_PREFIX": str(root / "prefix")},
    )

    assert result.returncode == 1
    assert "reports 0.0.1, expected 0.4.8" in result.stderr


def test_install_autoillustrate_rejects_unknown_argument_and_missing_nix_before_installing(tmp_path: Path) -> None:
    # rejects an unknown argument without installing
    result = subprocess.run(
        ["bash", str(INSTALL_AUTOILLUSTRATE), "--unknown"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 2
    assert "Unknown argument" in result.stderr

    # the drawio option requires an existing Nix before npm install
    fakes = {
        "node": 'if [ "${1:-}" = -p ]; then echo 22; else echo v22; fi',
        "npm": f"printf called > {tmp_path / 'npm-called'}",
        "uname": "echo Linux",
    }

    result = _run_install(tmp_path, fakes, {}, "--install-drawio-with-nix")

    assert result.returncode == 1
    assert "requires an existing Nix" in result.stderr
    assert not (tmp_path / "npm-called").exists()
    assert 'bash "$AUTOILLUSTRATE_PACKAGE_DIR/scripts/install_drawio.sh"' in INSTALL_AUTOILLUSTRATE.read_text(
        encoding="utf-8"
    )
