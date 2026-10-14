"""derive_paths, directory-descriptor path resolution, and paper-memory alias restoration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.pipeline.config import PipelineConfig
from src.pipeline.descriptor_paths import directory_descriptor_path
from src.pipeline.paper_memory import restore_materialized_paper_memory
from src.pipeline.paths import derive_paths

DEFAULT_RUN_DIR = Path("results/circpack/claude/20260101_000000")


def _cfg(**kw: object) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.task.dir = "examples/circpack"
    cfg.agent.name = "claude"
    cfg.run_id = "20260101_000000"
    for key, value in kw.items():
        head, _, tail = key.partition(".")
        setattr(getattr(cfg, head), tail, value)
    return cfg


# --- derive_paths -----------------------------------------------------------------

_OVERRIDE_CASES = [
    (
        "run-dir-flows-into-ablations",
        {"paths.run_dir": "/tmp/my_run"},
        {"run_dir": Path("/tmp/my_run"), "ablations_dir": Path("/tmp/my_run/ablations")},
    ),
    (
        "ablations-dir-does-not-move-paper",
        {"paths.ablations_dir": "/tmp/bundle"},
        {"ablations_dir": Path("/tmp/bundle"), "paper_dir": DEFAULT_RUN_DIR / "paper"},
    ),
    ("paper-dir", {"paths.paper_dir": "/tmp/paper_out"}, {"paper_dir": Path("/tmp/paper_out")}),
]


def test_derive_paths_defaults_overrides_and_minted_run_id() -> None:
    # defaults match the stage conventions
    paths = derive_paths(_cfg())

    assert paths.task_name == "circpack"
    assert paths.run_id == "20260101_000000"
    assert paths.source_task_dir == Path("examples/circpack")
    assert paths.run_dir == DEFAULT_RUN_DIR
    assert paths.task_dir == paths.run_dir / "task"
    assert paths.task_drafting_dir == paths.run_dir / "task_drafting"
    assert paths.investigation_dir == paths.run_dir / "investigation"
    assert not paths.task_dir.is_relative_to(paths.investigation_workspace_dir)
    assert paths.ablations_dir == paths.run_dir / "ablations"
    assert paths.ablations_aggregated_task_dir == paths.ablations_dir / "aggregated" / "circpack"
    assert paths.posthoc_dir == paths.run_dir / "posthoc"
    assert paths.final_dir == paths.run_dir / "final_evaluation"
    assert paths.paper_dir == paths.run_dir / "paper"

    # explicit output overrides
    for name, overrides, expected in _OVERRIDE_CASES:
        paths = derive_paths(_cfg(**overrides))

        assert {key: getattr(paths, key) for key in expected} == expected, name

    # an unset run id is minted once and shared
    cfg = _cfg()
    cfg.run_id = None

    paths = derive_paths(cfg)

    assert paths.run_id in str(paths.run_dir)
    assert paths.ablations_dir == paths.run_dir / "ablations"
    assert paths.paper_dir == paths.run_dir / "paper"


def test_tilde_expands_in_task_results_root_and_explicit_output_paths(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    # task dir and results root
    paths = derive_paths(_cfg(**{"task.dir": "~/task", "paths.results_root": "~/results"}))

    assert paths.source_task_dir == tmp_path / "task"
    assert paths.task_dir == paths.run_dir / "task"
    assert paths.run_dir == tmp_path / "results" / "task" / "claude" / "20260101_000000"

    # explicit output paths
    cfg = _cfg(
        **{
            "paths.run_dir": "~/run",
            "paths.ablations_dir": "~/ablations",
            "paths.posthoc_dir": "~/posthoc",
            "paths.final_evaluation_dir": "~/final",
            "paths.paper_dir": "~/paper",
        }
    )

    paths = derive_paths(cfg)

    assert paths.run_dir == tmp_path / "run"
    assert paths.ablations_dir == tmp_path / "ablations"
    assert paths.posthoc_dir == tmp_path / "posthoc"
    assert paths.final_dir == tmp_path / "final"
    assert paths.paper_dir == tmp_path / "paper"


# --- directory_descriptor_path ------------------------------------------------------


@pytest.fixture
def descriptor(tmp_path: Path):
    (tmp_path / "directory").mkdir()
    fd = os.open(tmp_path / "directory", os.O_RDONLY | os.O_DIRECTORY)
    yield fd
    os.close(fd)


@pytest.mark.parametrize(
    ("linked", "bases", "expected"),
    [
        pytest.param(("proc/self/fd", "dev/fd"), ("proc/self/fd", "dev/fd"), "proc/self/fd", id="first-matching-base"),
        pytest.param(("dev/fd",), ("missing", "dev/fd"), "dev/fd", id="falls-back-to-dev-fd"),
        pytest.param((), ("missing",), None, id="fails-closed-without-match"),
    ],
)
def test_directory_descriptor_path_selects_base(
    tmp_path: Path, descriptor: int, linked: tuple[str, ...], bases: tuple[str, ...], expected: str | None
) -> None:
    for name in linked:
        (tmp_path / name).mkdir(parents=True)
        (tmp_path / name / str(descriptor)).symlink_to(tmp_path / "directory", target_is_directory=True)
    descriptor_bases = tuple(tmp_path / name for name in bases)

    if expected is None:
        with pytest.raises(RuntimeError, match="descriptor path"):
            directory_descriptor_path(descriptor, descriptor_bases=descriptor_bases)
    else:
        selected = directory_descriptor_path(descriptor, descriptor_bases=descriptor_bases)
        assert selected == tmp_path / expected / str(descriptor)


# --- restore_materialized_paper_memory ------------------------------------------------


def _memory_dirs(base: Path) -> tuple[Path, Path]:
    target = base / "target"
    target.mkdir(parents=True)
    return target, base / "workspace" / "global_memory"


_RESTORE_CASES = [
    (
        "merges-materialized-directory",
        {"memory.md": "existing\n"},
        {"memory.md": "existing\nresume state\n"},
        {"memory.md": "existing\nresume state\n"},
    ),
    ("creates-alias-when-materialized-absent", {}, None, {}),
    (
        "unions-divergent-jsonl-histories",
        {"memory.jsonl": '{"step":"investigation"}\n'},
        {"memory.jsonl": '{"step": "paper"}\n{"step":"investigation"}\n'},
        {"memory.jsonl": '{"step":"investigation"}\n{"step": "paper"}\n'},
    ),
]


def test_resume_restores_alias(tmp_path: Path) -> None:
    for name, canonical, materialized, expected in _RESTORE_CASES:
        target, alias = _memory_dirs(tmp_path / name)
        for file_name, content in canonical.items():
            (target / file_name).write_text(content, encoding="utf-8")
        if materialized is not None:
            alias.mkdir(parents=True)
            for file_name, content in materialized.items():
                (alias / file_name).write_text(content, encoding="utf-8")

        result = restore_materialized_paper_memory(alias, target)

        assert result.is_symlink(), name
        assert result.resolve() == target.resolve(), name
        for file_name, content in expected.items():
            assert (target / file_name).read_text(encoding="utf-8") == content, name


def test_resume_rejects_collisions_divergent_state_and_nested_symlink_escapes(
    tmp_path: Path,
) -> None:
    # a non-directory collision is rejected
    target, alias = _memory_dirs(tmp_path / "collision")
    alias.parent.mkdir()
    alias.write_text("collision\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not a symlink"):
        restore_materialized_paper_memory(alias, target)

    # divergent materialized state is rejected without mutation
    target, alias = _memory_dirs(tmp_path / "divergent")
    canonical = target / "state.bin"
    canonical.write_text("canonical\n", encoding="utf-8")
    alias.mkdir(parents=True)
    materialized = alias / "state.bin"
    materialized.write_text("divergent\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected materialized paper memory paths"):
        restore_materialized_paper_memory(alias, target)

    assert not alias.is_symlink()
    assert canonical.read_text(encoding="utf-8") == "canonical\n"
    assert materialized.read_text(encoding="utf-8") == "divergent\n"

    # a nested path is rejected without following the canonical symlink
    target, alias = _memory_dirs(tmp_path / "nested")
    outside = tmp_path / "outside"
    outside.mkdir()
    (target / "nested").symlink_to(outside, target_is_directory=True)
    (alias / "nested").mkdir(parents=True)
    (alias / "nested" / "injected.txt").write_text("injected\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected materialized paper memory paths"):
        restore_materialized_paper_memory(alias, target)

    assert not (outside / "injected.txt").exists()
    assert alias.is_dir()
    assert not alias.is_symlink()
