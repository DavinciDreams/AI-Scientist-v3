"""Keep in-process entrypoint scratch settings local to each test."""
import os
from pathlib import Path
import stat
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from src.investigation.runtime import shutdown_event_logging


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Give ``tmp_path`` a short root so AF_UNIX socket paths fit in 107 bytes.

    pytest's default root is ``$TMPDIR/pytest-of-<user>/pytest-N`` and xdist adds
    ``popen-gwN``; with a long user name the broker sockets tests bind under
    ``tmp_path`` exceed the kernel limit. An explicit ``--basetemp`` still wins,
    and xdist workers keep the directory the controller hands them.
    """
    if config.option.basetemp is not None or hasattr(config, "workerinput"):
        return
    from _pytest.pathlib import make_numbered_dir_with_cleanup

    root = Path(tempfile.gettempdir()) / f"agsci-{os.getuid()}"
    try:
        root.mkdir(mode=0o700, exist_ok=True)
        metadata = root.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            return
        run = make_numbered_dir_with_cleanup(
            root=root, prefix="t", keep=3, lock_timeout=172800, mode=0o700
        )
    except OSError:
        return
    config.option.basetemp = run / "pt"


@pytest.fixture(autouse=True)
def restore_runtime_storage_environment() -> Iterator[None]:
    names = (
        "AGENTIC_TEMP_ROOT", "AGENTIC_TEMP_DIR", "AGENTIC_CACHE_ROOT",
        "AGENTIC_CONTAINER_TEMP_ROOT", "TMPDIR", "TMP", "TEMP",
    )
    previous = {name: os.environ.get(name) for name in names}
    previous_tempdir = tempfile.tempdir
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        tempfile.tempdir = previous_tempdir


@pytest.fixture(autouse=True)
def release_process_event_logger() -> Iterator[None]:
    """Drop a process-wide event logger left behind by tests that drive ``_main`` directly.

    While the logger is configured, ``log_event`` emits to it instead of printing, so a leaked
    logger silently changes the behaviour of later investigation CLI logging tests.
    """
    yield
    shutdown_event_logging()


@pytest.fixture(autouse=True)
def stub_investigation_image_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep investigation CLI tests off Docker: the pinned image id is the reference itself.

    Tests of the real pinning override ``src.investigation.cli.acquire_image_lease``.
    """

    @dataclass
    class _StubLease:
        image_id: str
        tag: str

        def verify(self) -> None:
            return None

        def release(self) -> None:
            return None

    def acquire(image_reference: str, *, lease_scope: str, role: str) -> _StubLease:
        return _StubLease(image_reference, f"stub-lease:{role}")

    monkeypatch.setattr("src.investigation.cli.acquire_image_lease", acquire)


@pytest.fixture
def stub_image_leases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace Docker image resolution with in-memory leases for orchestration tests.

    Pipeline tests that drive ``src.pipeline.cli.main`` with a stubbed ``run_cmd`` do not need
    a Docker daemon or a built agent image; opt in with
    ``pytestmark = pytest.mark.usefixtures("stub_image_leases")``. Tests that exercise the real
    lease plumbing override these attributes with their own monkeypatch.
    """

    @dataclass
    class _StubLease:
        image_id: str
        tag: str

        def verify(self) -> None:
            return None

        def release(self) -> None:
            return None

    def acquire(image_reference: str, *, lease_scope: str, role: str) -> _StubLease:
        return _StubLease(image_reference, f"stub-lease:{role}")

    monkeypatch.setattr("src.pipeline.cli.acquire_image_lease", acquire)
    monkeypatch.setattr("src.pipeline.cli.resolve_immutable_image", lambda image, **_: image)


# These files drive src/scripts/task.sh end to end; every test spawns hundreds of Python
# subprocesses and takes 10-40 s on a shared host, so together they are over 80% of the suite's
# wall time. A bare ``pytest`` skips them; naming a file (or a test inside it) on the command line
# runs it as usual.
TASK_SH_END_TO_END_FILES = frozenset(
    {"test_agent_execution_pipeline.py", "test_task_drafting_robust.py"}
)


def _explicitly_requested_files(config: pytest.Config) -> set[str]:
    requested: set[str] = set()
    for argument in config.invocation_params.args:
        name = Path(str(argument).split("::", 1)[0]).name
        if name in TASK_SH_END_TO_END_FILES:
            requested.add(name)
    return requested


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    requested = _explicitly_requested_files(config)
    deselected = [
        item
        for item in items
        if item.path.name in TASK_SH_END_TO_END_FILES and item.path.name not in requested
    ]
    if not deselected:
        return
    config.hook.pytest_deselected(items=deselected)
    dropped = set(deselected)
    items[:] = [item for item in items if item not in dropped]


def pytest_report_header(config: pytest.Config) -> list[str]:
    skipped = sorted(TASK_SH_END_TO_END_FILES - _explicitly_requested_files(config))
    if not skipped:
        return []
    return [
        "task.sh end-to-end files deselected (name one on the command line to run it): "
        + ", ".join(f"tests/{name}" for name in skipped)
    ]
