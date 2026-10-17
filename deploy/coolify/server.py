"""Authenticated deployment wrapper for the inspector and run control panel."""

import base64
import binascii
import os
from pathlib import Path
import secrets

from fastapi import Request
from fastapi.responses import Response
import uvicorn

from apps.run_inspector.backend.app import create_app
from apps.run_inspector.backend.control import (
    BASE_RESERVED_KEYS,
    ControlConfig,
    TaskRoot,
    config_names,
)

DATA = Path("/data/sakanascientist")
SOURCE = Path(__file__).resolve().parents[2]


def control_config(*, launch_enabled: bool) -> ControlConfig:
    """Jobs run through ``agsci-zai``, which pins the provider, image and paths."""
    return ControlConfig(
        results_root=DATA / "results",
        jobs_dir=DATA / "jobs",
        task_roots=(
            TaskRoot("examples", SOURCE / "examples"),
            TaskRoot("tasks", DATA / "tasks", writable=True),
        ),
        command_prefix=("agsci-zai",),
        launch_enabled=launch_enabled,
        reserved_keys=BASE_RESERVED_KEYS
        | {"provider", "container.image", "container.env_file"},
        agent_locked="pi · zai/glm-5.3-flash",
        presets=config_names(SOURCE / "configs" / "preset"),
        env_scrub=("INSPECTOR_USERNAME", "INSPECTOR_PASSWORD"),
    )


def build_app(username: str, password: str, *, launch_enabled: bool = True):
    if not username or not password:
        raise ValueError("Inspector credentials must be configured")
    app = create_app(
        results_root=DATA / "results",
        frontend_dist=SOURCE / "apps/run_inspector/frontend/dist",
        agentlens_base_url="http://127.0.0.1:8787",
        control=control_config(launch_enabled=launch_enabled),
    )
    expected = f"{username}:{password}".encode()

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if request.url.path == "/api/healthz":
            return await call_next(request)
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        try:
            supplied = base64.b64decode(token, validate=True)
        except (ValueError, binascii.Error):
            supplied = b""
        if scheme.lower() != "basic" or not secrets.compare_digest(supplied, expected):
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Sakana Scientist"'})
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    return app


if __name__ == "__main__":
    uvicorn.run(
        build_app(
            os.environ["INSPECTOR_USERNAME"],
            os.environ["INSPECTOR_PASSWORD"],
            launch_enabled=os.environ.get("SCIENTIST_LAUNCH_ENABLED", "1") != "0",
        ),
        host="0.0.0.0",
        port=8765,
    )
