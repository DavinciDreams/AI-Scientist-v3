"""Deployment authentication must cover API routes and static frontend assets."""

from fastapi.testclient import TestClient
import pytest

from deploy.coolify.server import build_app
from src.agsci.doctor import Host, OK, check_presets
from src.pipeline.cli import load_pipeline_config


def test_inspector_requires_credentials_but_allows_healthcheck() -> None:
    with TestClient(build_app("owner", "test-password")) as client:
        assert client.get("/api/healthz").status_code == 200
        for path in ("/", "/api/runs", "/assets/app.js"):
            assert client.get(path).status_code == 401
        assert client.post("/api/refresh").status_code == 401
        assert client.get("/api/runs", auth=("owner", "wrong")).status_code == 401
        response = client.get("/api/runs", auth=("owner", "test-password"))
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"


def test_control_panel_launches_through_the_zai_wrapper_behind_auth() -> None:
    with TestClient(build_app("owner", "test-password")) as client:
        body = {"kind": "run", "task_path": "/x", "overrides": []}
        same_origin = {"sec-fetch-site": "same-origin"}
        assert client.post("/api/jobs", json=body, headers=same_origin).status_code == 401
        assert client.post("/api/tasks", json={}, headers=same_origin).status_code == 401
        config = client.get("/api/control/config", auth=("owner", "test-password")).json()
        assert config["launch_enabled"] is True
        assert config["agent_locked"]
        assert {"provider", "container.image", "container.env_file"} <= set(config["reserved_keys"])
    with TestClient(build_app("owner", "test-password", launch_enabled=False)) as client:
        config = client.get("/api/control/config", auth=("owner", "test-password")).json()
        assert config["launch_enabled"] is False


def test_inspector_fails_closed_without_password() -> None:
    with pytest.raises(ValueError):
        build_app("owner", "")


def test_provider_preset_does_not_require_a_research_task() -> None:
    assert check_presets(Host()).status == OK
    config = load_pipeline_config(["provider=zai"])
    assert config.agent.name == "pi"
    assert config.agent.model == "zai/glm-5.3-flash"
    assert config.container.enabled
