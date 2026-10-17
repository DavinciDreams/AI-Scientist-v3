"""Authenticated deployment wrapper for the upstream read-only inspector."""

import base64
import binascii
import os
from pathlib import Path
import secrets

from fastapi import Request
from fastapi.responses import Response
import uvicorn

from apps.run_inspector.backend.app import create_app


def build_app(username: str, password: str):
    if not username or not password:
        raise ValueError("Inspector credentials must be configured")
    app = create_app(
        results_root=Path("/data/sakanascientist/results"),
        frontend_dist=Path(__file__).resolve().parents[2]
        / "apps/run_inspector/frontend/dist",
        agentlens_base_url="http://127.0.0.1:8787",
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
        build_app(os.environ["INSPECTOR_USERNAME"], os.environ["INSPECTOR_PASSWORD"]),
        host="0.0.0.0",
        port=8765,
    )
