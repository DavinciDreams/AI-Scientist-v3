"""AgentLens placeholder URL helpers."""

from __future__ import annotations

import base64
from pathlib import Path
from urllib.parse import urlencode


def build_agentlens_placeholder_url(
    *,
    base_url: str,
    run_id: str,
    node_id: str,
    source_log: str | None,
    session_id: str | None,
) -> str:
    """Build a placeholder AgentLens deep-link URL for future integration."""
    base = base_url.rstrip("/")
    params: dict[str, str] = {
        "run_id": run_id,
        "node_id": node_id,
    }
    if source_log:
        params["source_log"] = source_log
    if session_id:
        params["session_id"] = session_id
    return f"{base}/trace-placeholder?{urlencode(params)}"


def _base64url_encode(value: str) -> str:
    token = base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii")
    return token.rstrip("=")


def build_agentlens_trace_file_url(*, base_url: str, trace_file_path: Path) -> str:
    """Build AgentLens direct URL for ad-hoc trace file loading."""
    base = base_url.rstrip("/")
    token = _base64url_encode(str(trace_file_path.resolve()))
    return f"{base}/trace-file/{token}"
