"""Best-effort, metadata-only accounting for automated review API calls."""

from __future__ import annotations

import os
import sys
import uuid
from decimal import Decimal
from pathlib import Path

from .cost_tracking import ApiTokenPrice


# OpenAI list-price snapshot checked 2026-08-30:
# https://developers.openai.com/api/docs/models/o4-mini
_O4_MINI_PRICE = ApiTokenPrice(
    input_per_million=Decimal("1.10"),
    cached_input_per_million=Decimal("0.275"),
    output_per_million=Decimal("4.40"),
    pricing_source="openai",
    pricing_status="pinned",
)
_REVIEW_TOKEN_PRICES = {
    "o4-mini": _O4_MINI_PRICE,
    "o4-mini-2025-04-16": _O4_MINI_PRICE,
}


def record_review_api_cost(
    response: object | None,
    *,
    stage: str,
    model: str,
    failed: bool = False,
) -> None:
    root = os.environ.get("AGENTIC_COST_ROOT")
    if not root:
        return
    try:
        from .cost_tracking import record_api_failure, record_api_response

        invocation_id = f"aireview:{stage}:{uuid.uuid4().hex}"
        if failed:
            record_api_failure(
                Path(root),
                phase="review",
                stage=stage,
                invocation_id=invocation_id,
                source="openai",
                model=model,
            )
        else:
            record_api_response(
                Path(root),
                response,
                phase="review",
                stage=stage,
                invocation_id=invocation_id,
                source="openai",
                token_prices=_REVIEW_TOKEN_PRICES,
            )
    except Exception as exc:
        print(f"[cost] warning: accounting unavailable ({type(exc).__name__})", file=sys.stderr)
