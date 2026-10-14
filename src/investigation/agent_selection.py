"""Agent harness selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from .config import AgentConfig, InvestigationConfig


@dataclass(frozen=True)
class AgentSelection:
    """Concrete agent harness and optional model for a stage."""

    agent: str
    model: str | None


def resolve_agent_model(agent_config: AgentConfig, agent: str) -> str | None:
    """Resolve model override for a concrete agent."""
    per_agent_model = agent_config.models.get(agent)
    if isinstance(per_agent_model, str) and per_agent_model.strip():
        return per_agent_model.strip()
    if isinstance(agent_config.model, str) and agent_config.model.strip():
        return agent_config.model.strip()
    return None


def resolve_cycle_agent(
    cfg: InvestigationConfig,
    outer_state: dict[str, Any],
    *,
    cycle_index: int,
) -> AgentSelection:
    """Select the concrete agent harness for an experiment cycle."""
    if not cfg.agent.pool:
        return AgentSelection(
            agent=cfg.agent.name,
            model=resolve_agent_model(cfg.agent, cfg.agent.name),
        )

    assignments_raw = outer_state.setdefault("agent_cycle_assignments", {})
    if not isinstance(assignments_raw, dict):
        assignments_raw = {}
        outer_state["agent_cycle_assignments"] = assignments_raw

    cycle_key = str(cycle_index)
    assigned = assignments_raw.get(cycle_key)
    if not isinstance(assigned, str) or not assigned.strip():
        assigned = random.choice(list(cfg.agent.pool))
        assignments_raw[cycle_key] = assigned

    return AgentSelection(
        agent=assigned,
        model=resolve_agent_model(cfg.agent, assigned),
    )
