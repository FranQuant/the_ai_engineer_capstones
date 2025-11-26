"""
Planner stub for the Incident Command Agent.

Responsibilities:
- Consume observations (alerts, deltas, evidence, capabilities).
- Produce ordered OPAL plan steps under budget/time constraints.
- Annotate steps with metadata for traceability and tool routing.

TODO:
- Integrate with LLM for dynamic planning.
- Add safety rails, retries, and budget-aware planning strategies.
- Support plan serialization to memory://plans/current.
"""

from __future__ import annotations

from typing import Any, Dict, List

from telemetry import Budget


class IncidentPlanner:
    def __init__(self, config: Dict[str, Any]) -> None:
        """Configure planner with model/tooling parameters."""
        self.config = config

    async def plan(
        self,
        observations: Dict[str, Any],
        budget: Budget,
    ) -> List[Dict[str, Any]]:
        """Return an ordered list of planned steps (callTool and memory operations)."""
        raise NotImplementedError

