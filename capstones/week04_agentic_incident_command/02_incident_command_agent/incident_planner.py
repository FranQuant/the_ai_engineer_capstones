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


class IncidentPlanner:
    def __init__(self, config: Dict[str, Any]) -> None:
        """Configure planner with model/tooling parameters."""
        self.config = config

    def plan(
        self,
        observations: Dict[str, Any],
        budget: Any,  # budget unused in minimal implementation
    ) -> List[Dict[str, Any]]:
        """Return an ordered list of planned steps (callTool and memory operations)."""
        del observations, budget  # unused in minimal implementation
        return [
            {
                "type": "callTool",
                "name": "retrieve_runbook",
                "input": {"query": "CPU", "top_k": 2},
            },
            {
                "type": "callTool",
                "name": "run_diagnostic",
                "input": {"command": "kubectl top pod", "host": "staging-api"},
            },
            {
                "type": "callTool",
                "name": "create_incident",
                "input": {"id": "INC-001", "title": "Investigate incident", "severity": "medium"},
            },
            {
                "type": "callTool",
                "name": "add_evidence",
                "input": {"id": "EV-001", "content": "Diagnostics and runbook retrieved", "source": "system"},
            },
            {
                "type": "callTool",
                "name": "summarize_incident",
                "input": {"alert_id": "ALRT-0001", "evidence": ["run_diagnostic", "retrieve_runbook"]},
            },
        ]
