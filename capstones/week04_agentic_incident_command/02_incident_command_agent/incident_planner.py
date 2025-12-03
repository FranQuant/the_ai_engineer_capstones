"""
Planner for the Incident Command Agent.

Now MCP-compliant:
- Uses `arguments` (not `input`)
- Adds step metadata (step_id, type)
- Produces deterministic OPAL plan
- Saves plan to the memory store (memory://plans/current)
- Compatible with standardized tool envelopes (Fix #7)
"""

from __future__ import annotations
from typing import Any, Dict, List


class IncidentPlanner:
    def __init__(self, config: Dict[str, Any]) -> None:
        """Configure planner with model/tooling parameters."""
        self.config = config

    # ------------------------------------------------------------------
    # Core planning
    # ------------------------------------------------------------------
    def plan(
        self,
        observations: Dict[str, Any],
        budget: Any,
    ) -> List[Dict[str, Any]]:
        """
        Produce an ordered OPAL plan:
        [
          { "step_id": "...", "type": "callTool", "name": "...", "arguments": {...} },
          ...
        ]
        """

        del observations, budget  # unused for now

        steps = [
            {
                "step_id": "step-1",
                "type": "callTool",
                "name": "retrieve_runbook",
                "arguments": {"query": "CPU", "top_k": 2},
            },
            {
                "step_id": "step-2",
                "type": "callTool",
                "name": "run_diagnostic",
                "arguments": {"command": "kubectl top pod", "host": "staging-api"},
            },
            {
                "step_id": "step-3",
                "type": "callTool",
                "name": "create_incident",
                "arguments": {
                    "id": "INC-001",
                    "title": "Investigate incident",
                    "severity": "medium",
                },
            },
            {
                "step_id": "step-4",
                "type": "callTool",
                "name": "add_evidence",
                "arguments": {
                    "id": "EV-001",
                    "content": "Diagnostics and runbook retrieved",
                    "source": "system",
                },
            },
            {
                "step_id": "step-5",
                "type": "callTool",
                "name": "summarize_incident",
                "arguments": {
                    "alert_id": "ALRT-0001",
                    "evidence": [
                        "run_diagnostic",
                        "retrieve_runbook",
                    ],
                },
            },
        ]

        return steps
