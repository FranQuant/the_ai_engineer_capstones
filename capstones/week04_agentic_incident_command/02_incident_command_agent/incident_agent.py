"""
Incident Command OPAL orchestrator.

Responsibilities:
- Manage session lifecycle and OPAL loops (observe → plan → act → learn).
- Coordinate memory reads/writes, tool execution, and telemetry emission.
- Enforce budgets and correlate events via correlation_id and loop_id.

TODO:
- Wire to incident_planner for LLM-based planning.
- Integrate incident_memory for resource access and memory:// updates.
- Connect to MCP server client for callTool RPCs.
- Implement budget enforcement and retry/backoff policies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from incident_memory import IncidentMemoryStore
from incident_planner import IncidentPlanner
from mcp_server import call_tool
from telemetry import Budget, RunContext, TelemetryLogger


class IncidentAgent:
    def __init__(
        self,
        memory: IncidentMemoryStore,
        planner: IncidentPlanner,
        telemetry: TelemetryLogger,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the agent with memory, planner, telemetry, and configuration."""
        self.memory = memory
        self.planner = planner
        self.telemetry = telemetry
        self.config = config or {}

    async def observe(self, ctx: RunContext) -> Dict[str, Any]:
        """Collect capabilities and resources needed for planning."""
        del ctx  # telemetry not used in minimal loop
        return {"resources": self.memory.list_resources()}

    async def plan(self, ctx: RunContext, observations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Produce an ordered plan (callTool and memory operations) under budget constraints."""
        del ctx  # telemetry not used in minimal loop
        return self.planner.plan(observations, Budget(tokens=0, ms=0, dollars=0.0))

    async def act(self, ctx: RunContext, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute planned steps via MCP tools and record results."""
        del ctx  # telemetry not used in minimal loop
        results: List[Dict[str, Any]] = []
        for step in steps:
            if step.get("type") == "callTool":
                name = step.get("name", "")
                arguments = step.get("input", {}) or {}
                result = call_tool(name, arguments)
                results.append({"step": step, "result": result})
            else:
                results.append({"step": step, "result": {"ok": True}})
        return results

    async def learn(
        self,
        ctx: RunContext,
        observations: Dict[str, Any],
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Write deltas, summaries, and updated state back to memory."""
        del ctx, observations  # unused in minimal loop
        # Record deltas for each callTool result for traceability.
        for item in results:
            delta = {"action": "completed_step", "details": item}
            self.memory.write_delta(delta)
        # Store last plan used.
        self.memory.write_plan([item.get("step", {}) for item in results])
        return {"deltas_written": len(results)}

    async def run_loop(self, ctx: RunContext) -> Dict[str, Any]:
        """Run a single OPAL loop and return loop outputs."""
        observations = await self.observe(ctx)
        plan_steps = await self.plan(ctx, observations)
        results = await self.act(ctx, plan_steps)
        learn_result = await self.learn(ctx, observations, results)
        return {
            "observations": observations,
            "plan": plan_steps,
            "results": results,
            "learn": learn_result,
        }
