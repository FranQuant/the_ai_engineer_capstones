"""
Remote Incident Agent that drives planning and action through an MCP client.
"""

from __future__ import annotations

from typing import Any, Dict, List

from telemetry import Budget


class RemoteIncidentAgent:
    def __init__(self, mcp_client, planner) -> None:
        self.client = mcp_client
        self.planner = planner

    async def observe(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch capabilities/resources from the MCP server."""
        del ctx  # unused in minimal implementation
        return await self.client.initialize()

    async def plan(self, ctx: Dict[str, Any], observations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run planner locally with a zeroed budget."""
        del ctx  # unused in minimal implementation
        return self.planner.plan(observations, Budget(tokens=0, ms=0, dollars=0.0))

    async def act(self, ctx: Dict[str, Any], steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute callTool steps via MCP client."""
        del ctx  # unused in minimal implementation
        results: List[Dict[str, Any]] = []
        for step in steps:
            if step.get("type") == "callTool":
                name = step.get("name", "")
                arguments = step.get("input", {}) or {}
                result = await self.client.call_tool(name, arguments)
                results.append({"step": step, "result": result})
            else:
                results.append({"step": step, "result": {"ok": True}})
        return results

    async def learn(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch recent deltas and current plan from memory resources."""
        del ctx  # unused in minimal implementation
        deltas = await self.client.get_resource("memory://deltas/recent")
        plan = await self.client.get_resource("memory://plans/current")
        return {"deltas": deltas, "plan": plan}

    async def run_loop(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Run a full OPAL loop through the MCP client."""
        observations = await self.observe(ctx)
        plan_steps = await self.plan(ctx, observations)
        results = await self.act(ctx, plan_steps)
        learn_result = await self.learn(ctx)
        return {
            "observations": observations,
            "plan": plan_steps,
            "results": results,
            "learn": learn_result,
        }
