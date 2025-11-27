"""
Remote Incident Agent that drives planning and action through an MCP client with guardrails and telemetry.
"""

from __future__ import annotations

from typing import Any, Dict, List

from telemetry import Budget, RunContext, TelemetryEvent, TelemetryLogger


class RemoteIncidentAgent:
    def __init__(self, mcp_client, planner, telemetry: TelemetryLogger) -> None:
        self.client = mcp_client
        self.planner = planner
        self.telemetry = telemetry
        self.budget = Budget(tokens=2000, ms=150, dollars=0.0)
        self.max_steps = 5
        self.max_latency_ms = 150
        self.max_retries = 2

    async def observe(self, ctx: RunContext) -> Dict[str, Any]:
        """Fetch capabilities/resources from the MCP server."""
        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="observe_start",
                method="initialize",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload={},
            )
        )
        result = await self.client.initialize()
        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="observe_end",
                method="initialize",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload={"capabilities": result},
            )
        )
        return result

    async def plan(self, ctx: RunContext, observations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run planner locally under budget constraints."""
        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="plan_start",
                method="planner",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload={"observations": observations},
            )
        )
        plan = self.planner.plan(observations, self.budget)
        if len(plan) > self.max_steps:
            self.telemetry.log(
                TelemetryEvent(
                    correlation_id=ctx.correlation_id,
                    loop_id=ctx.loop_id,
                    phase="plan_guardrail",
                    method="max_steps",
                    status="error",
                    latency_ms=0,
                    budget=self.budget,
                    payload={"reason": "max_steps_exceeded", "allowed": self.max_steps},
                )
            )
            plan = plan[: self.max_steps]
        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="plan_end",
                method="planner",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload={"plan": plan},
            )
        )
        return plan

    async def act(self, ctx: RunContext, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute callTool steps via MCP client with guardrails."""
        results: List[Dict[str, Any]] = []
        cumulative_latency = 0
        failure_count = 0

        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="act_start",
                method="callTool_batch",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload={"steps": steps},
            )
        )

        for step in steps:
            if len(results) >= self.max_steps:
                self.telemetry.log(
                    TelemetryEvent(
                        correlation_id=ctx.correlation_id,
                        loop_id=ctx.loop_id,
                        phase="act_guardrail",
                        method="max_steps",
                        status="error",
                        latency_ms=0,
                        budget=self.budget,
                        payload={"reason": "max_steps_exceeded"},
                    )
                )
                break
            if step.get("type") == "callTool":
                name = step.get("name", "")
                arguments = step.get("input", {}) or {}
                try:
                    result = await self.client.call_tool(name, arguments)
                    status = result.get("status") if isinstance(result, dict) else "ok"
                except Exception as exc:  # noqa: BLE001
                    result = {"status": "error", "error": str(exc), "metrics": {"latency_ms": 0}}
                    status = "error"
                results.append({"step": step, "result": result})
                metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
                latency_ms = int(metrics.get("latency_ms", 0) or 0)
                cumulative_latency += latency_ms
                if status != "ok":
                    failure_count += 1
                if cumulative_latency > self.max_latency_ms:
                    self.telemetry.log(
                        TelemetryEvent(
                            correlation_id=ctx.correlation_id,
                            loop_id=ctx.loop_id,
                            phase="act_guardrail",
                            method="latency_budget",
                            status="error",
                            latency_ms=latency_ms,
                            budget=self.budget,
                            payload={"reason": "latency_budget_exceeded", "cumulative_ms": cumulative_latency},
                        )
                    )
                    break
                if failure_count > self.max_retries:
                    self.telemetry.log(
                        TelemetryEvent(
                            correlation_id=ctx.correlation_id,
                            loop_id=ctx.loop_id,
                            phase="act_guardrail",
                            method="max_retries",
                            status="error",
                            latency_ms=latency_ms,
                            budget=self.budget,
                            payload={"reason": "max_retries_exceeded", "failures": failure_count},
                        )
                    )
                    break
            else:
                results.append({"step": step, "result": {"ok": True}})

        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="act_end",
                method="callTool_batch",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload={"results": results, "cumulative_latency_ms": cumulative_latency},
            )
        )
        return results

    async def learn(self, ctx: RunContext) -> Dict[str, Any]:
        """Fetch recent deltas and current plan from memory resources."""
        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="learn_start",
                method="getResource_batch",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload={},
            )
        )
        deltas = await self.client.get_resource("memory://deltas/recent")
        plan = await self.client.get_resource("memory://plans/current")
        learn_result = {"deltas": deltas, "plan": plan}
        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="learn_end",
                method="getResource_batch",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload=learn_result,
            )
        )
        return learn_result

    async def run_loop(self, ctx: RunContext) -> Dict[str, Any]:
        """Run a full OPAL loop through the MCP client."""
        self.client.set_context(ctx)
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
