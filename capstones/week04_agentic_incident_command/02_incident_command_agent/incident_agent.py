"""
Incident Command OPAL orchestrator with guardrails and telemetry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from incident_memory import IncidentMemoryStore
from incident_planner import IncidentPlanner
from telemetry import Budget, RunContext, TelemetryEvent, TelemetryLogger


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
        self.budget = Budget(tokens=2000, ms=150, dollars=0.0)
        self.max_steps = 5
        self.max_latency_ms = 150
        self.max_retries = 2
        self.LOCAL_TOOLS = {
            "retrieve_runbook": self._local_retrieve_runbook,
            "run_diagnostic": self._local_run_diagnostic,
            "summarize_incident": self._local_summarize_incident,
            "create_incident": self._local_create_incident,
            "add_evidence": self._local_add_evidence,
            "append_delta": self._local_append_delta,
        }

    async def observe(self, ctx: RunContext) -> Dict[str, Any]:
        """Collect capabilities and resources needed for planning."""
        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="observe_start",
                method="list_resources",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload={},
            )
        )
        resources = {"resources": self.memory.list_resources()}
        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="observe_end",
                method="list_resources",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload=resources,
            )
        )
        return resources

    async def plan(self, ctx: RunContext, observations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Produce an ordered plan (callTool and memory operations) under budget constraints."""
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
        """Execute planned steps via MCP tools and record results."""
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
                tool_fn = self.LOCAL_TOOLS.get(name)
                if not tool_fn:
                    result = {"status": "error", "error": f"Unknown local tool: {name}"}
                else:
                    result = tool_fn(arguments)
                results.append({"step": step, "result": result})
                metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
                latency_ms = int(metrics.get("latency_ms", 0) or 0)
                cumulative_latency += latency_ms
                if result.get("status") != "ok":
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

    async def learn(
        self,
        ctx: RunContext,
        observations: Dict[str, Any],
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Write deltas, summaries, and updated state back to memory."""
        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="learn_start",
                method="memory_write",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload={},
            )
        )
        for item in results:
            delta = {"action": "completed_step", "details": item}
            self.memory.write_delta(delta)
        self.memory.write_plan([item.get("step", {}) for item in results])
        learn_result = {"deltas_written": len(results)}
        self.telemetry.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="learn_end",
                method="memory_write",
                status="ok",
                latency_ms=0,
                budget=self.budget,
                payload=learn_result,
            )
        )
        return learn_result

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

    # ------------------------------------------------------------------
    # Local deterministic tool implementations (no MCP RPC)
    # ------------------------------------------------------------------

    def _local_retrieve_runbook(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query", "")).lower()
        top_k = int(args.get("top_k", 1))
        runbooks = self.memory.get_resource("memory://runbooks/index")
        matches = [
            rb for rb in runbooks
            if isinstance(rb, dict) and query in rb.get("title", "").lower()
        ]
        return {
            "status": "ok",
            "result": matches[:top_k],
            "metrics": {"latency_ms": 1},
        }

    def _local_run_diagnostic(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "ok",
            "result": {"output": "simulated diagnostics"},
            "metrics": {"latency_ms": 1},
        }

    def _local_summarize_incident(self, args: Dict[str, Any]) -> Dict[str, Any]:
        summary = "Summary generated from memory."
        return {
            "status": "ok",
            "result": summary,
            "metrics": {"latency_ms": 1},
        }

    def _local_create_incident(self, args: Dict[str, Any]) -> Dict[str, Any]:
        incident = {"id": args.get("id", "INC-LOCAL"), "title": args.get("title", "")}
        self.memory.update_incident(incident["id"], incident)
        return {
            "status": "ok",
            "result": incident,
            "metrics": {"latency_ms": 1},
        }

    def _local_add_evidence(self, args: Dict[str, Any]) -> Dict[str, Any]:
        evidence = {
            "id": args.get("id"),
            "content": args.get("content"),
            "source": args.get("source"),
        }
        self.memory.write_evidence(evidence)
        return {
            "status": "ok",
            "result": evidence,
            "metrics": {"latency_ms": 1},
        }

    def _local_append_delta(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self.memory.write_delta(args)
        return {
            "status": "ok",
            "result": args,
            "metrics": {"latency_ms": 1},
        }
