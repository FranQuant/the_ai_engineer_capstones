"""
Client/orchestrator for the MCP Tool Harness warm-up.

Runs a single Observe → Plan → Act → Learn (OPAL) loop against
`mcp_tool_harness_server.py`:

    1. initialize
    2. getResource(memory://alerts/latest)
    3. plan a fixed sequence of callTool invocations
    4. execute tools
    5. log a memory delta

All telemetry is written to JSONL in `samples/client_telemetry.log`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import websockets  # pip install websockets

from telemetry import (
    Budget,
    RunContext,
    TelemetryEvent,
    TelemetryLogger,
    new_correlation_id,
)

ENDPOINT = "ws://127.0.0.1:8765/mcp"

logger = TelemetryLogger(sink=Path("samples/client_telemetry.log"))
budget = Budget()


# ============================================================
# Low-level JSON-RPC helper
# ============================================================

async def rpc(ws, message: Dict[str, Any]) -> Dict[str, Any]:
    """Send one JSON-RPC request and await a response."""
    await ws.send(json.dumps(message))
    raw = await ws.recv()
    return json.loads(raw)


# ============================================================
# OPAL phases
# ============================================================

async def observe(ws, ctx: RunContext) -> Dict[str, Any]:
    """initialize + getResource(alerts/latest) with telemetry."""
    caps = await rpc(
        ws,
        {
            "jsonrpc": "2.0",
            "id": "init-1",
            "method": "initialize",
            "params": {"clientName": "incident-cli", "clientVersion": "0.1.0"},
        },
    )

    alerts = await rpc(
        ws,
        {
            "jsonrpc": "2.0",
            "id": "res-1",
            "method": "getResource",
            "params": {"uri": "memory://alerts/latest", "cursor": None},
        },
    )

    logger.log(
        TelemetryEvent(
            correlation_id=ctx.correlation_id,
            loop_id=ctx.loop_id,
            phase="observe",
            method="initialize",
            status="ok",
            latency_ms=0,  # warm-up: server logs real latencies
            budget=budget,
            payload={"caps": caps},
        )
    )

    logger.log(
        TelemetryEvent(
            correlation_id=ctx.correlation_id,
            loop_id=ctx.loop_id,
            phase="observe",
            method="getResource",
            status="ok",
            latency_ms=0,
            budget=budget,
            payload={"alerts": alerts},
        )
    )

    return alerts


def plan(alerts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a deterministic plan (sequence of callTool steps) based on the alert.
    In the full capstone this will be replaced by a richer planner.
    """
    alert_id = (
        alerts.get("result", {})
        .get("data", {})
        .get("alert", {})
        .get("id", "")
    )

    return [
        {
            "method": "callTool",
            "name": "retrieve_runbook",
            "arguments": {"query": "CPU", "top_k": 2},
        },
        {
            "method": "callTool",
            "name": "run_diagnostic",
            "arguments": {
                "command": "kubectl get pods",
                "host": "staging-api",
            },
        },
        {
            "method": "callTool",
            "name": "summarize_incident",
            "arguments": {
                "alert_id": alert_id,
                "evidence": ["run_diagnostic"],
            },
        },
    ]


async def act(
    ws,
    ctx: RunContext,
    steps: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Execute planned callTool steps, log telemetry for each."""
    results: List[Dict[str, Any]] = []

    for idx, step in enumerate(steps, start=1):
        req_id = f"tool-{idx}"
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": step["method"],
            "params": {
                "name": step["name"],
                "arguments": step["arguments"],
            },
        }

        out = await rpc(ws, payload)
        results.append(out)

        logger.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase="act",
                method=step["name"],
                status="ok" if "result" in out else "error",
                latency_ms=out.get("result", {})
                .get("metrics", {})
                .get("latency_ms", 0),
                budget=budget,
                payload={"request": payload, "response": out},
            )
        )

    return results


def learn(ctx: RunContext, alerts: Dict[str, Any], tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    alert_id = alerts.get("result", {}).get("data", {}).get("alert", {}).get("id", "")

    # Normalize tool result shapes
    normalized = []
    for r in tool_results:
        if isinstance(r, dict):
            # case 1: correct shape
            if "result" in r:
                normalized.append(r["result"])
            # case 2: server returned raw "response" only
            elif "response" in r:
                normalized.append(r["response"])
            else:
                normalized.append({})
        else:
            normalized.append({})

    # Extract the summary from normalized results
    summary = next(
        (
            r.get("data", {}).get("summary")
            for r in normalized
            if isinstance(r, dict) and "data" in r and "summary" in r["data"]
        ),
        "No summary",
    )

    delta = {"alert_id": alert_id, "action": "loop_complete", "summary": summary}

    logger.log(
        TelemetryEvent(
            correlation_id=ctx.correlation_id,
            loop_id=ctx.loop_id,
            phase="learn",
            method="memory_write",
            status="ok",
            latency_ms=0,
            budget=budget,
            payload={"delta": delta},
        )
    )
    return delta


# ============================================================
# Entrypoint
# ============================================================

async def main():
    ctx = RunContext(
        correlation_id=new_correlation_id(),
        loop_id="loop-1",
    )

    async with websockets.connect(ENDPOINT, subprotocols=["mcp"]) as ws:
        alerts = await observe(ws, ctx)
        steps = plan(alerts)
        results = await act(ws, ctx, steps)
        delta = learn(ctx, alerts, results)

        print(
            json.dumps(
                {
                    "delta": delta,
                    "results": results,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
