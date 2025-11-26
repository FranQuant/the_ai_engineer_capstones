"""
Minimal MCP server for the Week-04 warm-up (Tool Harness).
Implements:
    - initialize
    - getResource
    - callTool

JSON-RPC 2.0 over WebSockets, subprotocol = "mcp".
Structured telemetry with correlation IDs and budgets.

This server exposes:
    • 3 tools (retrieve_runbook, run_diagnostic, summarize_incident)
    • 3 resource URIs (alerts/latest, runbooks/index, deltas/recent)

Aligned with:
    • TAE Week-04 Incident Command blueprint
    • MCP Agents PDF
    • engcode-main patterns
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import websockets  # pip install websockets

from schemas import TOOL_SCHEMAS, RESOURCE_FIXTURES
from telemetry import (
    Budget,
    RunContext,
    TelemetryEvent,
    TelemetryLogger,
    new_correlation_id,
    timed,
)


# ============================================================
# Server configuration
# ============================================================

HOST = "127.0.0.1"
PORT = 8765

logger = TelemetryLogger(sink=Path("samples/server_telemetry.log"))
budget = Budget()


# ============================================================
# Canned data for stub tools
# ============================================================

RUNBOOKS: List[Dict[str, Any]] = [
    {
        "id": "rb-101",
        "title": "High CPU playbook",
        "service": "staging-api",
        "steps": [
            "Check pod CPU across nodes",
            "Capture logs before restart",
            "Restart service if CPU > 90% for 5 minutes",
        ],
    },
    {
        "id": "rb-202",
        "title": "Crashloop restart",
        "service": "staging-api",
        "steps": [
            "Describe pods",
            "Gather events",
            "Redeploy with rollback flag",
        ],
    },
]

ALERT = RESOURCE_FIXTURES["memory://alerts/latest"]["alert"]

MEMORY_DELTAS: List[Dict[str, Any]] = RESOURCE_FIXTURES[
    "memory://deltas/recent"
].copy()


# ============================================================
# Capabilities payload
# ============================================================

def capabilities_payload() -> Dict[str, Any]:
    """Return MCP capabilities: tools + resource list."""
    return {
        "capabilities": {
            "tools": [
                {
                    "name": name,
                    "description": desc,
                    "schema": TOOL_SCHEMAS[name],
                }
                for name, desc in [
                    ("retrieve_runbook", "Retrieve runbook snippets by keyword."),
                    ("run_diagnostic", "Return canned diagnostics for a host."),
                    ("summarize_incident", "Summarize the incident with evidence."),
                ]
            ],
            "resources": [
                {"uri": uri, "description": "stub resource"}
                for uri in RESOURCE_FIXTURES
            ],
        }
    }


# ============================================================
# Resource handlers
# ============================================================

def get_resource(uri: str) -> Dict[str, Any]:
    """Return fixture for the requested stub resource."""
    if uri not in RESOURCE_FIXTURES:
        raise ValueError(f"Unknown resource URI: {uri}")
    return {"uri": uri, "data": RESOURCE_FIXTURES[uri]}


# ============================================================
# Tool handlers
# ============================================================

def tool_retrieve_runbook(arguments: Dict[str, Any]) -> Dict[str, Any]:
    query = str(arguments.get("query", "")).lower()
    top_k = int(arguments.get("top_k", 3))

    hits = [
        rb for rb in RUNBOOKS
        if query in rb["title"].lower()
        or any(query in s.lower() for s in rb["steps"])
    ]

    return {
        "status": "ok",
        "data": [
            {"title": rb["title"], "steps": rb["steps"], "score": 1.0 - idx * 0.1}
            for idx, rb in enumerate(hits[:top_k])
        ],
        "metrics": {"latency_ms": 1},
    }


def tool_run_diagnostic(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "ok",
        "data": {
            "command": arguments.get("command"),
            "host": arguments.get("host"),
            "stdout": "All pods healthy; CPU normalized.",
            "stderr": "",
        },
        "metrics": {"latency_ms": 5},
    }


def tool_summarize_incident(arguments: Dict[str, Any]) -> Dict[str, Any]:
    alert_id = arguments.get("alert_id") or ALERT["id"]
    evidence = arguments.get("evidence", [])

    summary = (
        f"Incident {alert_id}: reviewed evidence; recommend restart if CPU > 90% persists."
    )

    # Append to memory deltas (warm-up demo only)
    MEMORY_DELTAS.append(
        {"alert_id": alert_id, "action": "summarize_incident", "summary": summary}
    )

    return {
        "status": "ok",
        "data": {"summary": summary, "citations": evidence},
        "metrics": {"latency_ms": 2},
    }


TOOL_DISPATCH = {
    "retrieve_runbook": tool_retrieve_runbook,
    "run_diagnostic": tool_run_diagnostic,
    "summarize_incident": tool_summarize_incident,
}


# ============================================================
# Core handler (JSON-RPC over websockets)
# ============================================================

async def handle_session(ws):
    ctx = RunContext(correlation_id=new_correlation_id(), loop_id="loop-1")

    async for raw in ws:
        req = json.loads(raw)
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        try:
            if method == "initialize":
                latency_ms, result = timed(capabilities_payload)
                phase = "observe"

            elif method == "getResource":
                uri = params.get("uri", "")
                latency_ms, result = timed(get_resource, uri)
                phase = "observe"

            elif method == "callTool":
                name = params.get("name", "")
                arguments = params.get("arguments", {})
                if name not in TOOL_DISPATCH:
                    raise ValueError(f"Unknown tool: {name}")
                latency_ms, result = timed(TOOL_DISPATCH[name], arguments)
                phase = "act"

            else:
                raise ValueError(f"Unknown method: {method}")

            response = {"jsonrpc": "2.0", "id": req_id, "result": result}

            logger.log(
                TelemetryEvent(
                    correlation_id=ctx.correlation_id,
                    loop_id=ctx.loop_id,
                    phase=phase,
                    method=method or "unknown",
                    status="ok",
                    latency_ms=latency_ms,
                    budget=budget,
                    payload={"request": req, "response": result},
                )
            )

        except Exception as exc:  # noqa: BLE001
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": str(exc)},
            }

            logger.log(
                TelemetryEvent(
                    correlation_id=ctx.correlation_id,
                    loop_id=ctx.loop_id,
                    phase="act",
                    method=method or "unknown",
                    status="error",
                    latency_ms=0,
                    budget=budget,
                    payload={"request": req, "error": str(exc)},
                )
            )

        await ws.send(json.dumps(response))


# ============================================================
# Server entrypoint
# ============================================================

async def main():
    async with websockets.serve(
        handle_session,
        HOST,
        PORT,
        subprotocols=["mcp"],
    ):
        print(f"MCP harness server running at ws://{HOST}:{PORT}/mcp")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
