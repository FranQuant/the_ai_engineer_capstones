"""
MCP server for the Incident Command Agent (Week 04).

Implements initialize/getResource/callTool with:
- Deterministic tool handlers (retrieve_runbook, run_diagnostic, summarize_incident, create_incident, add_evidence, append_delta).
- Resource surfaces: alerts/latest, runbooks/index, memory/deltas plus incident memory resources.
- Schema validation on callTool.
- Structured telemetry per request/response with correlation_id.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import websockets

from incident_memory import IncidentMemoryStore
from incident_schemas import get_tool_schemas, resource_descriptions, tool_descriptions
from telemetry import Budget, RunContext, TelemetryEvent, TelemetryLogger, new_correlation_id

# ---------------------------------------------------------------------------
# Canned fixtures for deterministic behavior
# ---------------------------------------------------------------------------

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

ALERT = {
    "id": "ALRT-0001",
    "service": "staging-api",
    "symptom": "CPU spike on node-3",
    "severity": "high",
    "detected_at": "2025-11-23T09:00:00Z",
}

RESOURCE_FIXTURES: Dict[str, Any] = {
    "memory://alerts/latest": ALERT,
    "memory://runbooks/index": RUNBOOKS,
}

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def validate_arguments(schema: Dict[str, Any], arguments: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Minimal JSON schema validator for required fields and primitive types."""
    errors: Dict[str, str] = {}

    for field in schema.get("required", []):
        if field not in arguments:
            errors[field] = "Missing required field"

    properties = schema.get("properties", {})
    for name, value in arguments.items():
        if name not in properties:
            continue
        expected_type = properties[name].get("type")
        if expected_type and not _type_matches(value, expected_type):
            errors[name] = f"Expected {expected_type}"
        if expected_type == "integer":
            minimum = properties[name].get("minimum")
            maximum = properties[name].get("maximum")
            if minimum is not None and value < minimum:
                errors[name] = f"Must be >= {minimum}"
            if maximum is not None and value > maximum:
                errors[name] = f"Must be <= {maximum}"
    return (len(errors) == 0, errors)


# ---------------------------------------------------------------------------
# Tool handlers (deterministic)
# ---------------------------------------------------------------------------


def _envelope(data: Dict[str, Any], latency_ms: int, cost_tokens: int = 10) -> Dict[str, Any]:
    return {
        "status": "ok",
        "data": data,
        "metrics": {"latency_ms": latency_ms, "cost_tokens": cost_tokens},
    }


def tool_retrieve_runbook(arguments: Dict[str, Any]) -> Dict[str, Any]:
    query = str(arguments.get("query", "")).lower()
    top_k = int(arguments.get("top_k", 3))
    hits = [
        rb
        for rb in RUNBOOKS
        if query in rb["title"].lower() or any(query in step.lower() for step in rb["steps"])
    ]
    data = {"results": hits[:top_k]}
    return _envelope(data, latency_ms=5)


def tool_run_diagnostic(arguments: Dict[str, Any]) -> Dict[str, Any]:
    data = {
        "command": arguments.get("command"),
        "host": arguments.get("host"),
        "stdout": "All pods healthy; CPU normalized.",
        "stderr": "",
    }
    return _envelope(data, latency_ms=7)


def tool_summarize_incident(arguments: Dict[str, Any], memory: IncidentMemoryStore) -> Dict[str, Any]:
    alert_id = arguments.get("alert_id") or ALERT["id"]
    evidence = arguments.get("evidence", [])
    summary = (
        f"Incident {alert_id}: CPU spikes observed on staging-api. "
        "Diagnostics show pods healthy and CPU normalized. "
        "Recommend restart if sustained > 90% for 5 minutes. "
        "Capture logs before restart; monitor for recurrence."
    )
    delta = {"alert_id": alert_id, "action": "summarize_incident", "summary": summary, "evidence": evidence}
    memory.append_delta(delta)
    return _envelope({"summary": summary, "citations": evidence}, latency_ms=6)


TOOL_DISPATCH = {
    "retrieve_runbook": tool_retrieve_runbook,
    "run_diagnostic": tool_run_diagnostic,
}

SERVER_BUDGET = Budget(tokens=2000, ms=150, dollars=0.0)

# ---------------------------------------------------------------------------
# Resource handling
# ---------------------------------------------------------------------------


def capabilities_payload(memory: IncidentMemoryStore) -> Dict[str, Any]:
    """Return MCP capabilities including tools and resources."""
    del memory
    return {
        "capabilities": {"sampling": {"available": False}},
        "tools": tool_descriptions(),
        "resources": resource_descriptions(),
    }


def get_resource(memory: IncidentMemoryStore, uri: str, cursor: str | None = None) -> Dict[str, Any]:
    """Fetch resource via memory store or canned fixtures."""
    if uri in RESOURCE_FIXTURES:
        return RESOURCE_FIXTURES[uri]
    if uri == "memory://memory/deltas":
        return {"items": memory.get_resource("memory://deltas/recent").get("items", [])}
    return memory.get_resource(uri, cursor)


def call_tool(memory: IncidentMemoryStore, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch tool call based on registered handlers with deterministic envelopes."""
    if name in ("retrieve_runbook", "run_diagnostic"):
        return TOOL_DISPATCH[name](arguments)
    if name == "summarize_incident":
        return tool_summarize_incident(arguments, memory)
    if name == "create_incident":
        incident_id = arguments.get("id", "INC-001")
        result = memory.update_incident(incident_id, arguments)
        return _envelope(result, latency_ms=4)
    if name == "add_evidence":
        result = memory.write_evidence(arguments)
        return _envelope(result, latency_ms=3)
    if name == "append_delta":
        result = memory.append_delta(arguments)
        return _envelope(result, latency_ms=2)
    return _envelope({"echo": arguments, "tool": name}, latency_ms=1)


# ---------------------------------------------------------------------------
# Session handling with telemetry and validation
# ---------------------------------------------------------------------------


def _validation_error_response(req_id: Any, details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32602, "message": "Invalid params", "data": details},
    }


async def handle_session(ws, logger: TelemetryLogger, memory: IncidentMemoryStore) -> None:
    """Handle JSON-RPC messages for one websocket session."""
    ctx = RunContext(correlation_id=new_correlation_id(), loop_id="loop-1")
    tool_schemas = get_tool_schemas()

    async for raw in ws:
        try:
            request = json.loads(raw)
        except Exception:
            continue

        if not isinstance(request, dict):
            continue

        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {}) or {}

        # Ignore notifications
        if req_id is None:
            continue

        phase = "observe" if method in ("initialize", "getResource") else "act"
        response: Dict[str, Any]
        status = "ok"
        start = time.perf_counter()

        try:
            if method == "initialize":
                result = capabilities_payload(memory)
                response = {"jsonrpc": "2.0", "id": req_id, "result": result}
            elif method == "getResource":
                uri = params.get("uri", "")
                cursor = params.get("cursor")
                result = get_resource(memory, uri, cursor)
                response = {"jsonrpc": "2.0", "id": req_id, "result": result}
            elif method == "callTool":
                name = params.get("name", "")
                arguments = params.get("arguments", {}) or {}
                schema = tool_schemas.get(name)
                if schema:
                    valid, errors = validate_arguments(schema, arguments)
                    if not valid:
                        status = "error"
                        response = _validation_error_response(req_id, errors)
                        await ws.send(json.dumps(response))
                        logger.log(
                            TelemetryEvent(
                                correlation_id=ctx.correlation_id,
                                loop_id=ctx.loop_id,
                                phase=phase,
                                method=name or "unknown",
                                status="error",
                                latency_ms=0,
                                budget=SERVER_BUDGET,
                                payload={"request": request, "errors": errors},
                            )
                        )
                        continue
                result = call_tool(memory, name, arguments)
                response = {"jsonrpc": "2.0", "id": req_id, "result": result}
            else:
                status = "error"
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": "Method not found"},
                }
        except Exception as exc:  # noqa: BLE001
            status = "error"
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": str(exc)},
            }

        latency_ms = int((time.perf_counter() - start) * 1000)

        logger.log(
            TelemetryEvent(
                correlation_id=ctx.correlation_id,
                loop_id=ctx.loop_id,
                phase=phase,
                method=method or "unknown",
                status=status,
                latency_ms=latency_ms,
                budget=SERVER_BUDGET,
                payload={"request": request, "response": response},
            )
        )

        await ws.send(json.dumps(response))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def main() -> None:
    """Start the MCP server and listen for client connections."""
    memory = IncidentMemoryStore(
        {
            "incidents": {"INC-000": {"id": "INC-000", "title": "Seed incident", "severity": "low"}},
            "evidence": {"EV-000": {"id": "EV-000", "content": "Seed evidence", "source": "seed"}},
            "deltas": [],
            "plans": {},
        }
    )
    logger = TelemetryLogger(Path("artifacts/telemetry.jsonl"))

    async def handler(ws):
        await handle_session(ws, logger, memory)

    server = await websockets.serve(handler, "127.0.0.1", 8765, subprotocols=None)
    print("MCP server listening on ws://127.0.0.1:8765/mcp")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
