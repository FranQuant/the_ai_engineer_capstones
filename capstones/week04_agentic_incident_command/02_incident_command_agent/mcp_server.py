"""
MCP server for the Incident Command Agent.

Responsibilities:
- Expose initialize, getResource, and callTool over JSON-RPC 2.0 (WebSockets).
- Advertise tool/resource capabilities from incident_schemas and incident_memory.
- Dispatch tool calls to tool handlers; emit telemetry with correlation_id/loop_id.
- Maintain alignment with warm-up harness patterns (resource registry, tool dispatch).

TODO:
- Implement websocket server startup and session handling.
- Wire tool dispatch to real implementations or stubs.
- Validate arguments against incident_schemas.
- Integrate telemetry logger for every request/response.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from incident_memory import IncidentMemoryStore
from incident_schemas import resource_descriptions, tool_descriptions
from telemetry import TelemetryLogger


def capabilities_payload(memory: IncidentMemoryStore) -> Dict[str, Any]:
    """Return MCP capabilities including tools and resources."""
    del memory  # unused in minimal implementation
    return {
        "capabilities": {"sampling": {"available": False}},
        "tools": tool_descriptions(),
        "resources": resource_descriptions(),
    }


def get_resource(memory: IncidentMemoryStore, uri: str, cursor: str | None = None) -> Dict[str, Any]:
    """Fetch resource via memory store."""
    return memory.get_resource(uri, cursor)


def call_tool(memory: IncidentMemoryStore, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch tool call based on registered handlers."""
    if name == "create_incident":
        incident_id = arguments.get("id", "INC-001")
        return memory.update_incident(incident_id, arguments)
    if name == "add_evidence":
        return memory.write_evidence(arguments)
    if name == "append_delta":
        return memory.append_delta(arguments)
    # Deterministic echo fallback.
    return {"ok": True, "tool": name, "echo": arguments}


async def handle_session(ws, logger: TelemetryLogger, memory: IncidentMemoryStore) -> None:
    """Handle JSON-RPC messages for one websocket session."""
    del logger  # telemetry integration is out of scope for minimal implementation
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

        # Ignore notifications (no id).
        if req_id is None:
            continue

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
            args = params.get("arguments", {}) or {}
            result = call_tool(memory, name, args)
            response = {"jsonrpc": "2.0", "id": req_id, "result": result}
        else:
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": "Method not found"},
            }

        await ws.send(json.dumps(response))


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
    logger = TelemetryLogger  # placeholder, not used in minimal loop

    import websockets

    async def handler(ws):
        await handle_session(ws, logger, memory)

    server = await websockets.serve(handler, "127.0.0.1", 8765, subprotocols=None)
    print("MCP server listening on ws://127.0.0.1:8765/mcp")
    await server.wait_closed()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
