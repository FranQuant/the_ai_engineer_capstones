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


def call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch tool call based on registered handlers."""
    # Deterministic stub responses only; side effects handled by caller if needed.
    return {"ok": True, "tool": name, "echo": arguments}


async def handle_session(ws, logger: TelemetryLogger, memory: IncidentMemoryStore) -> None:
    """Handle JSON-RPC messages for one websocket session."""
    del logger  # telemetry integration is out of scope for minimal implementation
    message = await ws.recv()
    request = {} if not message else message
    method = request.get("method") if isinstance(request, dict) else None
    params = request.get("params", {}) if isinstance(request, dict) else {}

    if method == "initialize":
        response = {"result": capabilities_payload(memory)}
    elif method == "getResource":
        uri = params.get("uri", "")
        cursor = params.get("cursor")
        response = {"result": get_resource(memory, uri, cursor)}
    elif method == "callTool":
        name = params.get("name", "")
        args = params.get("arguments", {}) or {}
        response = {"result": call_tool(name, args)}
    else:
        response = {"error": {"code": -32601, "message": "Method not found"}}

    await ws.send(response)


async def main() -> None:
    """Start the MCP server and listen for client connections."""
    # Minimal placeholder: no network server started in scaffold.
    return None


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
