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
    raise NotImplementedError


def get_resource(memory: IncidentMemoryStore, uri: str, cursor: str | None = None) -> Dict[str, Any]:
    """Fetch resource via memory store."""
    raise NotImplementedError


def call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch tool call based on registered handlers."""
    raise NotImplementedError


async def handle_session(ws, logger: TelemetryLogger, memory: IncidentMemoryStore) -> None:
    """Handle JSON-RPC messages for one websocket session."""
    raise NotImplementedError


async def main() -> None:
    """Start the MCP server and listen for client connections."""
    raise NotImplementedError


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
