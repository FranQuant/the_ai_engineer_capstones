"""
JSON-RPC 2.0 WebSocket client for the Incident MCP server with telemetry.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import websockets

from telemetry import Budget, RunContext, TelemetryEvent, TelemetryLogger


class MCPClient:
    def __init__(
        self,
        uri: str = "ws://127.0.0.1:8765/mcp",
        telemetry: Optional[TelemetryLogger] = None,
        budget: Optional[Budget] = None,
    ) -> None:
        self.uri = uri
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._next_id = 1
        self.telemetry = telemetry
        self.budget = budget or Budget(tokens=2000, ms=150, dollars=0.0)
        self.ctx: Optional[RunContext] = None

    def set_context(self, ctx: RunContext) -> None:
        """Attach a run context for correlation-id aware telemetry."""
        self.ctx = ctx

    async def connect(self) -> None:
        """Open a WebSocket connection to the MCP server."""
        self._ws = await websockets.connect(self.uri)
        self._next_id = 1

    async def close(self) -> None:
        """Close the WebSocket connection if open."""
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def rpc(self, method: str, params: Dict[str, Any]) -> Any:
        """Send a JSON-RPC 2.0 request and return the result or raise on error."""
        if self._ws is None:
            raise RuntimeError("Client is not connected")
        req_id = self._next_id
        self._next_id += 1
        request = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        if self.telemetry and self.ctx:
            self.telemetry.log(
                TelemetryEvent(
                    correlation_id=self.ctx.correlation_id,
                    loop_id=self.ctx.loop_id,
                    phase="rpc_send",
                    method=method,
                    status="ok",
                    latency_ms=0,
                    budget=self.budget,
                    payload={"request": request},
                )
            )
        await self._ws.send(json.dumps(request))
        raw = await self._ws.recv()
        response = json.loads(raw)
        if self.telemetry and self.ctx:
            self.telemetry.log(
                TelemetryEvent(
                    correlation_id=self.ctx.correlation_id,
                    loop_id=self.ctx.loop_id,
                    phase="rpc_recv",
                    method=method,
                    status="ok" if "error" not in response else "error",
                    latency_ms=0,
                    budget=self.budget,
                    payload={"response": response},
                )
            )
        if "error" in response:
            raise RuntimeError(response["error"])
        return response.get("result")

    async def initialize(self) -> Any:
        """Call initialize on the MCP server."""
        return await self.rpc("initialize", {})

    async def get_resource(self, uri: str, cursor: Optional[str] = None) -> Any:
        """Call getResource on the MCP server."""
        params: Dict[str, Any] = {"uri": uri}
        if cursor is not None:
            params["cursor"] = cursor
        return await self.rpc("getResource", params)

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call callTool on the MCP server."""
        return await self.rpc("callTool", {"name": name, "arguments": arguments})
