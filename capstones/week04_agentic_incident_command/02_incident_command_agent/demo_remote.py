"""
Demo script: run a single remote OPAL loop against the local MCP server.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from incident_planner import IncidentPlanner
from mcp_client import MCPClient
from remote_agent import RemoteIncidentAgent
from telemetry import RunContext


async def main() -> None:
    """Connect to MCP, run one OPAL loop, print summary, then close."""
    client = MCPClient()
    await client.connect()

    planner = IncidentPlanner(config={})
    agent = RemoteIncidentAgent(client, planner)
    ctx = RunContext(correlation_id="demo", loop_id="loop-1")

    summary: Dict[str, Any] = await agent.run_loop(ctx)
    print("=== Remote OPAL Summary ===")
    print(summary)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
