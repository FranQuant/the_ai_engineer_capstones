"""
Demo script: run a single remote OPAL loop against the local MCP server.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict
from pathlib import Path

from incident_planner import IncidentPlanner
from mcp_client import MCPClient
from remote_agent import RemoteIncidentAgent
from telemetry import RunContext, TelemetryLogger, new_correlation_id


async def main() -> None:
    """Connect to MCP, run one OPAL loop, print summary, then close."""
    telemetry = TelemetryLogger(Path("artifacts/telemetry.jsonl"))
    client = MCPClient(telemetry=telemetry)
    await client.connect()

    planner = IncidentPlanner(config={})
    agent = RemoteIncidentAgent(client, planner, telemetry)
    ctx = RunContext(correlation_id=new_correlation_id(), loop_id="loop-1")

    # ------------------------------------------------------------
    # Run a full OPAL loop
    # ------------------------------------------------------------
    summary: Dict[str, Any] = await agent.run_loop(ctx)

    # ------------------------------------------------------------
    # Fix #5B — write the OPAL plan into MCP memory via write_plan
    # ------------------------------------------------------------
    if "plan" in summary and summary["plan"]:
        await client.call_tool("write_plan", {"plan": summary["plan"]})

    print("=== Remote OPAL Summary ===")
    print(summary)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
