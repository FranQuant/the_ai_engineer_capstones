"""
CLI entrypoint for the Incident Command Agent.

TODO:
- Wire CLI arguments to IncidentAgent lifecycle and loop execution.
- Add replay/debug flags and configuration loading.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from incident_agent import IncidentAgent
from incident_memory import IncidentMemoryStore
from incident_planner import IncidentPlanner
from telemetry import RunContext, TelemetryLogger, new_correlation_id


async def main(argv: Optional[list[str]] = None) -> None:
    """Launch the incident agent with provided CLI arguments."""
    del argv  # unused in minimal CLI
    memory = IncidentMemoryStore()
    planner = IncidentPlanner(config={})
    telemetry = TelemetryLogger(Path("telemetry.log"))
    agent = IncidentAgent(memory, planner, telemetry)

    ctx = RunContext(correlation_id=new_correlation_id(), loop_id="loop-1")
    summary = await agent.run_loop(ctx)
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
