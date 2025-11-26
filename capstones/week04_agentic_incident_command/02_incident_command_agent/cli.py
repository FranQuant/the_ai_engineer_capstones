"""
CLI entrypoint for the Incident Command Agent.

TODO:
- Wire CLI arguments to IncidentAgent lifecycle and loop execution.
- Add replay/debug flags and configuration loading.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from incident_agent import IncidentAgent


async def main(argv: Optional[list[str]] = None) -> None:
    """Launch the incident agent with provided CLI arguments."""
    raise NotImplementedError


if __name__ == "__main__":
    asyncio.run(main())
