# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 6: use the tool adapter inside a tiny loop (local only).

Shows how call_tool(...) returns latency and status without blocking the loop.
"""

from __future__ import annotations  # Future annotations.

from dataclasses import dataclass  # Lightweight container.
from typing import Dict, List  # Type hints.

try:
    from .ch06_tool_adapter import ToolSpec, call_tool  # Adapter pieces.
except ImportError:
    from ch06_tool_adapter import ToolSpec, call_tool  # Adapter pieces.
try:
    from .ch06_tool_adapter import flaky_tool  # Demo tool with one failure.
except ImportError:
    from ch06_tool_adapter import flaky_tool  # Demo tool with one failure.


@dataclass  # Minimal observation.
class Observation:
    text: str  # Raw input per turn.
    turn: int  # Loop counter.


def policy(observation: Observation) -> Dict[str, Dict[str, str]]:  # Simple router.
    """Route to the flaky tool once and then succeed on retry."""

    if observation.turn == 1:  # First turn triggers a failure.
        return {"tool": {"name": "flaky", "payload": {"n": 0}}}  # n=0 → fail.
    return {"tool": {"name": "flaky", "payload": {"n": 1}}}  # n=1 → succeed.


def run_loop(observations: List[Observation]) -> None:  # Adapter in a loop.
    """Call tools via the adapter and print latency/status per turn."""

    spec = ToolSpec(  # Configure adapter knobs.
        name="flaky", schema={"n": "int"}, timeout_ms=150, max_retries=2
    )
    for obs in observations:  # Iterate turns.
        decision = policy(obs)  # Decide tool + payload.
        payload = decision["tool"]["payload"]  # Extract payload.
        out = call_tool(spec, tool=flaky_tool, payload=payload)  # Adapter call.
        print(  # Compact telemetry.
            f"turn={obs.turn} status={out['status']} latency_ms={out['latency_ms']} "
            f"result={out['result']}"
        )


def main() -> None:  # Demo runner.
    turns = [Observation("first", 1), Observation("second", 2)]  # Two turns.
    run_loop(turns)  # Execute.


if __name__ == "__main__":
    main()
# tag::ch06_adapter_integration[]
# The integration entry point is run_loop(); it calls call_tool() and logs both
# status and latency_ms for each turn.
# end::ch06_adapter_integration[]
