# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 5: integrate MemoryStore with the Chapter 2 harness (local only).

Adds read-before / write-after behavior so memory informs policy and records
results. This keeps state visible without external services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol

try:
    # Reuse the tiny store implementation.
    from .ch05_memory_store import MemoryStore
except ImportError:
    from ch05_memory_store import MemoryStore  # type: ignore
    # Local fallback when running as a script.


@dataclass
class Observation:
    text: str  # Raw input for this loop turn.
    turn: int  # Loop counter for traceability in logs.


class Tool(Protocol):
    def __call__(self, *, payload: Dict[str, str]) -> Dict[str, str]:
        """Shape of tools."""



def policy(observation: Observation, memory_snippet: str) -> Dict[str, str]:
    """Choose a tool using both the observation and memory snippet."""

    text = observation.text.lower()  # Normalize for simple checks.
    if "calculate" in text or "add" in text:  # Route math to calculator.
        # Combine current text and memory to form an expression string.
        expr = " ".join(s for s in [observation.text, memory_snippet] if s)
        return {"tool": "calculator", "payload": {"expression": expr}}
    # Default: just echo the message as the result.
    return {"tool": "echo", "payload": {"message": observation.text}}


def run_loop_with_memory(
    observations: List[Observation],
    tools: Dict[str, Tool],
) -> None:
    """Run the loop with a tiny memory store wired in."""

    store = MemoryStore()  # Create empty episodic memory.
    for obs in observations:  # Process each incoming turn.
        # Read: retrieve top‑1 relevant note to show how memory informs policy.
        top = store.retrieve_topk(query=obs.text, k=1)
        memory_snippet = top[0][0] if top else ""
        # Decide next action based on current text + memory.
        decision = policy(obs, memory_snippet)
        tool_name = decision["tool"]
        # Execute the chosen tool with the policy‑provided payload.
        out = tools[tool_name](payload=decision["payload"])
        # Write: persist observation + a short result string.
        store.add(text=f"{obs.text} -> {out['result']}")
        # Emit a concise log line for traceability.
        msg = (
            f"turn={obs.turn} memory='{memory_snippet}' "
            f"tool={tool_name} -> {out['result']}"
        )
        print(msg)


def _calculator(*, payload: Dict[str, str]) -> Dict[str, str]:
    """Add the first two numbers found in the expression string."""

    import re

    text = payload["expression"]
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) < 2:
        return {"result": "no numbers found"}
    a, b = map(float, nums[:2])
    return {"result": str(a + b)}


def _echo(*, payload: Dict[str, str]) -> Dict[str, str]:
    """Return the message field verbatim as the result."""

    return {"result": payload["message"]}


def main() -> None:
    """Run a tiny two‑turn demo.

    Turn 1 writes a note; Turn 2 reads it and uses it in the calculation.
    """

    tools: Dict[str, Tool] = {"calculator": _calculator, "echo": _echo}
    turns = [
        Observation(text="Remember 2 and 3", turn=1),  # Seed memory with numbers.
        Observation(text="Calculate sum", turn=2),      # Read back and compute.
    ]
    run_loop_with_memory(turns, tools)


if __name__ == "__main__":
    main()
# tag::ch05_mem_integration[]
# The integration entry point is run_loop_with_memory(); it reads the top‑1 note
# before policy and writes a short derived fact after tools.
# end::ch05_mem_integration[]
