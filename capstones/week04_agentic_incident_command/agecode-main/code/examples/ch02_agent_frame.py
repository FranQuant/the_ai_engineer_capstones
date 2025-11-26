# tag::ch02_agent_frame[]
# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Logging demo for Chapter 2: illustrates policy, tools, memory, environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Protocol
# TODO(Ch3-Ex1): Extend _execute_tool() with timeout_ms + max_retries and
#                exponential backoff.
# TODO(Ch3-Ex2): Add a simple tool-call budget in run_loop().
# Typing aliases for clarity.


@dataclass
class Observation:
    """Container for a single turn’s input."""

    text: str  # Natural-language description of the task or signal.
    turn: int  # Loop iteration number for traceability.


class Tool(Protocol):
    """Protocol for callables that accept a payload and return structured data."""

    def __call__(self, *, payload: Dict[str, str]) -> Dict[str, str]:
        ...  # No implementation here; concrete tools provide it.


class Memory:
    """Extremely small episodic memory store."""

    def __init__(self) -> None:
        self.episodic: List[Dict[str, str]] = []  # Append-only log of past turns.

    def write(self, *, record: Dict[str, str]) -> None:
        """Persist a dictionary describing the latest turn."""

        # Keep entries in arrival order for easy replay.
        self.episodic.append(record)

    def read(self) -> Dict[str, str]:
        """Return a lightweight summary for the policy."""

        history = " | ".join(event["observation"] for event in self.episodic)
        return {"history": history}  # Policy receives a single string snapshot.


def policy(
    observation: Observation, memory_snapshot: Dict[str, str]
) -> Dict[str, str]:
    """Decide which tool to call based on text cues and memory."""

    text = observation.text.lower()
    if "calculate" in text:
        expression = text.split("calculate", 1)[1].strip()
        return {"tool": "calculator", "payload": {"expression": expression}}
    if "remember" in text:
        return {"tool": "memory_write", "payload": {"note": observation.text}}
    # Default branch: echo the request back to the user with history context.
    reply = f"(history: {memory_snapshot.get('history', '∅')}) {observation.text}"
    return {"tool": "echo", "payload": {"message": reply}}


def _execute_tool(
    *, tools: Dict[str, Tool], name: str, payload: Dict[str, str]
) -> Dict[str, str]:
    """Call a tool with a place to add timeouts/retries.

    This is the seam for reliability work: add timeouts, retries with backoff,
    circuit breakers, and input/output validation before returning the result.
    """

    # TODO: implement timeout + retry/backoff here if a tool is flaky.
    return tools[name](payload=payload)


def run_loop(observations: List[Observation], tools: Dict[str, Tool]) -> None:
    """Wire policy, tools, and memory together with logging."""

    memory = Memory()  # Initialize the tiny episodic memory.
    for obs in observations:  # Iterate over input turns.
        snapshot = memory.read()  # Read memory before making a decision.
        decision = policy(obs, snapshot)  # Policy selects the next tool.
        tool_name = decision["tool"]  # Extract chosen tool name.
        # Execute via seam that can add timeouts/retries.
        response = _execute_tool(
            tools=tools, name=tool_name, payload=decision["payload"]
        )
        # Persist episode for later reads.
        memory.write(
            record={"observation": obs.text, "response": response["result"]}
        )
        print(
            f"turn={obs.turn} tool={tool_name} response={response['result']}"
        )  # Emit telemetry for this turn.


def _calculator(*, payload: Dict[str, str]) -> Dict[str, str]:
    """Toy calculator; eval is safe here because payload is under our control."""

    expression = payload["expression"]
    result = eval(expression, {"__builtins__": {}})  # Evaluate simple arithmetic.
    return {"result": str(result)}


def _memory_write(*, payload: Dict[str, str]) -> Dict[str, str]:
    """Pretend to persist a note and acknowledge it."""

    note = payload["note"]
    return {"result": f"noted: {note}"}


def _echo(*, payload: Dict[str, str]) -> Dict[str, str]:
    """Return the message payload as-is."""

    return {"result": payload["message"]}


if __name__ == "__main__":
    tools: Dict[str, Tool] = {
        "calculator": _calculator,
        "memory_write": _memory_write,
        "echo": _echo,
    }
    sample_observations = [
        Observation(text="Calculate 2 + 3", turn=1),
        Observation(text="Remember to send the report", turn=2),
        Observation(text="Any updates?", turn=3),
    ]
    run_loop(sample_observations, tools)
# end::ch02_agent_frame[]
