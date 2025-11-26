# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 8: minimal planner-critic chat over a tiny message bus.

All local; messages are small dicts for clarity and logging.
"""

from __future__ import annotations  # Future-proof typing.

from dataclasses import dataclass  # Lightweight containers.
from typing import Dict, List  # Type hints.


# tag::ch08_chat[]
@dataclass  # Represent a message exchanged between agents.
class Msg:
    role: str  # Who is sending ("planner" | "critic").
    content: str  # Short text content.
    tags: List[str]  # Structured labels (e.g., ["revise"], ["approve"]).


class Bus:
    def __init__(self) -> None:
        self.log: List[Msg] = []  # Keep a transcript.

    def send(self, msg: Msg) -> None:  # Append to log.
        self.log.append(msg)
        print({"role": msg.role, "tags": msg.tags, "content": msg.content})


def planner(last: Msg | None) -> Msg:
    if last is None:  # First move: propose a tiny plan.
        return Msg(
            role="planner",
            content="step1: parse numbers; step2: divide; step3: report",
            tags=["proposal"],
        )
    # If the critic asked for a revision, shorten the plan.
    if "revise" in last.tags:
        return Msg(
            role="planner",
            content="step1: parse; step2: divide",
            tags=["proposal"],
        )
    # Otherwise, confirm and hand off.
    return Msg(role="planner", content="ready", tags=["handoff"])


def critic(prev: Msg) -> Msg:
    text = prev.content  # Inspect the proposal.
    if "report" not in text:  # Require a final reporting step.
        return Msg(
            role="critic",
            content="please include report step",
            tags=["revise"],
        )
    return Msg(role="critic", content="looks good", tags=["approve"])


def run_chat(max_turns: int = 4) -> None:
    bus = Bus()  # Initialize message bus.
    last: Msg | None = None  # No previous message.
    for turn in range(max_turns):  # Fixed budget loop.
        if turn % 2 == 0:  # Planner speaks on even turns.
            m = planner(last)  # Produce next message.
        else:  # Critic speaks on odd turns.
            assert last is not None  # Guard for type checkers.
            m = critic(last)  # Respond to planner.
        bus.send(m)  # Log and print message.
        if "approve" in m.tags:  # Stop rule: critic approved.
            break  # Exit the loop early.
        last = m  # Update last message.


if __name__ == "__main__":
    run_chat()  # Run when executed directly.
# end::ch08_chat[]
