# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 15: tiny scenario playbook generator (local).

Prints trade-offs for each scenario in a compact form.
"""

from __future__ import annotations  # Future-proof typing.

from dataclasses import dataclass  # Lightweight containers.
from typing import List  # Type hints.


# tag::ch15_playbook[]
@dataclass
class Scenario:
    id: str  # Scenario identifier.
    goal: str  # What success looks like.
    constraints: List[str]  # Time/cost/privacy constraints.
    designs: List[str]  # Candidate designs (1-2 lines each).


def print_playbook(items: List[Scenario]) -> None:
    for s in items:  # Iterate scenarios.
        print({"id": s.id, "goal": s.goal})  # Header line.
        print({"constraints": s.constraints})  # Constraints list.
        print({"designs": s.designs})  # Candidate designs.


def demo() -> None:
    items = [
        Scenario(
            id="kb-briefs",
            goal="Summarize briefs into one-liners",
            constraints=["120 chars", "no PII", "<200 ms"],
            designs=["rules+regex", "LLM small+memory"],
        ),
        Scenario(
            id="finance-notes",
            goal="Signal from short research notes",
            constraints=["allowlist tickers", "audit log", "retention policy"],
            designs=["rule cues→signal", "LLM propose cue + rule map"],
        ),
    ]
    print_playbook(items)  # Print compact forms.


if __name__ == "__main__":
    demo()  # Run demo.
# end::ch15_playbook[]

