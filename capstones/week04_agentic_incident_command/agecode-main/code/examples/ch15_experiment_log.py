# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 15: minimal experiment log (local).

Records metrics and a one-line decision as JSON lines.
"""

from __future__ import annotations  # Future-proof typing.

import json  # Serialize lines.
from dataclasses import dataclass, asdict  # Structured records.
from typing import Dict, List  # Type hints.


# tag::ch15_experiments[]
@dataclass
class Experiment:
    scenario_id: str  # Link to a scenario.
    metrics: Dict[str, float]  # Key metrics.
    decision: str  # One-line decision summary.


def write_line(e: Experiment) -> str:
    return json.dumps(asdict(e))  # Return a JSON string.


def demo() -> None:
    e = Experiment(
        scenario_id="kb-briefs",
        metrics={"latency_ms": 145.0, "acc": 0.92},
        decision="ship rules first; revisit LLM later",
    )
    print(write_line(e))  # JSON line.


if __name__ == "__main__":
    demo()  # Run demo.
# end::ch15_experiments[]

