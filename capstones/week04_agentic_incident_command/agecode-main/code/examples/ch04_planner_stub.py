# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 4 planner stub with simple validation.

This file simulates an LLM planner.

TODO(Ch4-Ex1): Adjust the `prompt` in main() and shape fake_llm_plan() (or
               your own make_plan()) so it yields exactly 3 steps for “parse two
               numbers and add them”.
TODO(Ch4-Ex2): Extend validate_plan() to reject missing fields or notes over
               12 words.
TODO(Ch4-Ex3): In ch04_plan_execute.py, add max_steps and a placeholder
               max_tokens limit.

Replace fake_llm_plan with a real model call later; keep the interface
identical so your loop code stays stable.
"""

from __future__ import annotations

from dataclasses import dataclass  # Lightweight records.
from typing import Dict, List  # Type hints for clarity.


@dataclass
class PlanStep:
    """One actionable step produced by the planner."""

    tool_name: str  # Name of the tool to invoke (e.g., "calculator").
    input_schema: Dict[str, str]  # Key → hint (e.g., {"expression": "str"}).
    notes: str  # Short human note (≤ 12 words).


def fake_llm_plan(prompt: str) -> List[PlanStep]:
    """Return a short plan; stands in for a real LLM call.

    The prompt is ignored here; we return a fixed, well-formed plan to show the
    interface clearly.
    """

    return [  # Fixed, deterministic sample plan.
        PlanStep(
            tool_name="extract_numbers",
            input_schema={"text": "str"},
            notes="parse two numbers",
        ),
        PlanStep(
            tool_name="calculator",
            input_schema={"expression": "str"},
            notes="add them",
        ),
        PlanStep(
            tool_name="echo",
            input_schema={"message": "str"},
            notes="report result",
        ),
    ]


def validate_plan(steps: List[PlanStep]) -> List[str]:
    """Validate required fields and brief notes.

    Returns a list of problems; empty means the plan is acceptable.
    """

    problems: List[str] = []  # Collected validation issues.
    if not (1 <= len(steps) <= 5):  # Keep plans short and readable.
        problems.append("plan must have 1..5 steps")
    for i, s in enumerate(steps, start=1):
        if not s.tool_name:
            problems.append(f"step {i}: missing tool_name")
        if not s.input_schema:
            problems.append(f"step {i}: missing input_schema")
        if len(s.notes.split()) > 12:
            problems.append(f"step {i}: notes too long")
    return problems


def render_plan(steps: List[PlanStep]) -> str:
    """Return a numbered plan for display/logging."""

    lines: List[str] = []  # Output buffer.
    for i, s in enumerate(steps, start=1):
        schema_keys = list(s.input_schema.keys())  # Ordered view for display.
        # Human-readable line.
        line = f"{i}. {s.tool_name} | schema={schema_keys} | {s.notes}"
        lines.append(line)  # Accumulate.
    return "\n".join(lines)  # Single printable block.


def main() -> None:
    """Demonstrate planning and validation."""

    prompt = (
        "You are a meticulous planner. Produce 3 steps with tool_name, "
        "input_schema, and short notes."
    )
    steps = fake_llm_plan(prompt)  # Deterministic placeholder LLM output.
    problems = validate_plan(steps)  # Run guardrails.
    if problems:  # Plan rejected path.
        print("Plan rejected:")  # Banner.
        for p in problems:  # Emit each violation.
            print(" -", p)  # Bullet.
        return
    print("Plan accepted:\n" + render_plan(steps))  # Happy path.


if __name__ == "__main__":
    main()
# tag::ch04_planner[]
# The planner entry points for the book are: fake_llm_plan, validate_plan,
# and render_plan. The loop calls these in sequence when planning is needed.
# end::ch04_planner[]
