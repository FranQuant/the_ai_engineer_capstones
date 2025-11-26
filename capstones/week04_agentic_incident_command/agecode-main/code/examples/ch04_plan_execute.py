# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 4: plan → validate → execute demo (no external API).

Produces a tiny plan for "add two numbers", validates it, and executes each
step via trivial tools. Swap the make_plan function for a real LLM call later.

TODO(Ch4-Ex3): Add max_steps enforcement in execute() and keep a placeholder
               max_tokens variable for future model calls.
"""

from __future__ import annotations  # Future annotations.

from dataclasses import dataclass  # Lightweight records.
from typing import Dict, List, Tuple  # Type hints.


# tag::ch04_plan_exec[]
@dataclass  # One plan step.
class PlanStep:
    tool_name: str  # Tool to call (e.g., "extract_numbers").
    input_schema: Dict[str, str]  # Key → type hint (e.g., {"text": "str"}).
    notes: str  # Short note (≤ 12 words).


def make_plan(task: str) -> List[PlanStep]:  # Deterministic 3-step plan.
    """Return 3 actionable steps for a simple add task."""

    return [  # Fixed demo plan.
        PlanStep("extract_numbers", {"text": "str"}, "parse two numbers"),  # Step 1.
        PlanStep("calculator", {"expression": "str"}, "add them"),  # Step 2.
        PlanStep("echo", {"message": "str"}, "report result"),  # Step 3.
    ]


def validate(steps: List[PlanStep]) -> List[str]:  # Guardrails.
    """Check required fields and brevity; return list of problems."""

    problems: List[str] = []  # Accumulator.
    if not (1 <= len(steps) <= 5):  # Keep plans short.
        problems.append("plan must have 1..5 steps")  # Error.
    for i, s in enumerate(steps, start=1):  # Check each step.
        if not s.tool_name:  # Missing tool.
            problems.append(f"step {i}: missing tool_name")  # Error.
        if not s.input_schema:  # Missing schema.
            problems.append(f"step {i}: missing input_schema")  # Error.
        if len(s.notes.split()) > 12:  # Overlong notes.
            problems.append(f"step {i}: notes too long")  # Error.
    return problems  # Return list.


# Tools (toy implementations)
def extract_numbers(*, text: str) -> Tuple[float, float]:  # Extract 2 numbers.
    """Find two numbers in a string and return them as floats."""

    import re  # Local import keeps global scope tidy.

    nums = re.findall(r"-?\d+(?:\.\d+)?", text)  # All numbers.
    if len(nums) < 2:  # Need at least two.
        raise ValueError("need two numbers, e.g., 'add 2 and 3'")  # Error.
    a, b = map(float, nums[:2])  # Convert two.
    return a, b  # Return.


def calculator(*, expression: str) -> float:  # Evaluate expression.
    """Evaluate a simple arithmetic expression safely (digits + ops)."""

    import re  # For validation.

    if not re.fullmatch(r"[0-9+\-*/.()\s]+", expression):  # Guard input.
        raise ValueError("invalid characters in expression")  # Error.
    return eval(expression, {"__builtins__": {}})  # Safe eval.


def echo(*, message: str) -> str:  # Identity tool.
    """Return the provided message; stands in for user‑facing output."""

    return message  # Echo back.


def execute(task: str) -> str:  # Plan → validate → execute.
    """Plan, validate, and execute steps; return the final message."""

    steps = make_plan(task)  # Create plan.
    problems = validate(steps)  # Validate.
    if problems:  # Any issues?
        raise ValueError("; ".join(problems))  # Surface errors.

    context: Dict[str, str] = {"task": task}  # Scratchpad.

    a, b = extract_numbers(text=task)  # Step 1.
    context["a"], context["b"] = str(a), str(b)  # Save.

    expr = f"{a} + {b}"  # Step 2 expression.
    result = calculator(expression=expr)  # Compute sum.
    context["result"] = str(result)  # Save.

    message = echo(message=f"sum({a}, {b}) = {result}")  # Step 3.
    return message  # Final.


def main() -> None:  # Demo.
    """Run a tiny end‑to‑end demo."""

    task = "Please add 2 and 3."  # Input.
    print(execute(task))  # Run and print.


if __name__ == "__main__":  # Entry.
    main()  # Execute.
# end::ch04_plan_exec[]
