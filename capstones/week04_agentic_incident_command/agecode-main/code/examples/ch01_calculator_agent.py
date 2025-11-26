# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 1 calculator agent example."""

from __future__ import annotations

import operator  # Provides arithmetic operator functions (add, sub, etc.).
import re  # Supports regular expressions for parsing text.
from dataclasses import dataclass
# Supplies the dataclass decorator for simple data containers.
from typing import Callable, Dict, Optional  # Typing helpers for clarity.


# tag::calculator_agent[]
Ops: Dict[str, Callable[[float, float], float]] = {
    "add": operator.add,  # Map the word "add" to the addition operator.
    "plus": operator.add,  # Treat "plus" as addition as well.
    "sum": operator.add,  # Allow "sum" to trigger addition.
    "subtract": operator.sub,  # Words that imply subtraction map to operator.sub.
    "minus": operator.sub,
    "multiply": operator.mul,  # Multiplication keywords map to operator.mul.
    "times": operator.mul,
    "divide": operator.truediv,  # Division words use true division.
    "over": operator.truediv,
}


def calculator(a: float, b: float, op: str) -> float:
    """Apply the requested arithmetic operation to the inputs."""
    if op not in Ops:  # Reject operations that are not in the supported map.
        raise ValueError(f"unsupported op: {op}")
    return Ops[op](a, b)  # Invoke the operator associated with the keyword.


@dataclass
class Observation:
    text: str  # Raw natural-language query describing the task.


def parse(text: str) -> Optional[tuple[float, float, str]]:
    """Extract two numbers and an operation keyword from the text."""
    # Lowercase and trim whitespace for easier matching.
    normalized = text.lower().strip()
    # Collapse phrases like "divide by" into a simpler form.
    normalized = normalized.replace(" by ", " ")
    # Look for one of the supported operation keywords.
    match = re.search(
        r"(add|plus|sum|subtract|minus|multiply|times|divide|over)",
        normalized,
    )
    # Capture integer or decimal numbers.
    numbers = re.findall(r"-?\d+(?:\.\d+)?", normalized)
    # Bail out unless we find an operator and at least two numbers.
    if not match or len(numbers) < 2:
        return None
    # Convert the first two numbers into floats.
    first, second = map(float, numbers[:2])
    # Return operands plus the operation keyword.
    return first, second, match.group(1)


def policy(obs: Observation) -> str:
    """Decide whether to answer directly or use the calculator tool."""
    parsed = parse(obs.text)  # Attempt to interpret the observation text.
    if not parsed:  # If parsing fails, respond with a safe fallback message.
        return "I can only handle simple arithmetic like 'add 2 and 3'."
    a, b, op = parsed  # Unpack the parsed operands and operator keyword.
    try:
        result = calculator(a, b, op)  # Offload arithmetic to the calculator tool.
    except Exception as exc:  # Catch tool errors (e.g., division by zero).
        return f"Tool error: {exc}"
    return f"{result:g}"  # Format the numeric result compactly.


def run_demo() -> None:
    """Demonstrate the policy on a small batch of requests."""
    demos = [
        "add 2 and 3",
        "2 plus 3",
        "multiply 6 and 7",
        "divide 8 by 2",
        "subtract 10 and 3",
        "what is the capital of France?",
    ]  # Diverse prompts covering supported and unsupported tasks.
    for prompt in demos:  # Process each prompt in the sample list.
        print(f"{prompt} -> {policy(Observation(prompt))}")


if __name__ == "__main__":  # Run the demo when the script is executed directly.
    run_demo()
# end::calculator_agent[]
