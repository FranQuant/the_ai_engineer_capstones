# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 7: small ETL-style demo using the tiny workflow engine.

Shows dependency ordering, retries, and compact per-task logs.
"""

from __future__ import annotations  # Future-proof typing.

import json  # Emit compact JSON logs per task.
from typing import Any, Dict  # Type hints.

try:
    # Import engine and types from same folder.
    from .ch07_workflow_engine import Task, Workflow
except ImportError:
    # Fallback when executed as a script.
    from ch07_workflow_engine import Task, Workflow


# tag::ch07_demo[]
def fetch_numbers(*, context: Dict[str, Any]) -> Dict[str, Any]:
    return {"raw": "9,3"}  # Tiny payload, two numbers as CSV.


def parse_numbers(*, context: Dict[str, Any]) -> Dict[str, Any]:
    raw = context.get("fetch_numbers.raw", "0,0")  # Upstream field.
    a, b = [int(x) for x in raw.split(",")]  # Convert to ints.
    return {"a": a, "b": b}  # Structured output.


def divide(*, context: Dict[str, Any]) -> Dict[str, Any]:
    a = context.get("parse_numbers.a", 1)  # Read a.
    b = context.get("parse_numbers.b", 1)  # Read b.
    if b == 0:  # Guard against division by zero.
        raise ValueError("b must be non-zero")  # Clear error message.
    return {"result": a / b}  # Floating-point division.


def main() -> None:
    tasks = [
        Task(name="fetch_numbers", fn=fetch_numbers),  # First step.
        Task(name="parse_numbers", fn=parse_numbers, deps=["fetch_numbers"]),
        Task(name="divide", fn=divide, deps=["parse_numbers"], max_retries=0),
    ]  # Build small DAG.
    wf = Workflow(tasks)  # Create workflow engine.
    results = wf.run(context={})  # Execute with empty input.
    for name, res in results.items():  # Print compact JSON lines.
        print(
            json.dumps(
                {
                    "task": name,
                    "status": res.status,
                    "latency_ms": res.latency_ms,
                    "deps": tasks[[t.name for t in tasks].index(name)].deps,
                }
            )
        )


if __name__ == "__main__":
    main()  # Run the demo when executed directly.
# end::ch07_demo[]
