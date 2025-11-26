# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Optional snapshot: Prefect flow-of-three for Chapter 7.

Requires: prefect (optional dependency).
"""

from __future__ import annotations  # Future-proof typing.

from prefect import flow, task  # Prefect decorators.


@task
def fetch() -> str:
    return "2+3"  # Tiny expression payload.


@task
def parse(expr: str) -> tuple[int, int]:
    a, b = expr.split("+")  # Simple parsing.
    return int(a), int(b)  # Convert to ints.


@task
def report(a: int, b: int) -> str:
    return f"sum={a+b}"  # Surface final result.


@flow
def mini_flow() -> None:
    expr = fetch()  # Step 1: fetch expression.
    a, b = parse(expr)  # Step 2: parse operands.
    print(report(a, b))  # Step 3: print report.


if __name__ == "__main__":
    mini_flow()  # Execute flow when run directly.
