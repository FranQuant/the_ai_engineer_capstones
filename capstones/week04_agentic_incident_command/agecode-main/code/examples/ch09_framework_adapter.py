# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 9: tiny framework-style runner that executes planned steps.

Local, deterministic: maps steps to tools and enforces timeouts/retries.
"""

from __future__ import annotations  # Future-proof typing.

import time  # Measure latency.
from dataclasses import dataclass  # Lightweight records.
from typing import Any, Callable, Dict, List  # Type hints.


# tag::ch09_adapter[]
@dataclass
class Step:
    tool_name: str  # Target tool identifier.
    payload: Dict[str, Any]  # Inputs for the tool.


class Runner:
    def __init__(self, *, tools: Dict[str, Callable[..., Dict[str, Any]]]):
        self.tools = tools  # Map tool_name → function.

    def run(
        self,
        steps: List[Step],
        *,
        timeout_ms: int = 200,
        max_retries: int = 1,
    ) -> List[Dict[str, Any]]:
        logs: List[Dict[str, Any]] = []  # Accumulate compact log lines.
        for i, s in enumerate(steps, 1):  # Iterate steps.
            start = time.perf_counter()  # Start timer.
            tries = 0  # Attempt counter.
            while True:  # Retry loop.
                tries += 1  # Increment try.
                try:
                    # Call tool; for brevity no thread timeout — keep demo local.
                    out = self.tools[s.tool_name](**s.payload)  # Execute tool.
                    latency_ms = int((time.perf_counter() - start) * 1000)  # Time.
                    logs.append(
                        {
                            "i": i,
                            "tool": s.tool_name,
                            "status": "ok",
                            "latency_ms": latency_ms,
                        }
                    )
                    break  # Success: next step.
                except Exception as exc:  # noqa: BLE001 — small demo.
                    if tries > max_retries:  # Give up.
                        latency_ms = int((time.perf_counter() - start) * 1000)
                        logs.append(
                            {
                                "i": i,
                                "tool": s.tool_name,
                                "status": f"error:{exc}",
                                "latency_ms": latency_ms,
                            }
                        )
                        break  # Stop retrying this step.
                    time.sleep(0.05 * tries)  # Small backoff.
        return logs  # Return compact logs.


# Demo tools (pure functions for portability)
def add(a: int, b: int) -> Dict[str, Any]:
    return {"sum": a + b}  # Return a tiny dict.


def boom() -> Dict[str, Any]:
    raise RuntimeError("planned failure")  # Force an error.


def demo() -> None:
    steps = [
        Step("add", {"a": 2, "b": 3}),
        Step("boom", {}),
        Step("add", {"a": 1, "b": 1}),
    ]
    r = Runner(tools={"add": add, "boom": lambda: boom()})  # Register tools.
    logs = r.run(steps, timeout_ms=150, max_retries=1)  # Execute plan.
    for row in logs:  # Print compact logs.
        print(row)


if __name__ == "__main__":
    demo()  # Execute demo when run directly.
# end::ch09_adapter[]
