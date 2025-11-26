# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 10: cloud-style handler stub that enforces budgets/timeouts.

All local; demonstrates a single `handle(event)` entrypoint.
"""

from __future__ import annotations  # Future-proof typing.

import time  # Measure latency and CPU time.
from typing import Any, Dict  # Type hints.


# tag::ch10_deploy[]
def handle(event: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()  # Wall-clock start.
    cpu_start = time.process_time()  # CPU time start.
    max_ms = int(event.get("max_ms", 200))  # Budget in milliseconds.
    # Step 1: basic allowlist for tools.
    allow = {"add"}  # Single allowed tool name for the demo.
    tool = event.get("tool", "")  # Tool name from caller.
    if tool not in allow:  # Guard unknown tools.
        return {"status": "error", "message": "tool not allowed"}
    # Step 2: execute the tiny action under a soft timeout budget.
    try:
        a, b = int(event.get("a", 0)), int(event.get("b", 0))  # Parse ints.
        # Simulate tiny work; in real code call your loop/tool adapter here.
        result = a + b  # Compute sum.
    except Exception as exc:  # noqa: BLE001 — small demo.
        return {"status": "error", "message": str(exc)}
    # Step 3: finalize with latency and CPU time checks.
    latency_ms = int((time.perf_counter() - start) * 1000)  # Wall-clock.
    cpu_ms = int((time.process_time() - cpu_start) * 1000)  # CPU time.
    if latency_ms > max_ms:  # Enforce budget.
        return {"status": "error", "message": f"over budget {latency_ms} ms"}
    return {
        "status": "ok",
        "latency_ms": latency_ms,
        "cpu_ms": cpu_ms,
        "result": result,
    }


if __name__ == "__main__":
    print(handle({"tool": "add", "a": 2, "b": 3, "max_ms": 50}))  # Quick demo.
# end::ch10_deploy[]
