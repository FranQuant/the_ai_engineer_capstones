# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 13: tiny queue runner with budgets and metrics (local).

Deterministic processing with simple timing and counts.
"""

from __future__ import annotations  # Future-proof typing.

import random  # Simulate occasional slow items.
import time  # Time measurements.
from typing import Dict, List  # Type hints.


# tag::ch13_industry[]
def process(item: int) -> str:
    # Simulate variable work; occasionally sleep longer to test budgets.
    if item % 7 == 0:
        time.sleep(0.06)  # Slow path for testing.
    else:
        time.sleep(0.01)  # Normal path.
    if random.random() < 0.05:  # Rare error.
        raise RuntimeError("transient error")  # Fail sometimes.
    return "ok"  # Success marker.


def run(n: int = 10, budget_ms: int = 40) -> Dict[str, float]:
    ok = errors = violations = 0  # Counters.
    latencies: List[int] = []  # Per-item timings.
    for i in range(1, n + 1):  # Process items 1..n.
        start = time.perf_counter()  # Start timer.
        status = "ok"  # Default status.
        try:
            status = process(i)  # Run work.
            ok += 1  # Count success (adjusted below if violation).
        except Exception:  # noqa: BLE001 — small demo.
            errors += 1  # Count errors.
            status = "error"  # Mark status.
        ms = int((time.perf_counter() - start) * 1000)  # Duration.
        latencies.append(ms)  # Record.
        if ms > budget_ms:  # Budget violation?
            violations += 1  # Count violation.
        print({"item": i, "status": status, "latency_ms": ms})  # Log line.
    avg_ms = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
    print(f"summary_csv={n},{ok},{errors},{violations},{avg_ms}")  # CSV line.
    return {
        "items": n,
        "ok": ok,
        "errors": errors,
        "violations": violations,
        "avg_ms": avg_ms,
    }


if __name__ == "__main__":
    run()  # Execute demo when run directly.
# end::ch13_industry[]
