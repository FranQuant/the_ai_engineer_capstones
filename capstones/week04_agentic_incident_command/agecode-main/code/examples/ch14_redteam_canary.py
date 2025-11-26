# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 14: tiny red-team and canary harness (local).

Runs a few probes and checks expected outcomes; non-zero exit on failure.
"""

from __future__ import annotations  # Future-proof typing.

import json  # Print compact results.
import sys  # Exit codes.
import time  # Measure latency.
from typing import Any, Callable, Dict, List  # Type hints.


# tag::ch14_redteam[]
Probe = Dict[str, Any]


def run_probe(name: str, fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    start = time.perf_counter()  # Timer start.
    out = fn()  # Execute.
    ms = int((time.perf_counter() - start) * 1000)  # Latency.
    return {"name": name, "status": out.get("status", "ok"), "latency_ms": ms}


def main() -> None:
    # Probes return a dict with a status.
    def ok() -> Dict[str, Any]:
        return {"status": "ok"}

    def fail() -> Dict[str, Any]:
        return {"status": "error"}

    probes: List[Probe] = [
        run_probe("ok-probe", ok),  # Expected ok.
        run_probe("bad-probe", fail),  # Expected error.
    ]
    for p in probes:
        print(json.dumps(p))  # Emit JSON line per probe.
    # Canary policy: at least one ok and one error expected.
    ok_count = sum(1 for p in probes if p["status"] == "ok")
    err_count = sum(1 for p in probes if p["status"] != "ok")
    if ok_count < 1 or err_count < 1:
        sys.exit(1)  # Non-zero signals failure.


if __name__ == "__main__":
    main()  # Run harness.
# end::ch14_redteam[]

