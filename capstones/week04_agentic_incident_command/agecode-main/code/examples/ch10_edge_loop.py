# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 10: tiny edge loop with sandboxed state and budgets.

Local, deterministic loop; demonstrates ephemeral sandbox and audit log.
"""

from __future__ import annotations  # Future-proof typing.

import json  # Write a compact audit file.
import os  # Join paths.
import tempfile  # Ephemeral sandbox directory.
import time  # CPU guard.
from typing import Any, Dict, List  # Type hints.


# tag::ch10_edge[]
def run_edge(
    goal: str = "add 2 and 3",
    max_turns: int = 3,
    cpu_ms: int = 50,
) -> Dict[str, Any]:
    notes: List[Dict[str, Any]] = []  # Collect audit entries.
    cpu_start = time.process_time()  # Start CPU timer.
    with tempfile.TemporaryDirectory() as tmp:  # Sandboxed folder.
        audit = os.path.join(tmp, "audit.jsonl")  # Audit path.
        state: Dict[str, Any] = {"goal": goal}  # Minimal state.
        for turn in range(1, max_turns + 1):  # Budgeted loop.
            # CPU guard: stop if over CPU budget.
            if int((time.process_time() - cpu_start) * 1000) > cpu_ms:
                notes.append(
                    {"turn": turn, "action": "stop", "status": "cpu_budget"}
                )
                break  # Stop loop.
            # Single action: parse + add locally.
            a, b = 2, 3  # Toy numbers for demo.
            result = a + b  # Compute.
            notes.append(
                {"turn": turn, "action": "add", "status": "ok", "result": result}
            )
        # Persist compact audit file.
        with open(audit, "w", encoding="utf-8") as f:
            for n in notes:
                f.write(json.dumps(n) + "\n")  # JSON lines.
        return {"notes": notes, "audit_path": audit}  # Return outcome.


if __name__ == "__main__":
    print(run_edge())  # Quick demo.
# end::ch10_edge[]
