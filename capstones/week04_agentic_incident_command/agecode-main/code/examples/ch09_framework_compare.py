# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 9: compare plain plan-execute vs. framework-style runner.

Local, deterministic example; prints two small tables.
"""

from __future__ import annotations  # Future-proof typing.

from typing import Any, Dict, List  # Type hints.

try:
    from .ch09_framework_adapter import Runner, Step  # Reuse runner and step.
except ImportError:
    from ch09_framework_adapter import Runner, Step  # Reuse runner and step.


# tag::ch09_compare[]
def plain_execute(steps: List[Step]) -> List[Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []  # Collect status lines.
    for i, s in enumerate(steps, 1):  # Iterate steps.
        try:
            if s.tool_name == "add":  # Dispatch add.
                _ = s.payload["a"] + s.payload["b"]  # Compute; ignore value.
                logs.append({"i": i, "status": "ok"})  # Report ok.
            elif s.tool_name == "boom":  # Dispatch boom.
                raise RuntimeError("planned failure")  # Fail.
            else:
                logs.append({"i": i, "status": "skip"})  # Unknown tool.
        except Exception as exc:  # noqa: BLE001 — demo simplicity.
            logs.append({"i": i, "status": f"error:{exc}"})  # Record error.
    return logs  # Return compact table.


def main() -> None:
    steps = [
        Step("add", {"a": 2, "b": 3}),
        Step("boom", {}),
        Step("add", {"a": 1, "b": 1}),
    ]
    left = plain_execute(steps)  # Plain path.
    r = Runner(
        tools={
            "add": lambda a, b: {"sum": a + b},
            "boom": lambda: (_ for _ in ()).throw(
                RuntimeError("planned failure")
            ),
        }
    )
    right = r.run(steps, timeout_ms=150, max_retries=1)  # Runner path.
    print({"plain": left})  # Two-column style by labels.
    runner_rows = [{"i": row["i"], "status": row["status"]} for row in right]
    print({"runner": runner_rows})


if __name__ == "__main__":
    main()  # Execute demo when run directly.
# end::ch09_compare[]
