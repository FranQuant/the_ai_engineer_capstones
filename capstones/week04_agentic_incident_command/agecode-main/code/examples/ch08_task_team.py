# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 8: small specialist team — planner, operator, critic.

Local, deterministic example with compact audit logging.
"""

from __future__ import annotations  # Future-proof typing.

from dataclasses import dataclass  # Lightweight containers.
from typing import Any, Dict, List  # Type hints.


# tag::ch08_team[]
@dataclass
class Note:
    role: str  # Who acted.
    action: str  # What they did.
    status: str  # ok|error


def make_plan(goal: str) -> Dict[str, Any]:
    steps = [
        {"tool": "parse_numbers", "args": {}},  # Step 1.
        {"tool": "divide", "args": {}},  # Step 2.
        {"tool": "report", "args": {}},  # Step 3.
    ]  # Minimal structured plan.
    return {"goal": goal, "steps": steps}  # Return a dict plan.


def operate(step: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    tool = step["tool"]  # Select tool by name.
    if tool == "parse_numbers":  # Parse a tiny CSV from state.
        raw = state.get("raw", "9,3")  # Read a CSV string.
        a, b = [int(x) for x in raw.split(",")]  # Convert to ints.
        return {"a": a, "b": b}  # Emit fields.
    if tool == "divide":  # Divide a by b.
        a = state.get("a", 1)  # Read a.
        b = state.get("b", 1)  # Read b.
        if b == 0:  # Guard zero.
            raise ValueError("b must be non-zero")  # Clear error.
        return {"result": a / b}  # Emit result.
    if tool == "report":  # Produce a tiny summary string.
        return {"text": f"result={state.get('result', 'n/a')}"}  # Report.
    raise KeyError(f"unknown tool: {tool}")  # Unknown tool.


def criticize(plan: Dict[str, Any], notes: List[Note]) -> str:
    steps = plan.get("steps", [])  # Read steps.
    tools = [s.get("tool", "") for s in steps]  # Extract names.
    if "report" not in tools:  # Require a report.
        return "fallback"  # Trigger fallback path.
    return "approve"  # Otherwise approve.


def run_team(goal: str = "divide 9 over 3") -> None:
    state: Dict[str, Any] = {"raw": "9,3"}  # Initial input.
    notes: List[Note] = []  # Audit notes.
    plan = make_plan(goal)  # Planner drafts steps.
    for step in plan["steps"]:  # Iterate steps.
        try:
            out = operate(step, state)  # Run operator.
            state.update(out)  # Merge into state.
            notes.append(Note("operator", step["tool"], "ok"))  # Log.
        except Exception as exc:  # noqa: BLE001 — small demo.
            notes.append(Note("operator", step["tool"], f"error:{exc}"))  # Log.
            break  # Stop on error.
    verdict = criticize(plan, notes)  # Critic reviews.
    if verdict == "fallback":  # Simple fallback path.
        plan = {"goal": goal, "steps": plan["steps"][:2]}  # Shorten plan.
    # Print a compact audit table.
    for i, n in enumerate(notes, 1):  # Enumerate notes.
        print({"turn": i, "role": n.role, "action": n.action, "status": n.status})


if __name__ == "__main__":
    run_team()  # Execute demo when run directly.
# end::ch08_team[]

