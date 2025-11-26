# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 16 (case study): tiny triage with planner/critic/operator.

Dependency-free; adds a small budget and a safe operator.
"""

from __future__ import annotations  # Future annotations.

import time  # Timing and sleep.
from dataclasses import dataclass  # Ticket container.
from typing import Dict, List  # Type hints.


# tag::ch16_triage[]
@dataclass  # Minimal ticket.
class Ticket:
    text: str  # Description.


def categorize(text: str) -> str:  # Heuristic categorization.
    t = text.lower()  # Normalize.
    if "db" in t or "database" in t:  # Database hint.
        return "db"  # Database.
    if "network" in t:  # Network hint.
        return "network"  # Network.
    return "app"  # Default app.


def planner(t: Ticket) -> Dict[str, List[str]]:  # Propose small fix plan.
    cat = categorize(t.text)  # Category.
    if cat == "db":  # Database steps.
        return {
            "category": [cat],
            "steps": ["check db status", "restart db client"],
        }
    if cat == "network":  # Network steps.
        return {
            "category": [cat],
            "steps": ["check router load", "reduce sample rate"],
        }
    return {
        "category": [cat],
        "steps": ["inspect logs", "restart service"],
    }  # App steps.


def critic(plan: Dict[str, List[str]]) -> Dict[str, List[str]]:  # Enforce checklist.
    steps = plan["steps"]  # Step list.
    if "report" not in " ".join(steps):  # Missing report?
        steps.append("report result")  # Append.
    return plan  # Return amended plan.


def operator(step: str) -> str:  # Simulate execution.
    time.sleep(0.01)  # Sleep to simulate work.
    return "ok"  # Success.


def run(ticket: Ticket, max_turns: int = 3) -> Dict[str, str]:  # Run triage loop.
    start = time.perf_counter()  # Start timer.
    plan = planner(ticket)  # Make plan.
    plan = critic(plan)  # Check plan.
    turns = 0  # Turn counter.
    ok = True  # Success flag.
    for s in plan["steps"]:  # Iterate steps.
        turns += 1  # Increase turn.
        out = operator(s)  # Execute.
        print({"turn": turns, "step": s, "status": out})  # Log.
        if turns >= max_turns:  # Over budget?
            ok = False  # Mark stopped.
            print({"status": "budget exhausted"})  # Log budget.
            break  # Exit loop.
    ms = int((time.perf_counter() - start) * 1000)  # Total ms.
    return {
        "status": "ok" if ok else "stopped",
        "latency_ms": str(ms),
    }  # Result.


def demo() -> None:  # Try two tickets.
    for txt in ["DB timeout on node2", "Network spike at 9pm"]:  # Samples.
        out = run(Ticket(txt), max_turns=3)  # Run with budget.
        print(out)  # Print result.


if __name__ == "__main__":  # Entry point.
    demo()  # Execute demo.
# end::ch16_triage[]
