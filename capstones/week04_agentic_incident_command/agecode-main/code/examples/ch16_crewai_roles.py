# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Optional snapshot: CrewAI two-role (planner/writer) for Chapter 16.

Requires: crewai (optional dependency) and an LLM provider configured.
"""

from __future__ import annotations  # Future annotations.

try:
    from crewai import Agent, Task, Crew  # Optional CrewAI primitives.
except Exception:  # noqa: BLE001
    Agent = Task = Crew = None  # type: ignore  # Fallback when package missing.


def demo() -> None:
    if Agent is None:  # Guard optional dependency.
        print("crewai not installed; skip")  # Inform caller.
        return  # Exit early.

    planner = Agent(  # Role 1: create the plan.
        role="Planner",
        goal="Propose a short, safe fix plan for ops tickets",
        backstory=(
            "Triages issues and proposes 2–3 concrete, low-risk steps."
        ),
        allow_delegation=False,
        verbose=False,
    )
    writer = Agent(  # Role 2: summarize plan.
        role="Writer",
        goal="Summarize the approved plan in one paragraph",
        backstory=(
            "Turns steps into a terse summary for operators to execute."
        ),
        allow_delegation=False,
        verbose=False,
    )

    t1 = Task(  # Planner task with ticket prompt.
        description=(
            "Ticket: 'DB timeout on node2'. Propose 2–3 steps and "
            "include 'report result'."
        ),
        agent=planner,
    )
    t2 = Task(  # Writer task for final summary.
        description=(
            "Take the steps and write a one-paragraph summary (<=120 chars)."
        ),
        agent=writer,
    )

    crew = Crew(agents=[planner, writer], tasks=[t1, t2])  # Assemble crew + plan.
    res = crew.kickoff()  # Run the mini workflow.
    print(res)  # Display combined result.


if __name__ == "__main__":
    demo()  # Execute snapshot when run directly.
