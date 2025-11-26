# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Optional snapshot: CrewAI planner/critic for citations (Chapter 14).

Requires: crewai (optional) and an LLM provider.
"""

from __future__ import annotations  # Future annotations.

try:
    from crewai import Agent, Task, Crew  # CrewAI types.
except Exception:  # noqa: BLE001
    Agent = Task = Crew = None  # type: ignore  # Fallback if missing.


def demo() -> None:  # Planner/critic enforcing citations.
    if Agent is None:  # Missing package?
        print("crewai not installed; skip")  # Inform.
        return  # Exit.

    planner = Agent(  # Planning agent.
        role="Planner",
        goal=(
            "Draft 3 short bullets with [source:id] citations "
            "from provided sources"
        ),
        backstory=("Writes terse, factual bullets with required citations."),
        allow_delegation=False,
        verbose=False,
    )
    critic = Agent(  # Critic agent.
        role="Critic",
        goal=(
            "Check bullets include at least one [source:id] each; "
            "request revision otherwise"
        ),
        backstory=("Enforces citation policy and brevity."),
        allow_delegation=False,
        verbose=False,
    )

    topic = (
        "Topic: Alpha. Sources: s1='Alpha launched in 2022', "
        "s2='Key benefit: transparency'.\n"
        "Output 3 bullets. Each MUST include [source:s1] or [source:s2]."
    )  # Task prompt with sources and constraint.
    t1 = Task(description=topic, agent=planner)  # Draft task.
    t2 = Task(
        description="Review for citations; request revision if missing.",
        agent=critic,
    )  # Review.
    crew = Crew(agents=[planner, critic], tasks=[t1, t2])  # Group.
    print(crew.kickoff())  # Run.


if __name__ == "__main__":
    demo()
