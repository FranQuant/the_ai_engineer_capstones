# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Optional snapshot: AutoGen two-agent (planner/critic) for Chapter 16.

Requires: pyautogen (optional dependency) and a configured LLM provider.
"""

from __future__ import annotations  # Future annotations.

from autogen import AssistantAgent, UserProxyAgent  # AutoGen agents.


def demo() -> None:  # Minimal planner/critic.
    system = (
        "You are a helpful planner for ops triage. Keep steps short."
    )  # System prompt.
    planner = AssistantAgent(name="planner", system_message=system)  # Planner.
    critic = UserProxyAgent(  # Critic proxy.
        name="critic",
        human_input_mode="NEVER",
        code_execution_config=False,
    )
    topic = "Triage: DB timeout on node2"  # Task.
    chat = planner.initiate_chat(critic, message=topic)  # Start chat.
    print(chat.summary)  # Print recap.


if __name__ == "__main__":
    demo()
