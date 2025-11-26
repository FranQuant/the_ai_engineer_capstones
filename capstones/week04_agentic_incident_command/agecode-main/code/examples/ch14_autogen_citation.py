# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Optional snapshot: AutoGen planner/critic for citations (Chapter 14).

Requires: pyautogen (optional) and a configured LLM provider. The critic
requests revision if bullets lack [source:id] citations.
"""

from __future__ import annotations  # Future annotations.

try:
    from autogen import AssistantAgent, UserProxyAgent  # Agents.
except Exception:  # noqa: BLE001
    AssistantAgent = UserProxyAgent = None  # type: ignore  # Fallback if missing.


def demo() -> None:  # Planner/critic enforcing citations.
    if AssistantAgent is None:  # Missing package?
        print("pyautogen not installed; skip")  # Inform.
        return  # Exit.

    planner = AssistantAgent(  # Drafting agent.
        name="planner",
        system_message=(
            "You draft 3 short bullets with [source:id] "
            "citations from given sources."
        ),
    )
    critic = UserProxyAgent(  # Critic proxy.
        name="critic",
        human_input_mode="NEVER",
        code_execution_config=False,
    )

    sources = {  # Example sources.
        "s1": "Alpha launched in 2022 with a focus on simplicity.",
        "s2": "Key benefit: transparency in logs and short audits.",
    }
    topic = (
        "Topic: Alpha. Sources: s1='Alpha launched in 2022...', "
        "s2='Key benefit...'\nWrite 3 bullets. Each bullet MUST include "
        "one [source:id]."
    )  # Prompt with sources and constraint.

    chat = planner.initiate_chat(critic, message=topic)  # Start exchange.
    print(chat.summary)  # Show outcome.


if __name__ == "__main__":
    demo()
