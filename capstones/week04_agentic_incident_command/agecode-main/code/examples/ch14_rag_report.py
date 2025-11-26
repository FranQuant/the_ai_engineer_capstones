# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 14 (case study): plan → draft → fact-check → report (local).

Uses LocalRetriever (dependency-free). Swap in a framework later.
"""

from __future__ import annotations  # Future annotations.

from dataclasses import dataclass  # Available if extending types.
from typing import Dict, List, Tuple  # Type hints for clarity.

try:
    from .ch14_rag_local import Doc, LocalRetriever  # Local retriever + Doc.
except ImportError:
    from ch14_rag_local import Doc, LocalRetriever  # Local retriever + Doc.


# tag::ch14_rag_report[]
def plan(topic: str) -> Dict[str, str]:  # Constraints for drafting.
    return {
        "topic": topic,  # Subject.
        "bullets": "5",  # Bullet count.
        "style": "short, factual",  # Tone.
        "cite": "[source:id]",  # Citation format.
        "summary_len": "<=120 chars",  # Summary budget.
    }


def draft_bullets(
    topic: str,
    hits: List[Tuple[Doc, float]],
) -> List[str]:
    """Create cite-ready bullets."""
    bullets: List[str] = []  # Accumulator.
    for d, _ in hits:  # For each hit…
        clause = d.text.split(".")[0]  # Take first sentence.
        bullets.append(f"- {clause} [source:{d.id}]")  # Add citation.
    if len(bullets) < 5:  # Ensure 5 bullets.
        bullets.append(
            f"- Summary point on {topic} [source:{hits[0][0].id}]"
        )  # Filler.
    return bullets[:5]  # Trim to 5.


def fact_check(bullets: List[str], docs: Dict[str, Doc]) -> List[str]:
    """Verify citations."""
    checked: List[str] = []  # Results list.
    for b in bullets:  # For each bullet…
        import re  # Regex for id extraction.
        m = re.findall(r"\[source:([A-Za-z0-9_]+)\]", b)  # Find cited ids.
        ok = any(
            (cid in docs and docs[cid].text.split(".")[0] in b)
            for cid in m
        )  # Supported?
        checked.append(b if ok else (b + " [needs_review]"))  # Flag unsupported.
    return checked  # Return annotated bullets.


def summary_from_bullets(bullets: List[str]) -> str:  # Short summary.
    s = " ".join(x.lstrip("- ") for x in bullets[:2])  # Join first two.
    return s[:117] + "…" if len(s) > 120 else s  # Clip to 120 chars.


def report(topic: str, bullets: List[str], summary: str) -> str:
    """Assemble the final markdown string."""
    out = [
        f"# Brief: {topic}",
        "",
        "Key Points:",
        *bullets,
        "",
        f"Summary: {summary}",
    ]
    return "\n".join(out)  # Single string.


def demo() -> None:  # Run the local pipeline.
    docs = [  # Three short docs.
        Doc("s1", "Alpha launched in 2022 with a focus on simplicity."),  # Doc 1.
        Doc("s2", "Key benefit: transparency in logs and short audits."),  # Doc 2.
        Doc("s3", "Beta emphasized speed over explainability in 2021."),  # Doc 3.
    ]
    r = LocalRetriever(docs)  # Build retriever.
    topic = "Alpha vs Beta"  # Topic string.
    hits = r.topk(topic, k=3)  # Retrieve top‑3.
    plan_spec = plan(topic)  # Constraints (unused in stub).
    bullets = draft_bullets(topic, hits)  # Draft bullets.
    by_id = {d.id: d for d in docs}  # Map ids → docs.
    checked = fact_check(bullets, by_id)  # Verify citations.
    brief = report(topic, checked, summary_from_bullets(checked))  # Compose.
    print(brief)  # Print final brief.


if __name__ == "__main__":  # Entry point.
    demo()  # Execute demo.
# end::ch14_rag_report[]
