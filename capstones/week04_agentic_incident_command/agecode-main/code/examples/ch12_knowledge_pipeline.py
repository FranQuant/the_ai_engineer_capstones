# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 12: tiny parsing + summarization pipeline (local).

Extract dates/amounts and produce a one-line summary.
"""

from __future__ import annotations  # Future-proof typing.

import json  # Emit audit lines.
import re  # Simple regex extractors.
from typing import Dict, Tuple  # Type hints.


# tag::ch12_knowledge[]
GLOSSARY = {"KPI": "key performance indicator"}  # Tiny glossary.


def extract_fields(text: str) -> Dict[str, str]:
    date = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)  # ISO date.
    amt = re.search(r"\b(\d+(?:\.\d+)?)(?:\s*(USD|EUR))?\b", text)  # Amount.
    return {
        "date": date.group(1) if date else "n/a",
        "amount": amt.group(1) if amt else "n/a",
        "currency": (amt.group(2) if amt and amt.group(2) else "n/a"),
    }


def expand_once(summary: str, term: str, expansion: str) -> str:
    if term in summary:  # If acronym appears,
        return summary.replace(term, f"{expansion} ({term})", 1)  # expand first.
    return summary  # Otherwise unchanged.


def build_summary(doc_id: str, fields: Dict[str, str]) -> str:
    parts = [f"doc={doc_id}"]  # Start summary.
    parts.append(f"date={fields['date']}")  # Add date.
    if fields["amount"] != "n/a":  # Include amount when present.
        parts.append(f"amount={fields['amount']} {fields['currency']}")
    else:
        parts.append("no amounts found")  # Explicit note.
    summary = ", ".join(parts)  # Join into one line.
    for term, exp in GLOSSARY.items():  # Expand glossary once.
        summary = expand_once(summary, term, exp)  # Expand first use.
    # Max length guard (120 chars) with ellipsis.
    return (summary[:117] + "…") if len(summary) > 120 else summary


def main() -> None:
    doc = "On 2024-09-01 revenue rose to 3.5 USD; KPI improved."  # Input text.
    fields = extract_fields(doc)  # Parse fields.
    print(json.dumps({"doc_id": "ex1", "extracted": fields}))  # Audit.
    summary = build_summary("ex1", fields)  # Build one-liner.
    print(summary)  # Show summary.


if __name__ == "__main__":
    main()  # Execute when run directly.
# end::ch12_knowledge[]

