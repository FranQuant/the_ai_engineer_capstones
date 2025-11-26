# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 11: toy research → signal → report pipeline (local).

Deterministic rules with compact audit output.
"""

from __future__ import annotations  # Future-proof typing.

import json  # Emit a compact audit line.
from dataclasses import dataclass  # Lightweight records.
from typing import Dict  # Type hints.


# tag::ch11_finance[]
POSITIVE = {"beats", "surge", "record"}  # Toy positive cues.
NEGATIVE = {"misses", "downgrade", "cut"}  # Toy negative cues.


@dataclass
class Note:
    ticker: str  # Symbol under discussion.
    text: str  # Short research headline or note.


def extract(note: Note) -> Dict[str, str]:
    text = note.text.lower()  # Lowercase for matching.
    if any(w in text for w in POSITIVE):  # Positive cue?
        return {"cue": "positive", "reason": "positive phrasing"}
    if any(w in text for w in NEGATIVE):  # Negative cue?
        return {"cue": "negative", "reason": "negative phrasing"}
    return {"cue": "neutral", "reason": "no strong cue"}  # Neutral.


def decide(ticker: str, cue: str) -> Dict[str, str]:
    allow = {"ACME", "FOO", "BAR"}  # Tiny allowlist.
    if ticker not in allow:  # Guard unknown tickers.
        return {"status": "error", "signal": "n/a", "reason": "unknown ticker"}
    if cue == "positive":  # Map cue to signal.
        return {"status": "ok", "signal": "up"}
    if cue == "negative":
        return {"status": "ok", "signal": "down"}
    return {"status": "ok", "signal": "flat"}  # Neutral.


def report(note: Note, decision: Dict[str, str], extracted: Dict[str, str]) -> str:
    return (
        f"{note.ticker}: signal={decision['signal']} status={decision['status']} "
        f"because {extracted['reason']}"
    )  # One-line report.


def main() -> None:
    # Input note.
    note = Note(ticker="ACME", text="ACME beats estimates on record demand")
    extracted = extract(note)  # Extract cue + reason.
    decision = decide(note.ticker, extracted["cue"])  # Decide signal.
    line = report(note, decision, extracted)  # Build report line.
    print(line)  # Show report.
    # Emit a compact audit JSON line.
    print(json.dumps({
        "ticker": note.ticker,
        "signal": decision.get("signal", "n/a"),
        "reason": extracted["reason"],
    }))


if __name__ == "__main__":
    main()  # Run demo when executed directly.
# end::ch11_finance[]
