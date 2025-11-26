# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 15 (case study): signals pipeline with simple compliance.

Deterministic cues → signals; allowlist; position cap; compact audits.
"""

from __future__ import annotations  # Future annotations.

import json  # JSON for audits.
import time  # Timestamps.
from dataclasses import dataclass  # Note container.
from typing import Dict  # Type hints.


# tag::ch15_signals[]
POS = {"beats", "surge", "record"}  # Positive cues.
NEG = {"misses", "downgrade", "cut"}  # Negative cues.


@dataclass  # Ingested note.
class Note:
    ticker: str  # Symbol.
    text: str  # Text snippet.


def extract_cue(text: str) -> str:  # Map text → cue.
    t = text.lower()  # Normalize.
    if any(w in t for w in POS):  # Positive hit?
        return "positive"  # Return cue.
    if any(w in t for w in NEG):  # Negative hit?
        return "negative"  # Return cue.
    return "neutral"  # Default.


def decide_with_limits(
    ticker: str,
    cue: str,
    *,
    cap: float = 0.5,
) -> Dict[str, str]:
    """Apply allowlist and sizing caps."""
    allow = {"ACME", "FOO", "BAR"}  # Allowlist.
    if ticker not in allow:  # Unknown?
        return {
            "status": "error",
            "signal": "n/a",
            "size": "0",
            "reason": "unknown",
        }  # Refuse.
    if cue == "positive":
        sig = "up"
    elif cue == "negative":
        sig = "down"
    else:
        sig = "flat"  # Signal.
    base = 0.5 if sig in {"up", "down"} else 0.0  # Nominal size.
    size = min(base, cap)  # Clamp to cap.
    reason = "ok" if size == base else "clamped"  # Rationale.
    return {
        "status": "ok",
        "signal": sig,
        "size": f"{size:.2f}",
        "reason": reason,
    }  # Decision.


def audit_line(note: Note, dec: Dict[str, str]) -> str:  # JSONL audit.
    return json.dumps({  # Serialize compact line.
        "ts": int(time.time()),  # Unix ts.
        "ticker": note.ticker,  # Symbol.
        "signal": dec.get("signal", "n/a"),  # Signal.
        "size": dec.get("size", "0"),  # Position size.
        "reason": dec.get("reason", ""),  # Rationale.
    })


def demo() -> None:  # Run pipeline on three notes.
    notes = [  # Sample notes.
        Note("ACME", "ACME beats estimates on new demand"),  # Positive.
        Note("FOO", "FOO misses guidance amid supply issues"),  # Negative.
        Note("XYZ", "Unknown ticker example"),  # Unknown.
    ]
    for n in notes:  # Process each.
        cue = extract_cue(n.text)  # Compute cue.
        dec = decide_with_limits(n.ticker, cue, cap=0.4)  # Decide with cap.
        csv_row = [n.ticker, dec["signal"], dec["size"], dec["reason"]]
        print("CSV:", ",".join(csv_row))  # CSV line.
        print("AUD:", audit_line(n, dec))  # JSONL audit.


if __name__ == "__main__":  # Script entry.
    demo()  # Execute demo.
# end::ch15_signals[]
