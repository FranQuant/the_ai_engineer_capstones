# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Optional snapshot: Prefect flow for Chapter 15 signals pipeline.

Requires: prefect (optional dependency).
"""

from __future__ import annotations  # Future annotations.

from prefect import flow, task  # Prefect primitives.


@task  # Extract cue from note.
def extract(note: str) -> str:
    t = note.lower()  # Normalize.
    if any(w in t for w in ("beats", "surge", "record")):  # Positive.
        return "positive"  # Cue.
    if any(w in t for w in ("misses", "downgrade", "cut")):  # Negative.
        return "negative"  # Cue.
    return "neutral"  # Default.


@task  # Decide signal and size.
def decide(ticker: str, cue: str, cap: float = 0.5) -> tuple[str, float, str]:
    allow = {"ACME", "FOO", "BAR"}  # Allowlist.
    if ticker not in allow:  # Unknown?
        return ("n/a", 0.0, "unknown")  # Refuse.
    if cue == "positive":
        sig = "up"
    elif cue == "negative":
        sig = "down"
    else:
        sig = "flat"  # Signal.
    base = 0.5 if sig in {"up", "down"} else 0.0  # Nominal size.
    size = min(base, cap)  # Clamp.
    return (sig, size, "ok" if size == base else "clamped")  # Result.


@flow  # Orchestrate tasks.
def daily_signals() -> None:
    notes = [  # Sample items.
        ("ACME", "ACME beats estimates"),  # Positive.
        ("FOO", "FOO misses guidance"),  # Negative.
    ]
    for t, txt in notes:  # Tasks per item.
        c = extract.submit(txt)  # Async extract.
        sig, size, reason = decide.submit(t, c).result()  # Decide.
        print(
            {"ticker": t, "signal": sig, "size": size, "reason": reason}
        )  # Log.


if __name__ == "__main__":  # Entry.
    daily_signals()  # Run flow.
