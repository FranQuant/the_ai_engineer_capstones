# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Optional FastAPI wrapper for Chapter 16 triage loop.

Expose POST /triage with json {text, max_turns} → {status, latency_ms}.
"""

from __future__ import annotations  # Future annotations.

from fastapi import FastAPI  # Web framework.
from pydantic import BaseModel  # Request model.

try:
    from .ch16_ops_triage import Ticket, run  # Local loop.
except ImportError:
    from ch16_ops_triage import Ticket, run  # Local loop.


class TriageRequest(BaseModel):  # Request schema.
    text: str  # Ticket text.
    max_turns: int = 3  # Turn budget.


app = FastAPI()  # App instance.


@app.post("/triage")  # HTTP endpoint.
def triage(req: TriageRequest) -> dict:  # Handler.
    return run(Ticket(req.text), max_turns=req.max_turns)  # Delegate to loop.


# To run locally:
#   uvicorn code.examples.ch16_triage_service:app --reload
