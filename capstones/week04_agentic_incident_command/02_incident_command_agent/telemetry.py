"""
Structured telemetry utilities for the Incident Command Agent.

Responsibilities:
- Generate correlation IDs and loop IDs.
- Track budgets (tokens, milliseconds, dollars).
- Emit telemetry events for observe/plan/act/learn phases.
- Provide JSONL logger compatible with warm-up harness patterns.

TODO:
- Implement logging sinks (JSONL, stdout).
- Add timing helpers and budget enforcement hooks.
- Define telemetry schemas for replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def new_correlation_id() -> str:
    """Return a unique correlation ID for sessions."""
    raise NotImplementedError


@dataclass
class Budget:
    tokens: int
    ms: int
    dollars: float


@dataclass
class RunContext:
    correlation_id: str
    loop_id: str


@dataclass
class TelemetryEvent:
    correlation_id: str
    loop_id: str
    phase: str
    method: str
    status: str
    latency_ms: int
    budget: Budget
    payload: Dict[str, Any]


class TelemetryLogger:
    def __init__(self, sink: Path) -> None:
        """Initialize telemetry logger with JSONL sink."""
        self.sink = sink

    def log(self, event: TelemetryEvent) -> None:
        """Record telemetry event."""
        raise NotImplementedError


def timed(fn, *args, **kwargs) -> Tuple[int, Any]:
    """Measure latency and return (latency_ms, result)."""
    raise NotImplementedError
