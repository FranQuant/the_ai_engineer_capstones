# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 14: minimal policy guard wrapper (local).

Checks allowlist, simple PII patterns, and budgets before calling an action.
"""

from __future__ import annotations  # Future-proof typing.

import re  # Simple pattern checks.
import time  # Budget timing.
from dataclasses import dataclass  # Lightweight containers.
from typing import Any, Callable, Dict  # Type hints.


# tag::ch14_guard[]
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


@dataclass
class Guard:
    allow_tools: set[str]  # Allowed tool names.
    max_ms: int = 200  # Time budget per request.

    def check(self, tool: str, payload: Dict[str, Any]) -> tuple[bool, str]:
        if tool not in self.allow_tools:
            return False, "tool not allowed"
        text = str(payload)  # Quick scan; customize per tool.
        if EMAIL_RE.search(text):  # Basic PII check.
            return False, "contains email; refuse or redact"
        return True, "ok"

    def call(
        self,
        tool: str,
        fn: Callable[..., Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        start = time.perf_counter()  # Start budget timer.
        ok, msg = self.check(tool, kwargs)  # Policy checks.
        if not ok:
            return {"status": "error", "message": msg}
        out = fn(**kwargs)  # Call the action.
        ms = int((time.perf_counter() - start) * 1000)  # Latency.
        if ms > self.max_ms:  # Enforce time budget.
            return {"status": "error", "message": f"over budget {ms} ms"}
        return {"status": "ok", "latency_ms": ms, "result": out}


def echo(**kwargs) -> Dict[str, Any]:  # Tiny demo action.
    return {"echo": kwargs}


def demo() -> None:
    g = Guard(allow_tools={"echo"}, max_ms=150)  # Configure guard.
    print(g.call("echo", echo, text="hello"))  # Allowed.
    print(g.call("write", echo, text="no"))  # Disallowed tool.
    print(g.call("echo", echo, text="a@b.com"))  # PII refused.


if __name__ == "__main__":
    demo()  # Run small demo.
# end::ch14_guard[]
