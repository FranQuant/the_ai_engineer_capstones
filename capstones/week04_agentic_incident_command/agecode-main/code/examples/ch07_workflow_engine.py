# AI Agents & Automation
# (c) Dr. Yves J. Hilpisch
# AI-Powered by GPT-5
"""Chapter 7: tiny workflow engine (tasks + dependencies + retries).

Implements a minimal DAG executor for local, deterministic demos.
"""

from __future__ import annotations  # Future-proof typing of annotations.

import time  # Measure latency and implement simple backoff.
from dataclasses import dataclass, field  # Lightweight containers.
from typing import Any, Callable, Dict, List, Optional, Set, Tuple  # Hints.


# tag::ch07_workflow[]
@dataclass  # Describe a unit of work with small knobs.
class Task:
    name: str  # Task id for logs.
    fn: Callable[..., Dict[str, Any]]  # Function that does the work.
    deps: List[str] = field(default_factory=list)  # Upstream task names.
    timeout_ms: int = 300  # Max time per attempt in ms.
    max_retries: int = 1  # Retries after the first attempt.
    skip_on_error: bool = False  # If True, downstream may run on error.


@dataclass  # Hold result fields surfaced by the engine.
class TaskResult:
    status: str  # "ok" | "error" | "skipped".
    latency_ms: int  # Elapsed time for the final attempt.
    message: str  # Short human-readable note.
    value: Optional[Dict[str, Any]] = None  # Optional successful payload.


def _call_with_timeout(
    fn: Callable[..., Dict[str, Any]], *, timeout_ms: int, **kwargs
) -> Tuple[bool, Dict[str, Any] | str]:
    """Run a function and enforce a soft timeout using time checks.

    For simplicity (and portability), we poll time rather than threads.
    """

    start = time.perf_counter()  # Start timer for the call.
    try:
        value = fn(**kwargs)  # Run the task function.
        elapsed_ms = int((time.perf_counter() - start) * 1000)  # Duration.
        if elapsed_ms > timeout_ms:  # Check elapsed after return.
            return False, f"timeout after {elapsed_ms} ms"  # Soft timeout.
        return True, value  # Success with returned value.
    except Exception as exc:  # noqa: BLE001 — small demo, catch all.
        return False, str(exc)  # Convert error to string message.


class Workflow:
    def __init__(self, tasks: List[Task]) -> None:
        self.tasks: Dict[str, Task] = {t.name: t for t in tasks}  # Name → Task.
        self.results: Dict[str, TaskResult] = {}  # Name → TaskResult.
        self._check_acyclic()  # Validate DAG upfront.

    def _check_acyclic(self) -> None:
        seen: Set[str] = set()  # Track nodes visited.
        stack: Set[str] = set()  # Track recursion stack to detect cycles.

        def dfs(n: str) -> None:  # Depth-first traversal.
            if n in stack:  # Found a back-edge → cycle.
                raise ValueError(f"cycle detected at {n}")
            if n in seen:  # Already validated.
                return
            stack.add(n)  # Enter node.
            for d in self.tasks[n].deps:  # Validate children.
                if d not in self.tasks:  # Unknown dependency.
                    raise KeyError(f"unknown dependency: {d}")
                dfs(d)  # Recurse.
            stack.remove(n)  # Leave node.
            seen.add(n)  # Mark visited.

        for name in self.tasks:  # Check all nodes.
            dfs(name)

    def _ready(self) -> List[str]:
        ready: List[str] = []  # Collect runnable tasks.
        for name, task in self.tasks.items():  # Iterate all tasks.
            if name in self.results:  # Already finished.
                continue
            if all(d in self.results for d in task.deps):  # All deps done.
                # If any dep failed and we do not skip-on-error, hold back.
                if not task.skip_on_error:
                    if any(self.results[d].status == "error" for d in task.deps):
                        continue  # Block due to upstream error.
                ready.append(name)  # Task is ready to run.
        return ready

    def run(
        self,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, TaskResult]:
        ctx = dict(context or {})  # Shared context dict across tasks.
        pending = set(self.tasks.keys())  # Names not yet completed.
        while pending:  # Loop until all tasks reach a terminal state.
            ran_any = False  # Track progress to avoid infinite loops.
            for name in list(pending):  # Iterate over a snapshot.
                if name not in self._ready():  # Not ready → skip for now.
                    continue
                task = self.tasks[name]  # Fetch task.
                attempts = 0  # Count tries including first attempt.
                last_error = ""  # Store last error message.
                while True:  # Retry loop.
                    attempts += 1  # Increment attempt counter.
                    start = time.perf_counter()  # Start attempt timer.
                    ok, out = _call_with_timeout(
                        task.fn, timeout_ms=task.timeout_ms, context=ctx
                    )  # Call task.
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    if ok:  # Success path.
                        self.results[name] = TaskResult(
                            status="ok",
                            latency_ms=latency_ms,
                            message="ok",
                            value=out,
                        )  # Record success.
                        # Merge any declared outputs into context.
                        if isinstance(out, dict):  # Only dicts are merged.
                            ctx.update({f"{name}.{k}": v for k, v in out.items()})
                        break  # Exit retry loop.
                    # Failure path.
                    last_error = str(out)  # Keep last error message.
                    if attempts > task.max_retries:  # Retries exhausted.
                        self.results[name] = TaskResult(
                            status="error", latency_ms=latency_ms, message=last_error
                        )  # Record failure.
                        break  # Exit retry loop.
                    time.sleep(0.05 * attempts)  # Linear backoff for demo.
                pending.remove(name)  # Mark task as completed (ok or error).
                ran_any = True  # Progress made.
            if not ran_any:  # Deadlock due to upstream errors.
                # Mark remaining tasks as skipped to terminate cleanly.
                for name in list(pending):  # Iterate remaining tasks.
                    self.results[name] = TaskResult(
                        status="skipped", latency_ms=0, message="blocked by upstream"
                    )  # Record skip.
                    pending.remove(name)  # Mark done.
        return self.results  # Return full result map.


# Demo task functions (local, deterministic)
def t_fetch(*, context: Dict[str, Any]) -> Dict[str, Any]:
    time.sleep(0.05)  # Simulate IO latency.
    n = context.get("n", 2)  # Read a small input.
    return {"raw": f"{n}+3"}  # Return a tiny payload.


def t_parse(*, context: Dict[str, Any]) -> Dict[str, Any]:
    expr = context.get("fetch.raw", "0+0")  # Read upstream output.
    a, b = expr.split("+")  # Split a minimal expression.
    return {"a": int(a), "b": int(b)}  # Emit structured fields.


def t_sum(*, context: Dict[str, Any]) -> Dict[str, Any]:
    a = int(context.get("parse.a", 0))  # Read parsed a.
    b = int(context.get("parse.b", 0))  # Read parsed b.
    return {"total": a + b}  # Emit the sum.


def demo() -> None:
    tasks = [
        Task(name="fetch", fn=t_fetch),  # No deps.
        Task(name="parse", fn=t_parse, deps=["fetch"]),  # After fetch.
        Task(name="sum", fn=t_sum, deps=["parse"]),  # After parse.
    ]  # Define the small DAG.
    wf = Workflow(tasks)  # Build engine with tasks.
    results = wf.run(context={"n": 5})  # Execute with an input.
    for name, res in results.items():  # Print compact results.
        print({"task": name, "status": res.status, "latency_ms": res.latency_ms})


if __name__ == "__main__":
    demo()  # Run the demo when executed directly.
# end::ch07_workflow[]
