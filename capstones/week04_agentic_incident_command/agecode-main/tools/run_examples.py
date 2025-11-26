#!/usr/bin/env python3
"""Execute every example script under code/examples and report the results."""

from __future__ import annotations  # future annotations

import subprocess  # run Python scripts in subprocesses
import sys  # access interpreter path
import time  # measure execution duration
from dataclasses import dataclass  # lightweight result container
from pathlib import Path  # path manipulations


@dataclass
class ScriptResult:
    path: Path  # path to the executed script
    status: str  # OK, FAIL, or SKIP
    duration: float  # execution time in seconds
    message: str  # optional message or error payload


def human_duration(seconds: float) -> str:
    """Format durations for concise console output."""

    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f}µs"
    if seconds < 1:
        return f"{seconds * 1e3:.1f}ms"
    return f"{seconds:.2f}s"


def run_script(script: Path) -> ScriptResult:
    """Execute a Python script and capture the result."""

    start = time.perf_counter()  # timestamp for duration
    try:
        completed = subprocess.run(  # run script as subprocess
            [sys.executable, str(script)],
            cwd=script.parent,
            check=False,
            capture_output=True,
            text=True,
        )
        duration = time.perf_counter() - start  # compute elapsed time
        ok = completed.returncode == 0  # success if exit code zero
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        message = "OK" if ok else stderr or stdout
        status = "OK" if ok else "FAIL"
        if status == "FAIL" and "No module named" in message:
            status = "SKIP"
        relative = script.relative_to(script.parents[1])
        return ScriptResult(relative, status, duration, message)
    except OSError as exc:  # handle OS-level failures
        duration = time.perf_counter() - start
        return ScriptResult(script, "FAIL", duration, str(exc))


def gather_scripts(examples_dir: Path) -> list[Path]:
    """Return Python scripts under code/examples."""

    return sorted(p for p in examples_dir.rglob("*.py") if p.is_file())


def main() -> int:
    """Entrypoint for running all example scripts."""

    repo_root = Path(__file__).resolve().parent.parent  # repository root
    examples_dir = repo_root / "code" / "examples"  # examples directory
    if not examples_dir.exists():
        print(f"[examples] Directory missing: {examples_dir}")
        return 1

    scripts = gather_scripts(examples_dir)  # collect script paths
    total = len(scripts)
    if total == 0:
        print("[examples] No scripts found.")
        return 0

    results: list[ScriptResult] = []
    for index, script in enumerate(scripts, start=1):  # run each script
        rel_path = script.relative_to(repo_root)
        print(f"[{index}/{total}] Running {rel_path}")
        result = run_script(script)
        status = result.status
        print(f"  • Execute: {status:<4} ({human_duration(result.duration)})")
        if status != "OK" and result.message:
            print(f"    -> {result.message}")
        results.append(result)

    # Summary
    ok_count = sum(1 for r in results if r.status == "OK")
    fail_count = sum(1 for r in results if r.status == "FAIL")
    skip_count = sum(1 for r in results if r.status == "SKIP")
    print("\nSummary:")
    print(f"  Scripts:  {total}")
    print(f"  Success:  {ok_count}")
    print(f"  Skipped:  {skip_count}")
    print(f"  Failed:   {fail_count}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":  # script entry point
    raise SystemExit(main())
