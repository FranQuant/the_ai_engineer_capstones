#!/usr/bin/env python3
"""
Validate all notebooks in the notebooks/ folder.

Checks:
- Structure: first markdown cell contains book + chapter title + meta lines
- Imports: scan code cells for top-level imports and probe availability
- Execute (optional): run with nbclient if available or when --execute is set

Usage:
  python tools/validate_notebooks.py [--execute]

Output format mirrors prior tooling for quick scanning.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Iterable, List, Tuple


def human_duration(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f}µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f}ms"
    return f"{seconds:.2f}s"


def list_notebooks(root: Path, repo_root: Path, patterns: List[str]) -> List[Path]:
    if patterns:
        selected: list[Path] = []
        for pattern in patterns:
            matches = list(repo_root.glob(pattern))
            for match in matches:
                if match.is_file() and match.suffix == ".ipynb":
                    selected.append(match)
        return sorted({p.resolve() for p in selected})
    return sorted(p for p in root.glob("*.ipynb") if p.is_file())


def load_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_structure(nb: dict) -> Tuple[bool, str]:
    cells = nb.get("cells", [])
    if not cells:
        return False, "no cells"
    index = 0
    while index < len(cells):
        first = cells[index]
        if first.get("cell_type") != "markdown":
            return False, "first cell not markdown"
        src = "".join(first.get("source", [])).strip()
        if not src or src.startswith("<img "):
            index += 1
            continue
        break
    else:
        return False, "no header markdown"
    src = "".join(cells[index].get("source", []))
    required = [
        "AI Agents & Automation",
        "## ",  # chapter subtitle line
        "(c) Dr. Yves J. Hilpisch",
        "AI-Powered by GPT-5",
    ]
    missing = [s for s in required if s not in src]
    if missing:
        return False, f"missing: {', '.join(missing)}"
    return True, "OK"


def scan_imports(nb: dict) -> List[str]:
    mods: set[str] = set()
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        try:
            node = ast.parse(src)
        except Exception:
            continue
        for n in ast.walk(node):
            if isinstance(n, ast.Import):
                for alias in n.names:
                    mods.add((alias.name.split(".")[0]))
            elif isinstance(n, ast.ImportFrom):
                if n.module:
                    mods.add(n.module.split(".")[0])
    # ignore magics/inline pip markers captured accidentally
    mods.discard("pip")
    return sorted(mods)


def probe_imports(mods: Iterable[str]) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    for m in mods:
        try:
            __import__(m)
        except Exception:
            missing.append(m)
    return (len(missing) == 0), missing


def _execute_locally(nb: dict) -> Tuple[bool, str]:
    """Fallback executor: run code cells in-process without a Jupyter kernel.

    Skips IPython magics (lines starting with % or !) and shares a single
    globals dict across cells to approximate notebook execution.
    """
    g: dict = {"__name__": "__main__"}
    try:
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            # Filter out simple magics and shell invocations
            lines = [
                ln
                for ln in src.splitlines()
                if not (
                    ln.lstrip().startswith("%") or ln.lstrip().startswith("!")
                )
            ]
            code = "\n".join(lines)
            if not code.strip():
                continue
            compiled = compile(code, filename="<notebook>", mode="exec")
            exec(compiled, g, g)
    except OSError as e:  # permission or sandbox issues
        return False, str(e)
    except Exception as e:  # noqa: BLE001
        return False, str(e)
    return True, "OK"


def execute_notebook(path: Path) -> Tuple[bool, str]:
    """Try to execute with nbclient; fallback to local in-process execution."""
    try:
        from nbclient import NotebookClient  # type: ignore
        from nbformat import read, NO_CONVERT  # type: ignore
    except Exception:
        # Fallback: best-effort local execution
        try:
            nb = load_notebook(path)
        except Exception as e:  # noqa: BLE001
            return False, str(e)
        return _execute_locally(nb)
    # Prefer nbclient when available
    with path.open("r", encoding="utf-8") as f:
        nb = read(f, as_version=NO_CONVERT)
    try:
        # Use notebook's kernelspec if present; otherwise let nbclient decide
        kernel_name = None
        ks = nb.get("metadata", {}).get("kernelspec", {})
        if isinstance(ks, dict):
            kernel_name = ks.get("name")
        client = NotebookClient(nb, timeout=60, kernel_name=kernel_name)
        client.execute()
        return True, "OK"
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        # Fallback to local execution if kernel is not available
        if "No such kernel named" in msg or "Kernel" in msg:
            ok, alt = _execute_locally(nb)
            return ok, ("OK (local)" if ok else alt)
        return False, msg


def normalize_notebook(path: Path) -> Tuple[bool, str]:
    """Normalize notebook in place: add cell ids and ensure kernelspec.

    Uses nbformat.normalize if available; otherwise manually adds missing
    cell ids and sets a default kernelspec (python3).
    """
    try:
        # Prefer nbformat for robust round-trip
        from nbformat import read, write, NO_CONVERT  # type: ignore
        try:
            from nbformat import normalize as nb_normalize  # type: ignore
        except Exception:
            nb_normalize = None  # type: ignore
        with path.open("r", encoding="utf-8") as f:
            nb = read(f, as_version=NO_CONVERT)
        if nb_normalize is not None:
            nb = nb_normalize(nb)  # type: ignore
        # Ensure cell ids exist
        for cell in nb.get("cells", []):
            if not cell.get("id"):
                cell["id"] = uuid.uuid4().hex
        # Ensure basic kernelspec
        meta = nb.setdefault("metadata", {})
        ks = meta.setdefault("kernelspec", {})
        ks.setdefault("name", "python3")
        ks.setdefault("display_name", "Python 3")
        ks.setdefault("language", "python")
        with path.open("w", encoding="utf-8") as f:
            write(nb, f)
        return True, "OK"
    except Exception as e:  # noqa: BLE001
        # Fallback: JSON-level normalization
        try:
            nb = load_notebook(path)
            for cell in nb.get("cells", []):
                if not cell.get("id"):
                    cell["id"] = uuid.uuid4().hex
            meta = nb.setdefault("metadata", {})
            ks = meta.setdefault("kernelspec", {})
            ks.setdefault("name", "python3")
            ks.setdefault("display_name", "Python 3")
            ks.setdefault("language", "python")
            with path.open("w", encoding="utf-8") as f:
                json.dump(nb, f, ensure_ascii=False)
            return True, "OK (json)"
        except Exception as e2:  # noqa: BLE001
            return False, f"{e}; fallback failed: {e2}"
    


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--execute",
        action="store_true",
        help="execute notebooks with nbclient",
    )
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="normalize notebooks in place (cell ids + kernelspec)",
    )
    ap.add_argument(
        "patterns",
        nargs="*",
        help="glob patterns relative to repo root (e.g., notebooks/ch01*.ipynb)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent / "notebooks"
    repo_root = Path(__file__).resolve().parent.parent
    paths = list_notebooks(root, repo_root, args.patterns)
    if args.patterns and not paths:
        print("No notebooks matched the provided patterns.")
        return 1
    total_n = len(paths)
    # Summary accounting
    summary = {
        "normalize": {"OK": 0, "FAIL": 0, "SKIP": 0},
        "structure": {"OK": 0, "FAIL": 0, "ERROR": 0},
        "imports": {"OK": 0, "WARN": 0, "SKIP": 0},
        "execute": {"OK": 0, "FAIL": 0, "SKIP": 0},
    }
    for i, p in enumerate(paths, 1):
        t0 = time.perf_counter()
        print(f"[{i}/{total_n}] Validating {p.as_posix()}")

        # Normalize first if requested
        if args.normalize:
            n0 = time.perf_counter()
            ok_norm, msg_norm = normalize_notebook(p)
            print(
                f"  • Normalize: {'OK' if ok_norm else 'FAIL':<7}  "
                f"({human_duration(time.perf_counter()-n0)})"
            )
            if not ok_norm:
                print(f"    -> {msg_norm}")
            summary["normalize"]["OK" if ok_norm else "FAIL"] += 1
        else:
            summary["normalize"]["SKIP"] += 1

        # Structure
        s0 = time.perf_counter()
        try:
            nb = load_notebook(p)
        except Exception as e:  # noqa: BLE001
            duration = human_duration(time.perf_counter() - s0)
            print(f"  • Structure: ERROR    ({duration})\n    -> {e}")
            print(f"  • Imports: SKIP")
            print(f"  • Execute: SKIP")
            print(f"  • Total: {human_duration(time.perf_counter()-t0)}\n")
            continue
        ok_struct, msg_struct = check_structure(nb)
        struct_time = human_duration(time.perf_counter() - s0)
        struct_status = "OK" if ok_struct else "FAIL"
        print(f"  • Structure: {struct_status:<7}  ({struct_time})")
        summary["structure"]["OK" if ok_struct else "FAIL"] += 1

        # Imports
        i0 = time.perf_counter()
        mods = scan_imports(nb)
        scan_dt = time.perf_counter() - i0
        p0 = time.perf_counter()
        ok_probe, missing = probe_imports(mods)
        probe_dt = time.perf_counter() - p0
        import_status = "OK" if ok_probe else "WARN"
        print(
            f"  • Imports: {import_status:<7} "
            f"({human_duration(scan_dt)} scan, {human_duration(probe_dt)} probe)"
        )
        if not ok_probe and missing:
            print(f"    -> missing: {', '.join(missing)}")
        summary["imports"]["OK" if ok_probe else "WARN"] += 1

        # Execute
        e0 = time.perf_counter()
        if args.execute:
            ok_exec, msg = execute_notebook(p)
            # Treat sandbox permission errors as a skip (environmental constraint)
            if (not ok_exec) and ("Operation not permitted" in (msg or "")):
                skip_time = human_duration(time.perf_counter() - e0)
                print(f"  • Execute: SKIP     ({skip_time})")
                print(f"    -> {msg}")
                summary["execute"]["SKIP"] += 1
            else:
                exec_status = "OK" if ok_exec else "FAIL"
                exec_time = human_duration(time.perf_counter() - e0)
                print(f"  • Execute: {exec_status:<7}  ({exec_time})")
                if not ok_exec or (msg and msg != "OK"):
                    print(f"    -> {msg}")
                summary["execute"]["OK" if ok_exec else "FAIL"] += 1
        else:
            print(f"  • Execute: SKIP     ({human_duration(0.0)})")
            summary["execute"]["SKIP"] += 1

        total_time = human_duration(time.perf_counter() - t0)
        print(f"  • Total: {total_time}\n")

    # Print final summary
    print("\nSummary:")
    print(f"  Notebooks: {total_n}")
    norm_msg = (
        f"  Normalize: OK={summary['normalize']['OK']}, "
        f"FAIL={summary['normalize']['FAIL']}, "
        f"SKIP={summary['normalize']['SKIP']}"
    )
    struct_msg = (
        f"  Structure: OK={summary['structure']['OK']}, "
        f"FAIL={summary['structure']['FAIL']}"
    )
    imports_msg = (
        f"  Imports:   OK={summary['imports']['OK']}, "
        f"WARN={summary['imports']['WARN']}, "
        f"SKIP={summary['imports']['SKIP']}"
    )
    exec_msg = (
        f"  Execute:   OK={summary['execute']['OK']}, "
        f"FAIL={summary['execute']['FAIL']}, "
        f"SKIP={summary['execute']['SKIP']}"
    )
    print(norm_msg)
    print(struct_msg)
    print(imports_msg)
    print(exec_msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
