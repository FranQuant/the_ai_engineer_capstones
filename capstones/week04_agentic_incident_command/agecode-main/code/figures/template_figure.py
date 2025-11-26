#!/usr/bin/env python3
# Engineering Mathematics & Machine Learning — Template Project
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-Powered by GPT-5
"""Template figure generator (Matplotlib-only).

Saves both SVG and 300 DPI PNG under figures/.
"""
from __future__ import annotations

from pathlib import Path  # Resolve repo-relative figure paths.

import numpy as np  # Numerical grid generation.
import matplotlib.pyplot as plt  # Plotting API.


def main() -> None:
    plt.style.use("seaborn-v0_8")  # Consistent styling across plots.
    x = np.linspace(0.0, 2.0 * np.pi, 200)  # Even grid over two periods.
    y = np.sin(x)  # Sample sine wave.

    fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)  # Canvas.
    ax.plot(x, y, label="sin(x)")  # Line plot.
    ax.set_title("Template Figure: sin(x)")  # Title.
    ax.set_xlabel("x")  # X-axis label.
    ax.set_ylabel("y")  # Y-axis label.
    ax.grid(alpha=0.25)  # Light grid for readability.
    ax.legend(frameon=False)  # Minimal legend styling.

    repo_root = Path(__file__).resolve().parents[2]  # Project root.
    output_dir = repo_root / "figures"  # Shared figures directory.
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure writable path.
    fig.savefig(output_dir / "template_figure.svg")  # Vector export.
    fig.savefig(output_dir / "template_figure.png", dpi=300)  # Raster export.


if __name__ == "__main__":
    main()
