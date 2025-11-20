# Gradient Descent Optimization Capstone — README

This folder contains the **Week‑1 Gradient Descent Optimization Capstone** for *The AI Engineer* program.

## 📌 Overview

This notebook implements and explores gradient‑based optimization on the non‑convex cubic function:

$$
f(x) = x^3 - 3x.
$$

It demonstrates:

- Deterministic Gradient Descent (GD)
- Stochastic Gradient Descent (SGD)
- Basin‑dependent behavior
- Step‑size sensitivity
- Local linearization (tangent‑line intuition)
- Reproducible experiments with fixed seeds

All figures are generated programmatically.

## 📁 Contents

- `gd_capstone_final.ipynb` — main notebook
- All plots are generated at runtime (no hidden data).

## ▶️ Open in Google Colab

Click the badge below to launch the notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/week01_gd_optimization/gd_capstone_final.ipynb)

## 🔁 Reproducibility

- A global `MASTER_SEED = 42` fixes RNG behavior.
- GD/SGD functions use deterministic update rules.
- All experiments run top‑to‑bottom without modification.

## 🚀 How to Run

1. Open in Colab using the badge above **or** clone the repo locally.
2. Install NumPy and Matplotlib (Colab already includes them).
3. Run the notebook sequentially — all plots and metrics are generated automatically.

## 📄 License

This capstone is provided for educational use under the terms of the repository license.
