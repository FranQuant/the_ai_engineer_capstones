<table width="100%">
<tr>
<td style="vertical-align: top;">

<h1>Week 1 Capstone — Gradient Descent Optimization</h1>

This folder contains the Week-1 capstone for *The AI Engineer* program.  
The goal is to implement and visualize basic gradient-based optimization methods on simple 1-D functions.

</td>

<td align="right" width="200">
<img src="../../assets/tae_logo.png" alt="TAE Banner" width="160">
</td>
</tr>
</table>

## Overview

The notebook explores how Gradient Descent (GD) and Stochastic Gradient Descent (SGD) behave on:

1. **A quadratic baseline**
   - Simple convex objective  
   - Used to study step-size stability  
   - Includes the required learning-rate sweep

2. **A cubic non-convex function**
   - Shows basins of attraction and divergence  
   - Demonstrates differences between GD and SGD  
   - Highlights noisy vs. diminishing-noise behavior

Both objectives are one-dimensional so the dynamics can be plotted and understood visually.

---

## What’s Implemented

- Deterministic Gradient Descent  
- SGD with:
  - Constant step size  
  - Diminishing step size  
- Shared RNG for reproducibility  
- Convergence metrics:
  - Final gap  
  - Best gap  
  - Steps-to-tolerance  
- Plots for:
  - GD step-size sweep (quadratic)  
  - Cubic function & derivative  
  - GD trajectories from multiple initializations  
  - SGD vs. diminishing-SGD trajectories  

Runtime is under a few seconds and requires only **NumPy** and **Matplotlib**.

---

## File Structure

```test
week01_gd_optimization/
│
├── gd_capstone.ipynb # Full implementation & plots
└── README_week01_capstone.md # This document
```

---

## How to Run

The notebook runs top-to-bottom on:

- Google Colab  
- Local Jupyter Notebook  
- GitHub Codespaces  

### Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week01_gd_optimization/gd_capstone.ipynb
)

Dependencies: **NumPy** and **Matplotlib** only.
