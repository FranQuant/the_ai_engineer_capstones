# Week 1 Capstone — Gradient Descent Optimization

This folder contains the Week-1 Gradient Descent Optimization capstone for *The AI Engineer* program.  
The objective is to implement and analyze basic gradient-based optimization methods on simple one-dimensional functions, following the specifications in the Week-1 handout and coaching guide.

---

## Overview

This capstone studies two core instructional objectives:

### 1. **Quadratic baseline**

A convex objective used to analyze stability and step-size effects:

$$
q(x) = \tfrac{1}{2} x^2,
\qquad
q'(x) = x.
$$

### 2. **Simple cubic loss**

A non-convex objective used to illustrate basins of attraction, divergence, and differences between GD and SGD:

$$
f(x) = x^3 - 3x,
\qquad
f'(x) = 3x^2 - 3.
$$

The cubic has two stationary points:

- $x = -1$ — local maximum (unstable)  
- $x = +1$ — local minimum (stable for $0 < \eta < \tfrac{1}{3}$)

This non-convex structure makes it ideal for analyzing convergence, divergence, sensitivity to initialization, and stochastic behavior.

---

## Methods Implemented

### 1. **Deterministic Gradient Descent (GD)**

GD performs the update:

$$
x_{t+1} = x_t - \eta \nabla f(x_t).
$$

#### **Quadratic step-size sweep (required grid)**

The notebook uses the step-size set required by Week-1:

$$
\eta \in \{0.05,\; 0.10,\; 0.15,\; 0.20\}.
$$

This sweep highlights stability boundaries, convergence speed, and divergence at larger $\eta$.

#### **Cubic fixed step size**

A single learning rate is used:

$$
\eta_{\text{cubic}} = 0.05.
$$

This ensures stability around the attracting minimum at $x^\star = 1$.

---

### 2. **Stochastic Gradient Descent (SGD)**

SGD uses a noisy gradient estimator:

$$
g_t = f'(x_t) + \varepsilon_t,
\qquad
\varepsilon_t \sim \mathcal{N}(0, \sigma^2).
$$

Two learning-rate schedules are implemented:

#### **Constant step size**

$$
x_{t+1} = x_t - \eta\, g_t,
\qquad
\eta = 0.05.
$$

This schedule does **not** converge to the minimizer but instead stabilizes within a noise-controlled neighborhood whose radius scales like:

$$
\mathcal{O}\!\left(\sqrt{\eta\,\sigma^2}\right).
$$

#### **Diminishing step size**

$$
\eta_t = \frac{\eta_0}{1 + k t},
\qquad
\eta_0 = 0.05,\quad k = 0.01.
$$

Because $\eta_t \to 0$, the noise contribution contracts over time, producing tighter convergence than constant-step SGD.

---

## Experiments and Plots

The notebook includes four main experimental blocks:

### 1. **Quadratic GD step-size sweep**

Comparison of convergence speed and stability for  
$\eta \in \{0.05, 0.10, 0.15, 0.20\}$.

### 2. **Cubic GD from multiple initializations**

Shows the basin boundary at $x=-1$ and divergence when $x_0 < -1$.

### 3. **SGD trajectories**

- Constant-step SGD fluctuating around $x^\star = 1$  
- Diminishing-step SGD contracting toward the minimum  

### 4. **Convergence metrics (cubic)**

For GD, constant-step SGD, and diminishing-step SGD:

- **Final gap**

  $|f(x_T) - f(x^\star)|$


- **Best gap**


  $\min_{t \le T} |f(x_t) - f(x^\star)|$

- **Steps-to-tolerance**

  $|f(x_t) - f(x^\star)| < 10^{-4}$

All figures include labels, titles, and captions.

---

## Reproducibility

All results are fully reproducible:

- Global NumPy seed
  
  $\text{SEED} = 123$

- Centralized hyperparameters  
- Programmatically generated data  
- No external datasets  
- Runtime well under two minutes  
- Uses only **NumPy** and **Matplotlib**

---

## File Structure

```text
week01_gd_optimization/
│
├── gd_capstone.ipynb # Final capstone notebook
└── README_week01_capstone.md # This document
```

---

## How to Run

Runs top-to-bottom on:

- Google Colab  
- Local Jupyter Notebook  
- GitHub Codespaces  

### Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week01_gd_optimization/gd_capstone.ipynb
)

Dependencies: **NumPy** and **Matplotlib** only.

---

## Deliverables

This folder provides the complete Week-1 capstone implementation:

- Deterministic GD and Stochastic GD  
- Quadratic step-size sweep (using the required grid)  
- Cubic GD basin analysis  
- Constant vs diminishing-step SGD  
- Convergence metrics (final gap, best gap, steps-to-tolerance)  
- Fully reproducible, Colab-ready notebook  

This submission aligns with:

- **Gradient-Based Optimization Case Study**  
- **Week-1 Coaching Guide**
