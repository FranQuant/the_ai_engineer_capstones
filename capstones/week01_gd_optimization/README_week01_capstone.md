# Week 1 Capstone — Gradient Descent Optimization

This folder contains the Week-1 Gradient Descent Optimization capstone for *The AI Engineer* program.  
The objective of this assignment is to implement and analyze basic gradient-based optimization methods on simple one-dimensional functions, following the requirements of the Week-1 handout and coaching guide.

---

## Overview

This capstone studies two instructional objectives:

### 1. **Quadratic baseline**

A convex function used to study stability and step-size effects:

$$
q(x) = \tfrac{1}{2} x^2,
\qquad
q'(x) = x.
$$

### 2. **Simple cubic loss**

A non-convex function used to illustrate basins of attraction, divergence, and GD/SGD behavior:

$$
f(x) = x^3 - 3x,
\qquad
f'(x) = 3x^2 - 3.
$$

It has two stationary points:
- $x = -1$: local maximum (unstable),
- $x = +1$: local minimum (stable for $0 < \eta < \tfrac{1}{3}$).

This structure allows us to examine:

- Convergence vs divergence  
- Sensitivity to initialization  
- The effect of noise in stochastic gradients  

---

## Methods Implemented

### 1. **Deterministic Gradient Descent (GD)**

Exact gradient updates of the form:

$$
x_{t+1} = x_t - \eta \nabla f(x_t).
$$

- **Quadratic step-size sweep**:
  $$
  \eta \in \{0.01,\; 0.05,\; 0.10,\; 0.20\}.
  $$

- **Cubic fixed step size**:
  $$
  \eta_{\text{cubic}} = 0.05.
  $$

---

### 2. **Stochastic Gradient Descent (SGD)**

SGD uses a noisy gradient estimator:

$$
g_t = f'(x_t) + \varepsilon_t,
\qquad
\varepsilon_t \sim \mathcal{N}(0, \sigma^2).
$$

Two schedules are implemented:

#### **Constant step size**

$$
x_{t+1} = x_t - \eta\, g_t,
\qquad
\eta = 0.05.
$$

This converges only to a noise-controlled neighborhood of the minimizer.

#### **Diminishing step size**

$$
\eta_t = \frac{\eta_0}{1 + k t},
\qquad
\eta_0 = 0.05,\quad k = 0.01.
$$

This gradually reduces noise and produces tighter convergence.

---

## Experiments and Plots

The notebook includes:

### **1. Quadratic GD step-size sweep**

Comparison of convergence rates for different values of $\eta$.

### **2. Cubic GD from multiple initializations**

Illustrates the basin boundary at $x=-1$ and divergence when $x_0<-1$.

### **3. SGD trajectories**

- Constant step-size SGD wandering around $x^\star = 1$  
- Diminishing step-size SGD contracting toward the minimizer  

### **4. Convergence metrics on the cubic**

For GD, SGD-constant, and SGD-diminishing:

- **Final gap**
  $$
  |f(x_T) - f(x^\star)|.
  $$

- **Best gap**
  $$
  \min_{t \le T} |f(x_t) - f(x^\star)|.
  $$

- **Steps-to-tolerance**
  $$
  |f(x_t) - f(x^\star)| < 10^{-4}.
  $$

All plots include titles, axis labels, and captions.

---

## Reproducibility

All experiments use:

- A single global NumPy seed  
  $$
  \text{SEED} = 123
  $$
- Centralized hyperparameters  
- Programmatically generated data (no external files)  
- Execution time under two minutes  

The notebook is fully Colab-ready and includes an “Open in Colab” badge.

---

## File Structure

week01_gd_optimization/
│
├── gd_capstone.ipynb # Final capstone notebook
└── README_week01_capstone.md # This document


---

## How to Run

Runs top-to-bottom without modification on:

- **Google Colab**  
- **Local Jupyter Notebook**  
- **GitHub Codespaces**

### Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week01_gd_optimization/gd_capstone.ipynb
)

Dependencies: **NumPy** and **Matplotlib** only.

---

## Deliverables

This folder provides the required artifacts for the Week-1 capstone:

- GD and SGD implementations  
- Required plots and convergence metrics  
- Reproducible, Colab-ready notebook  
- No external data dependencies  

This submission aligns with the expectations described in:

- **Gradient-Based Optimization Case Study**  
- **Week-1 Coaching Guide**  



