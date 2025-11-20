<p align="center">
  <img src="assets/tae_logo.png" width="160">
</p>


# The AI Engineer — Capstone Projects 

---

<p align="center">

  <!-- Python Version -->
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" alt="Python 3.11">

  <!-- Conda Ready -->
  <img src="https://img.shields.io/badge/Conda-Ready-green?logo=anaconda&logoColor=white" alt="Conda Ready">

  <!-- Colab Friendly -->
  <img src="https://img.shields.io/badge/Colab-Friendly-brightgreen?logo=googlecolab&logoColor=white" alt="Colab Friendly">

  <!-- License -->
  <img src="https://img.shields.io/badge/License-Educational%20Use-lightgrey" alt="Educational Use License">

  <!-- Last Updated -->
  <img src="https://img.shields.io/badge/Last%20Updated-November%202025-purple" alt="Last Updated">

</p>

---

This repository contains my complete implementations of the capstone projects for *The AI Engineer* program (Nov 2025 Cohort).  
Each capstone is designed to be clean, reproducible, well‑structured, and aligned with software‑engineering best practices.

---

## 📘 Weekly Capstones Overview

| Week | Capstone | Summary | Colab Link |
|------|----------|---------|------------|
| **1** | **Gradient Descent Optimization** | Implement GD & SGD from scratch, analyze convergence, step‑size sensitivity, and basin‑dependent dynamics. | [Open in Colab](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week01_gd_optimization/gd_capstone_final.ipynb) |
| **2** | **Backpropagation** | Manual chain rule, custom autograd, tiny MLP, compare manual vs. PyTorch autograd. | _Coming soon_ |
| **3** | **Tiny Transformer** | Build tokenizer, attention, decoder block, training loop, inference sampling. | _Coming soon_ |
| **4** | **Agent Demo** | Minimal LLM‑powered agent with clean abstractions, tracing, and monitoring. | _Coming soon_ |

---

## 📁 Repository Structure

```
the_ai_engineer_capstones/
│
├── README.md
├── assets/
│   └── tae_logo.png
│
└── capstones/
    ├── week01_gd_optimization/
    │   ├── gd_capstone_final.ipynb
    │   └── README_week01_capstone.md
    ├── week02_backpropagation/        (placeholder)
    ├── week03_tiny_transformer/       (placeholder)
    └── week04_agent_demo/             (placeholder)
```

---

## 🔬 Environment & Reproducibility

Create the environment:

```bash
conda env create -f environment.yml
conda activate tae
```

All notebooks follow these principles:

- deterministic execution with fixed seeds  
- clear separation of code, plots, commentary  
- no hidden data (🏗 everything is generated programmatically)  
- clean folder structure: each capstone self‑contained under `capstones/`  

---

## 🔗 Helpful External Resources

- Python & Math: https://github.com/yhilpisch/pmcode  
- Deep Learning Basics: https://github.com/yhilpisch/dlcode  
- LLM Fundamentals: https://github.com/yhilpisch/llmcode  

---

## ✔️ Status

**Week 1 — Completed**  
Weeks 2–4 — *In progress (to be released sequentially)*  
