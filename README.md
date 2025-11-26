<p align="center">
  <img src="assets/tae_logo.png" alt="TAE Logo" width="160">
</p>

<h1 align="center">The AI Engineer — Capstone Projects</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/Conda-Ready-green?logo=anaconda&logoColor=white" alt="Conda Ready">
  <img src="https://img.shields.io/badge/Colab-Friendly-brightgreen?logo=googlecolab&logoColor=white" alt="Colab Friendly">
  <img src="https://img.shields.io/badge/License-Educational%20Use-lightgrey" alt="Educational Use License">
  <img src="https://img.shields.io/badge/Last%20Updated-November%202025-purple" alt="Last Updated">
</p>

<p align="center">··········································</p>

This repository contains my complete implementations of the capstone projects for <em>The AI Engineer</em> program (Nov 2025 Cohort).  
Each capstone is clean, reproducible, and aligned with software-engineering best practices.

<p align="center">··········································</p>



## Weekly Capstones Overview

| Week | Capstone | Summary | Colab Link |
|------|----------|---------|------------|
| **1** | **Gradient Descent Optimization** | Implement GD & SGD from scratch, analyze convergence, step‑size sensitivity, and basin‑dependent dynamics. | [Open in Colab](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week01_gd_optimization/gd_capstone_final.ipynb) |
| **2** | **Backpropagation** | Manual chain rule, custom autograd, tiny MLP, PyTorch autograd, and nn.Module training loop. | [01](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week02_backprop/01_numpy_manual.ipynb) • [02](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week02_backprop/02_pytorch_no_autograd.ipynb) • [03](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week02_backprop/03_pytorch_autograd.ipynb) • [04](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week02_backprop/04_pytorch_nn_module.ipynb) |
| **3** | **Tiny Transformer** | Build tokenizer, SDPA, MHA, pre-LN transformer block, decoder-only model, training loop, sampling, and a full diagnostics suite. | [Diagnostics Notebook](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week03_transformers/mini_gpt_diagnostics.ipynb) | 
| **4** | **Agent Demo** | Minimal LLM‑powered agent with clean abstractions, tracing, and monitoring. | _Coming soon_ |

---

## Repository Structure

```
the_ai_engineer_capstones/
│
├── README.md
├── assets/
│   └── tae_logo.png
│
└── capstones/
    ├── week01_gd_optimization/
    ├── week02_backprop/
    ├── week03_transformers/       
    └── week04_agent_demo/             
```

---

## Environment & Reproducibility

```
conda env create -f environment.yml
conda activate tae
```

- deterministic seeds  
- clean separation of logic/plots  
- fully programmatic datasets  
- each capstone self‑contained  

---

## Status

**Week 1 — Completed**  
**Week 2 — Completed**  
Weeks 3–4 — *In progress*  
