<p align="center">
  <img src="assets/tae_logo.png" width="160">
</p>

<h1 align="center">The AI Engineer — Capstones </h1>

<p align="center">
  <b>Gradient Descent • Backpropagation • Tiny Transformer • Agent Demo</b><br>
  Structured, reproducible solutions for all capstone projects (Nov 2025 Cohort).
</p>

---

## Overview

This repository contains my complete work for the **The AI Engineer (Nov 2025 Cohort)**.  
It follows the official 4-week structure:

- **Core Track:** ML fundamentals, optimization, PyTorch, transformers  
- **Engineering Track:** reproducibility, tooling, configs, clean ML code

Each capstone lives in its own folder with:

- **Notebooks**
- **Python scripts**
- **Figures and experiment outputs**
- **Environment specifications (`environment.yml`)**
- **Short write-ups summarizing intuition + implementation**

---

## Weekly Capstones

### 🟦 **Week 1 — Gradient Descent Optimization**
> *From calculus to GD/SGD*  
Derive gradients manually, visualize loss landscapes, implement GD, SGD, and momentum from scratch.

Folder: `week01_gd_optimization/`

---

### 🟩 **Week 2 — Backpropagation**
> *Manual gradients → autograd parity → training loop*  
Chain rule by hand, build a tiny MLP from scratch, verify via PyTorch autograd.

Folder: `week02_backpropagation/`

---

### 🟧 **Week 3 — Tiny Transformer**
> *Attention → transformer blocks → mini-GPT*  
Implement tokenizer, attention, decoder blocks, training loop, sampling, and evaluation.

Folder: `week03_tiny_transformer/`

---

### 🟥 **Week 4 — Agent Demo**
> *Production-grade agent with monitoring hooks*  
Implement a minimal agent pipeline using clean engineering patterns.

Folder: `week04_agent_demo/`

---

## Environment

To ensure reproducibility:

```bash
conda env create -f environment.yml
conda activate tae
```

All dependencies adhere to:
- Python ≥ 3.10  
- PyTorch ≥ 2.2  
- JupyterLab  
- numpy / matplotlib / tqdm  
- ruff / pytest optional for engineering track

A `requirements.txt` is included.

---

## 🗂️ Repository Structure

```
the_ai_engineer_capstones/
│
├── README.md
├── .gitignore
├── environment.yml
│
├── week01_gd_optimization/
├── week02_backpropagation/
├── week03_tiny_transformer/
└── week04_agent_demo/
```

---

## 🔗 Helpful Links

- **TAE Resource Hub:**  
- **Discord Community:**   
- **Python & Math Repo:** https://github.com/yhilpisch/pmcode  
- **DL Basics Repo:** https://github.com/yhilpisch/dlcode  
- **LLM Code Repo:** https://github.com/yhilpisch/llmcode  

---
