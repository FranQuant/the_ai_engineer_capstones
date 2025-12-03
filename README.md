<table width="100%">
<tr>
<td style="vertical-align: top;">

<h1>The AI Engineer — Capstone Projects</h1>

This repository contains all capstone projects for <i>The AI Engineer</i> (Nov 2025 Cohort).  
Each week builds a complete, self-contained project with a clean software-engineering  
structure, reproducibility, diagnostics, and proper documentation.

</td>

<td align="right" width="200">
  <img src="assets/tae_logo.png" alt="TAE Logo" width="160">
</td>
</tr>
</table>

<!-- Badges -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Conda-Ready-green?logo=anaconda&logoColor=white">
  <img src="https://img.shields.io/badge/Colab-Friendly-brightgreen?logo=googlecolab&logoColor=white">
  <img src="https://img.shields.io/badge/License-Educational%20Use-lightgrey">
  <img src="https://img.shields.io/badge/Last%20Updated-December%202025-purple">
</p>


---

## Weekly Capstones Overview

| Week | Capstone | Summary | Colab Link |
|------|----------|---------|------------|
| **1** | **Gradient Descent Optimization** | Implement GD & SGD from scratch, analyze convergence, step-size sensitivity, and basin-dependent dynamics. | [Open in Colab](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week01_gd_optimization/gd_capstone_final.ipynb) |
| **2** | **Backpropagation** | Manual chain rule, custom autograd, tiny MLP, PyTorch autograd, and nn.Module training loop. | [01](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week02_backprop/01_numpy_manual.ipynb) • [02](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week02_backprop/02_pytorch_no_autograd.ipynb) • [03](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week02_backprop/03_pytorch_autograd.ipynb) • [04](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week02_backprop/04_pytorch_nn_module.ipynb) |
| **3** | **Tiny Transformer** | Build tokenizer, SDPA, MHA, pre-LN transformer block, decoder-only model, training loop, sampling, and a full diagnostics suite. | [Diagnostics Notebook](https://colab.research.google.com/github/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week03_transformers/mini_gpt_diagnostics.ipynb) |
| **4** | **Agent Demo** | Minimal LLM-powered agent with clean abstractions, tracing, telemetry, and a deterministic OPAL loop implementation. |[README](https://github.com/FranQuant/the_ai_engineer_capstones/blob/main/capstones/week04_agentic_incident_command/README_week04_capstone.md)


---

## Repository Structure

```
the_ai_engineer_capstones/
│
├── README.md # Root README (you are here)
├── assets/
│ └── tae_logo.png # Branding assets
│
└── capstones/
├── week01_gd_optimization/ # Gradient Descent capstone
├── week02_backprop/ # Backpropagation capstone
├── week03_transformers/ # Tiny Transformer capstone
└── week04_agent_demo/ # Week 4 MCP/Agent demo + OPAL loop           
```

Each capstone folder contains:
- A dedicated README  
- Clean, modular Python files  
- Colab-friendly notebooks  
- Deterministic seeds  
- Reproducible plots and outputs

---

## Environment & Reproducibility

This repository supports both **Conda** and **pip** workflows.

### Option A — Conda (recommended)

```bash
conda create -n tae python=3.11
conda activate tae
pip install -r requirements.txt 

```

### Option B — pip
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # optional, add later

```

·## License (Educational Use)

All content in this repository is provided **for educational and illustrative purposes only**.  
No guarantees are made regarding correctness, performance, reliability, or suitability for any production environment.

---

### ⚠️ Warning — Agentic Systems

Agentic systems — especially those capable of taking actions, orchestrating tools, or modifying state — can introduce **significant safety risks**.

Before using any such system outside a controlled environment, always:

- Validate all outputs manually  
- Run code inside a sandboxed environment  
- Apply strict guardrails and permissions  
- Never connect an agent to real infrastructure without full safety checks  

Use responsibly.

© 2025 Francisco Salazar

---


