<table width="100%">
<tr>
<td style="vertical-align: top;">

<h1>Week 03 Capstone — Mini Transformer GPT</h1>

<p><strong>Four-Stage Capstone:</strong><br>
Tokenization → Positional Encoding → Multi-Head Attention → Mini GPT LM

<p>
This folder implements a fully-compliant decoder-only Transformer from first principles, following the Week-03 TAE capstone requirements.

</p>

</td>
<td align="right" width="200">
<img src="../../assets/tae_logo.png" alt="TAE Banner" width="160">
</td>
</tr>
</table>


## Project Overview

This capstone implements a complete minimal GPT-style language model:

- **Scaled Dot-Product Attention** (with numeric + causal mask tests)  
- **Multi-Head Self-Attention**  
- **Pre-LayerNorm Transformer Blocks**  
- **Sinusoidal positional encodings** (per Week-03 spec)  
- **Default causal masking inside the LM**  
- **Dropout hooks** (set to 0.0 but present architecturally)  
- **Mini GPT decoder-only model (`MiniTransformerLM`)**  
- **Training loop with LR warmup + cosine decay**  
- **Saved checkpoint (`mini_gpt.pt`)**  
- **Diagnostics notebook for attention maps, embeddings, entropy, sampling, etc.**

Everything is implemented **from scratch** without helper libraries.
The model is intentionally tiny and transparent, enabling full interpretability and step-by-step introspection of attention, residuals, embeddings, and sampling behavior.

---

## Repository Structure

```text
week03_transformers/
│
├── scaled_dot_product_attention.py      # Step 1: Scaled Dot-Product Attention
├── multihead_attention.py               # Step 2: Multi-Head Attention (MHA)
├── transformer_block.py                 # Step 3: Transformer Block (pre-LN)
├── mini_transformer.py                  # Step 4: Mini GPT-style LM
├── train_mini_gpt.py                    # Step 5: End-to-end training script
├── mini_gpt.pt                          # Saved checkpoint
├── mini_gpt_diagnostics.ipynb           # Step 6: Interpretability suite
│
└── README_week03_capstone.md            # ← This file

```
---

## Model Architecture (Minimal Diagram)

```text
            Input Token IDs
                   │
                   ▼
   Token Embedding + Sinusoidal Positional Encoding
                   │
                   ▼
    ┌─────────────────────────────┐
    │     Transformer Block       │
    │  • LayerNorm (pre-LN)       │
    │  • Multi-Head Attention     │
    │  • Residual Add + Dropout   │
    │  • LayerNorm (pre-LN)       │
    │  • Feedforward (GELU)       │
    │  • Residual Add + Dropout   │
    └─────────────────────────────┘
                   │
                   ▼
            Final LayerNorm
                   │
                   ▼
    LM Head (Linear → vocab_size)
                   │
                   ▼
            Next-Token Logits

```


## Training the Mini GPT

Run:

```bash
python train_mini_gpt.py
```

You’ll see a training log with warmup + cosine LR schedule:

```
step    0 | train loss 47.28 | val loss 56.72 | lr 3.00e-06
step  100 | train loss 3.34  | val loss 7.96  | lr 3.00e-04
...
Model checkpoint saved → mini_gpt.pt
```
---

Sample model outputs:
```text
=== Sample Model Output ===

hi transformer
hello tiny transformer
hello week three
hi transformer
hello tiny transformer
hello tiny transformer
```
---

## Diagnostics Notebook

The notebook `mini_gpt_diagnostics.ipynb` includes:

- Attention heatmaps (per-head + averaged)
- Residual stream norms
- Embedding PCA/TSNE visualization
- Logits histogram + entropy
- Temperature, greedy, and top-k sampling

---


## Google Colab Link

Open the diagnostics notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/FranQuant/the_ai_engineer/blob/main/capstones/week03_transformers/mini_gpt_diagnostics.ipynb
)


---

## Requirements

Install dependencies:

```bash
pip install matplotlib seaborn scikit-learn torch
```

---

## Notes

This capstone is intentionally small, transparent, and designed for learning:

- No external libraries for the Transformer blocks  
- No hidden helper utilities  
- Fully inspectable attention & residual streams  

The notebook provides interpretability tools similar to those used in real model-debugging workflows.
