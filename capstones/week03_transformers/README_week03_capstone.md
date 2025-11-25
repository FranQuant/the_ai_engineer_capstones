# Mini GPT — Week 03 Capstone (The AI Engineer)

Diagnostics & implementation of a tiny decoder-only Transformer LM trained in `train_mini_gpt.py`.

---

## 📘 Project Overview

This folder contains a complete, from-scratch implementation of a small GPT-style language model, including:

- Scaled Dot-Product Attention  
- Multi-Head Self-Attention  
- Transformer Blocks  
- A compact decoder-only LM (`MiniTransformerLM`)  
- Full training loop with LR warmup + cosine schedule  
- Saved checkpoint (`mini_gpt.pt`)  
- A Jupyter diagnostics suite for interpretability & visualization  

This capstone is fully runnable in Jupyter Lab or Google Colab.

---

## 📁 Repository Structure

```text
week03_transformers/
│
├── mini_transformer.py
├── transformer_block.py
├── multihead_attention.py
├── scaled_dot_product_attention.py
│
├── train_mini_gpt.py
├── mini_gpt_diagnostics.ipynb
│
└── README.md   ← (this file)
```

---

## 🧠 Model Architecture (Minimal Diagram)

```text
               ┌──────────────────────────┐
               │     Input Token IDs      │
               └────────────┬─────────────┘
                            ↓
               ┌──────────────────────────┐
               │ Token Embedding + PosEnc │
               └────────────┬─────────────┘
                            ↓
               ┌──────────────────────────┐
               │    Transformer Block     │
               │  ┌────────────────────┐  │
               │  │  LayerNorm (ln1)   │  │
               │  ├────────────────────┤  │
               │  │ Multi-Head Attn    │  │
               │  ├────────────────────┤  │
               │  │ Residual Add       │  │
               │  ├────────────────────┤  │
               │  │  LayerNorm (ln2)   │  │
               │  ├────────────────────┤  │
               │  │ Position FFN       │  │
               │  └────────────────────┘  │
               └────────────┬─────────────┘
                            ↓
               ┌──────────────────────────┐
               │    Final LayerNorm       │
               └────────────┬─────────────┘
                            ↓
               ┌──────────────────────────┐
               │     LM Head (Linear)     │
               └────────────┬─────────────┘
                            ↓
                     Next-token logits
```

---

## ▶️ Training the Mini GPT

Run:

```bash
python train_mini_gpt.py
```

You will see periodic training/validation loss and LR schedule:

```
step   50 | train 30.31 | val 39.99 | lr 3e-4
...
Saved checkpoint to mini_gpt.pt
```

---

## 🧪 Diagnostics Notebook

The notebook `mini_gpt_diagnostics.ipynb` includes:

- Attention heatmaps (per-head + averaged)
- Residual stream norms
- Embedding PCA/TSNE visualization
- Logits histogram + entropy
- Temperature, greedy, and top-k sampling

---

## 🔗 Google Colab Link

Open the diagnostics notebook in Colab:

👉  
https://colab.research.google.com/github/FranQuant/the_ai_engineer/blob/main/capstones/week03_transformers/mini_gpt_diagnostics.ipynb

*(Replace the path above if your repo uses a different directory.)*

---

## ✔️ Requirements

Install dependencies:

```bash
pip install matplotlib seaborn scikit-learn torch
```

---

## 📌 Notes

This capstone is intentionally small, transparent, and designed for learning:

- No external libraries for the Transformer blocks  
- No hidden helper utilities  
- Fully inspectable attention & residual streams  

The notebook provides interpretability tools similar to those used in real model-debugging workflows.

---

## 📜 License

MIT — feel free to use, modify, and publish.

