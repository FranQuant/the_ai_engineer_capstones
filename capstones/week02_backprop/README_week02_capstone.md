<table width="100%">
<tr>

<td style="vertical-align: top;">

<h1>Week 02 Capstone — Backpropagation (TAE Program)</h1>

<p><strong>Four-Stage Capstone:</strong><br>
Manual Gradients → PyTorch Autograd → <code>nn.Module</code>
</p>

<p>
This folder contains the full Week-02 Capstone completed following the TAE Program structure.<br>
The goal is to implement a tiny <strong>1-hidden-layer MLP</strong>, step-by-step, moving from fully manual NumPy backprop to PyTorch’s <code>nn.Module</code> API.
</p>

</td>

<td align="right" width="200">
<img src="../../assets/tae_logo.png" alt="TAE Banner" width="160">
</td>

</tr>
</table>














# Week 02 — Backpropagation (TAE Program)
## Four-Stage Capstone: Manual Gradients → PyTorch Autograd → `nn.Module`

This folder contains the full Week-02 Capstone completed following the TAE Program structure.
The goal is to implement a tiny **1-hidden-layer MLP**, step-by-step, moving from fully manual NumPy backprop to PyTorch’s `nn.Module` API.

All notebooks use:

- Deterministic seeds  
- Same XOR-style synthetic dataset  
- Same MLP architecture  

Forward pass:

$$
a_1 = W_1 x + b_1,\qquad
h_1 = \mathrm{ReLU}(a_1),\qquad
f = W_2 h_1 + b_2
$$

Loss:

$$
L = \frac{1}{2}(f - y)^2
$$

---

## Notebook 01 — `01_numpy_manual.ipynb`

**Goal:** Build the complete forward + backward pass manually using NumPy.

**Features:**

- Manual ReLU activation and derivative  
- Full chain-rule backprop  
- Gradient check and shape verification  
- Serves as the *source of truth* for all later notebooks  

---

## Notebook 02 — `02_pytorch_no_autograd.ipynb`

**Goal:** Reproduce the NumPy forward pass in PyTorch **without** autograd.

**Features:**

- Torch tensors with `requires_grad = False`  
- Forward consistency test vs. NumPy  
- Ensures the NumPy → Torch math is aligned before enabling autograd  
- Same datasets, seeds, and initialization  

---

## Notebook 03 — `03_pytorch_autograd.ipynb`

**Goal:** Use PyTorch autograd to compute gradients and compare with manual backprop.

**Features:**

- Parameters with `requires_grad = True`  
- Forward → loss → `loss.backward()`  
- Manual vs. autograd gradient comparison  
- Optional finite-difference gradient check  
- All gradients match the analytic ones from Notebook 01  
- Prepares structure for `nn.Module` in Notebook 04  

---

## Notebook 04 — `04_pytorch_nn_module.ipynb`

**Goal:** Wrap the model in a clean `nn.Module` and train with mini-batch SGD.

**Features:**

- Custom `TwoLayerXOR` class (manual parameters inside `nn.Module`)  
- `DataLoader` with deterministic shuffling  
- SGD training loop (~200 epochs)  
- Loss curve + gradient-norm diagnostics  
- Final XOR classification accuracy: ~0.93–0.97  
- Fully Colab-ready  

---

## Summary

This 4-notebook progression builds the full intuition and engineering workflow:

1. Manual gradients  
2. Torch forward  
3. Torch autograd  
4. `nn.Module` + training loop  

It prepares the foundation for future Capstones involving:

- Deep networks  
- Optimizers  
- Regularization  
- Vision/sequence models  
- Reinforcement learning  
- Agentic training workflows  
