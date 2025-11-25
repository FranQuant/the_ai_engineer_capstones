"""
Step 1 (Corrected): Scaled dot-product attention with optional causal mask
and tests aligned exactly with the Week-03 handout.

Grounding:
- Handout Sections 4.1–4.4 (raw score example, attention weights, output vectors).
- llmcode reference: code/ch7_attention.py.
- TAE Coach Guide: deterministic seeds, explicit shapes, micro-tests.
"""

from __future__ import annotations
import math
import torch

torch.manual_seed(0)


# ============================================================================
# Core attention function (with numerically stable softmax)
# ============================================================================
def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute Y = softmax(QK^T / sqrt(d_k)) @ V with optional mask.

    q, k, v: shape [T,D] or [B,T,D]
    mask: [T,T], [1,T,T], or [B,1,T,T] with 1=keep, 0=mask
    """
    assert q.shape == k.shape == v.shape, "q, k, v must share shape"

    original_2d = q.ndim == 2
    if original_2d:
        q, k, v = (t.unsqueeze(0) for t in (q, k, v))  # → [1,T,D]

    B, T, D = q.shape

    # raw scores before softmax stabilization
    raw_scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)  # [B,T,T]

    # masking
    scores = raw_scores.clone()
    if mask is not None:
        if mask.ndim == 2:                      # [T,T]
            mask = mask.unsqueeze(0)            # [1,T,T]
        elif mask.ndim == 3 and mask.shape[0] == B:  # [B,T,T]
            mask = mask.unsqueeze(1)            # [B,1,T,T]
        # If mask is already [1,T,T], do nothing.

        assert mask.shape[-2:] == (T, T), "mask must end with [T,T]"
        assert mask.shape[0] in (1, B), "batch dim must be 1 or B"

        mask = mask.to(device=scores.device, dtype=scores.dtype)
        scores = scores + (mask == 0) * (-1e9)

    # numerically stable softmax
    scores = scores - scores.max(dim=-1, keepdim=True).values
    attn = torch.softmax(scores, dim=-1)
    out = attn @ v

    if original_2d:
        out = out.squeeze(0)
    return out


# ============================================================================
# Helper for tests: returns RAW (unstabilized) scores like handout
# ============================================================================
def raw_scores_and_attn(q, k, mask=None):
    D = q.size(-1)
    raw = (q @ k.transpose(-2, -1)) / math.sqrt(D)

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3 and mask.shape[0] == q.size(0):
            mask = mask.unsqueeze(1)
        raw = raw + (mask == 0) * (-1e9)

    attn = torch.softmax(raw, dim=-1)
    return raw, attn


# ============================================================================
# Test 1 — Raw numeric example (exactly as in handout)
# ============================================================================
def _test_numeric_example():
    Q = torch.tensor([[1.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 1.0]])
    K = torch.tensor([[1.0, 0.0],
                      [1.0, 1.0],
                      [0.0, 1.0]])
    V = torch.tensor([[1.0, 0.0],
                      [0.0, 2.0],
                      [3.0, 1.0]])

    expected_raw = torch.tensor([
        [0.70710678, 0.70710678, 0.00000000],
        [0.00000000, 0.70710678, 0.70710678],
        [0.70710678, 1.41421356, 0.70710678],
    ])

    expected_attn = torch.tensor([
        [0.40111209, 0.40111209, 0.19777581],
        [0.19777581, 0.40111209, 0.40111209],
        [0.24825508, 0.50348984, 0.24825508],
    ])

    expected_Y = torch.tensor([
        [0.99443954, 1.00000000],
        [1.40111209, 1.20333628],
        [0.99302031, 1.25523477],
    ])

    raw, A = raw_scores_and_attn(Q, K)
    Y = scaled_dot_product_attention(Q, K, V)

    print("Test 1: Raw scores:\n", raw)
    print("Attention A:\n", A)
    print("Output Y:\n", Y)

    assert torch.allclose(raw, expected_raw, atol=1e-6)
    assert torch.allclose(A, expected_attn, atol=1e-6)
    assert torch.allclose(Y, expected_Y, atol=1e-6)
    print("✓ Numeric example matches handout\n")


# ============================================================================
# Test 2 — Causal mask
# ============================================================================
def _test_causal_mask():
    Q = torch.tensor([[1.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 1.0]])
    K = torch.tensor([[1.0, 0.0],
                      [1.0, 1.0],
                      [0.0, 1.0]])
    V = torch.tensor([[1.0, 0.0],
                      [0.0, 2.0],
                      [3.0, 1.0]])

    causal = torch.tril(torch.ones(3, 3))

    _, A_unmasked = raw_scores_and_attn(Q, K)
    _, A_masked = raw_scores_and_attn(Q, K, causal)
    Y_masked = scaled_dot_product_attention(Q, K, V, causal)

    print("Test 2: Causal masking")
    print("A (unmasked):\n", A_unmasked)
    print("A (masked):\n", A_masked)
    print("Y masked:\n", Y_masked)

    expected_masked = torch.tensor([
        [1.0,        0.0,        0.0],
        [0.33023845, 0.66976155, 0.0],
        [0.24825508, 0.50348984, 0.24825508],
    ])

    # A_masked returned as [1,1,3,3], so squeeze:
    A_masked_squeezed = A_masked.squeeze()
    assert torch.allclose(A_masked_squeezed, expected_masked, atol=1e-6)
    print("✓ Causal mask behaves correctly\n")


# ============================================================================
# Test 3 — Shape test
# ============================================================================
def _test_shapes():
    B, T, D = 2, 4, 3
    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    v = torch.randn(B, T, D)

    mask = torch.ones(1, T, T)  # [1,T,T] → should remain [1,T,T]

    y = scaled_dot_product_attention(q, k, v, mask)
    assert y.shape == (B, T, D)
    print("✓ Shape test passed:", y.shape, "\n")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    _test_numeric_example()
    _test_causal_mask()
    _test_shapes()
    print("All Step 1 tests passed.")
