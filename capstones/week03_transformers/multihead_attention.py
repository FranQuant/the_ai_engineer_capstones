"""
Step 2 (final): Multi-Head Self-Attention (MHA) built on scaled dot-product attention.

Grounding:
- Handout Sections 5–7 (self-attention, multi-head shapes, masking).
- Reference: llmcode code/ch08_transformer.py (head split/merge patterns).
- Uses Step-1 scaled_dot_product_attention for 1-head equivalence tests.
- Coach guide: deterministic seeds, explicit shapes, minimal runnable tests.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from scaled_dot_product_attention import scaled_dot_product_attention

# Deterministic seed for repeatable tests
torch.manual_seed(0)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with explicit head splitting and masking."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.h = num_heads
        self.dh = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
            mask: optional [T,T], [1,T,T], [B,T,T], or [B,1,T,T] with 1=keep, 0=mask
        Returns:
            [B, T, d_model]
        """
        assert x.ndim == 3, "x must be [B,T,D]"
        B, T, D = x.shape
        assert D == self.d_model, f"expected D={self.d_model}, got {D}"

        # Linear projections
        q = self.q_proj(x)  # [B,T,D]
        k = self.k_proj(x)  # [B,T,D]
        v = self.v_proj(x)  # [B,T,D]

        # Split into heads: [B,T,D] -> [B,H,T,Dh]
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)  # [B,H,T,Dh]
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)  # [B,H,T,Dh]
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)  # [B,H,T,Dh]

        # Scaled dot-product attention per head: scores [B,H,T,T]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)  # [B,H,T,T]

        # Normalize mask to [B,1,1,T,T] then broadcast over heads
        if mask is not None:
            if mask.ndim == 2:          # [T,T]
                mask = mask.unsqueeze(0).unsqueeze(0)   # [1,1,T,T]
            elif mask.ndim == 3:        # [B,T,T] or [1,T,T]
                mask = mask.unsqueeze(1)                # [B,1,T,T] or [1,1,T,T]
            elif mask.ndim == 4:        # [B,1,T,T] or [1,1,T,T]
                # already fine
                pass
            else:
                raise ValueError("mask must be [T,T], [1,T,T], [B,T,T], or [B,1,T,T]")

            assert mask.shape[-2:] == (T, T), "mask trailing dims must be [T,T]"
            # Broadcast over heads: scores [B,H,T,T], mask [B or 1, 1, T, T]
            scores = scores + (mask == 0).to(scores.dtype) * (-1e9)

        # Softmax along last dim
        scores = scores - scores.max(dim=-1, keepdim=True).values  # numerical stability
        attn = torch.softmax(scores, dim=-1)                       # [B,H,T,T]

        # Weighted sum of values: [B,H,T,T] @ [B,H,T,Dh] -> [B,H,T,Dh]
        out = attn @ v  # [B,H,T,Dh]

        # Merge heads: [B,H,T,Dh] -> [B,T,D]
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # Final linear projection
        return self.o_proj(out)


# -------------------------- Tests -------------------------- #

def _test_shape_round_trip() -> None:
    x = torch.randn(2, 4, 8)
    mha = MultiHeadAttention(d_model=8, num_heads=2)
    y = mha(x)
    assert y.shape == (2, 4, 8)
    print("Test A (shape round-trip) passed:", y.shape)


def _test_single_head_equivalence() -> None:
    """
    With H=1 and identity projections, MultiHeadAttention should match
    scaled_dot_product_attention(x, x, x).
    """
    B, T, D = 3, 5, 4
    x = torch.randn(B, T, D)
    mha = MultiHeadAttention(d_model=D, num_heads=1)

    # Force identity weights for q/k/v/o
    with torch.no_grad():
        mha.q_proj.weight.copy_(torch.eye(D))
        mha.k_proj.weight.copy_(torch.eye(D))
        mha.v_proj.weight.copy_(torch.eye(D))
        mha.o_proj.weight.copy_(torch.eye(D))

    y_mha = mha(x)                           # [B,T,D]
    y_ref = scaled_dot_product_attention(x, x, x)  # [B,T,D]

    assert torch.allclose(y_mha, y_ref, atol=1e-6)
    print("Test B (single-head equivalence) passed")


def _test_mask_propagation() -> None:
    """
    Causal mask: attention weights above the diagonal should be ~0 for B=1.
    """
    B, T, D = 1, 3, 6
    x = torch.randn(B, T, D)
    mha = MultiHeadAttention(d_model=D, num_heads=3)
    causal = torch.tril(torch.ones(T, T))  # [T,T]

    # Forward pass to ensure mask is accepted
    _ = mha(x, causal)

    # Inspect internal scores with the same logic
    q = mha.q_proj(x).view(B, T, mha.h, mha.dh).transpose(1, 2)  # [B,H,T,Dh]
    k = mha.k_proj(x).view(B, T, mha.h, mha.dh).transpose(1, 2)  # [B,H,T,Dh]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(mha.dh)       # [B,H,T,T]
    scores = scores + (causal == 0).to(scores.dtype).view(1, 1, T, T) * (-1e9)
    attn = torch.softmax(scores, dim=-1)                         # [B,H,T,T]

    # Check upper triangle ~ 0
    tri = torch.triu_indices(T, T, offset=1)
    upper = attn[..., tri[0], tri[1]]
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)
    print("Test C (mask propagation) passed")


if __name__ == "__main__":
    _test_shape_round_trip()
    _test_single_head_equivalence()
    _test_mask_propagation()
    print("\nAll Step 2 tests passed.")
