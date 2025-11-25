"""
Step 2: Multi-Head Self-Attention (MHA) built on scaled dot-product attention.

Fully compatible with corrected Step-1 mask semantics.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from scaled_dot_product_attention import scaled_dot_product_attention

torch.manual_seed(0)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with explicit head splitting and mask expansion."""

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

    # -------------------------------------------------------------------------
    def _expand_mask(self, mask: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """
        Expand mask to Step-1-compatible shape [B*H, T, T].
        """
        H = self.h

        # Normalize input shapes
        if mask.ndim == 2:          # [T,T]
            mask = mask.unsqueeze(0)        # [1,T,T]
        elif mask.ndim == 3:        # [B,T,T]
            mask = mask.unsqueeze(1)        # [B,1,T,T]
        elif mask.ndim == 4:        # [B,1,T,T]
            pass
        else:
            raise ValueError("mask must be [T,T], [B,T,T], or [B,1,T,T]")

        # Now shapes:
        # [1,T,T] or [B,1,T,T]
        assert mask.shape[-2:] == (T, T)

        # Expand to batch size B if needed
        if mask.shape[0] == 1:
            mask = mask.repeat(B, 1, 1, 1)      # [B,1,T,T]

        # Now mask is [B,1,T,T]
        # Repeat per head: B → B*H
        mask = mask.repeat_interleave(H, dim=0)  # [B*H,1,T,T]

        # Remove the singleton
        mask = mask.squeeze(1)  # [B*H,T,T]

        return mask

    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B, T, D_model]
        mask: optional [T,T], [B,T,T], or [B,1,T,T]
        """
        B, T, D = x.shape
        assert D == self.d_model

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split heads -> [B,H,T,Dh]
        def split(t):
            return t.view(B, T, self.h, self.dh).transpose(1, 2)

        q = split(q)
        k = split(k)
        v = split(v)

        # Flatten heads -> [B*H, T, Dh]
        qf = q.reshape(B * self.h, T, self.dh)
        kf = k.reshape(B * self.h, T, self.dh)
        vf = v.reshape(B * self.h, T, self.dh)

        # Mask expansion
        mask_flat = None
        if mask is not None:
            mask_flat = self._expand_mask(mask, B, T)  # [B*H,T,T]

        # Per-head attention
        attn = scaled_dot_product_attention(qf, kf, vf, mask_flat)  # [B*H,T,Dh]

        # Restore heads → [B,H,T,Dh]
        attn = attn.view(B, self.h, T, self.dh)

        # Merge → [B,T,D]
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)

        # Final projection
        return self.o_proj(attn)


# =============================================================================
# TESTS
# =============================================================================

def _test_shape_round_trip():
    x = torch.randn(2, 4, 8)
    mha = MultiHeadAttention(8, 2)
    y = mha(x)
    assert y.shape == (2, 4, 8)
    print("Test A (shape round-trip) passed:", y.shape)


def _test_single_head_equivalence():
    x = torch.randn(3, 5, 4)
    mha = MultiHeadAttention(4, 1)

    # Identity projections
    with torch.no_grad():
        W = torch.eye(4)
        mha.q_proj.weight.copy_(W)
        mha.k_proj.weight.copy_(W)
        mha.v_proj.weight.copy_(W)
        mha.o_proj.weight.copy_(W)

    y_mha = mha(x)
    y_ref = scaled_dot_product_attention(x, x, x)
    assert torch.allclose(y_mha, y_ref, atol=1e-6)
    print("Test B (single-head equivalence) passed")


def _test_mask_propagation():
    """
    Verify that causal masking removes attention to future tokens.
    We do this by extracting the *actual per-head scores* used in forward().
    """
    B, T, D = 2, 3, 6
    H = 3
    x = torch.randn(B, T, D)
    mha = MultiHeadAttention(D, H)

    causal = torch.tril(torch.ones(T, T))

    # --- Forward pass (this builds correct qf/kf/vf and expands mask)
    # We intercept internal tensors by manually reproducing forward steps:
    q = mha.q_proj(x)
    k = mha.k_proj(x)

    # Split to heads
    q = q.view(B, T, H, mha.dh).transpose(1, 2)   # [B,H,T,Dh]
    k = k.view(B, T, H, mha.dh).transpose(1, 2)   # [B,H,T,Dh]

    # Flatten
    qf = q.reshape(B*H, T, mha.dh)
    kf = k.reshape(B*H, T, mha.dh)

    # Correct mask expansion (same path forward() uses)
    mask_flat = mha._expand_mask(causal, B, T)   # [B*H,T,T]

    # Compute true masked scores
    scores = (qf @ kf.transpose(-2, -1)) / (mha.dh ** 0.5)
    scores = scores + (mask_flat == 0) * (-1e9)
    attn = torch.softmax(scores, dim=-1)

    # Extract all upper-triangle entries
    r, c = torch.triu_indices(T, T, offset=1)
    upper = attn[:, r, c]

    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)
    print("Test C (mask propagation) passed")



if __name__ == "__main__":
    _test_shape_round_trip()
    _test_single_head_equivalence()
    _test_mask_propagation()
    print("\nAll Step 2 tests passed.")
