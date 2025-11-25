"""
Step 3: Transformer Block (pre-LN) with MHA + FFN + residuals.

Grounding:
- Handout Sections 6–7: pre-layernorm, residual paths, feedforward.
- Reference: llmcode code/ch08_transformer.py::TransformerBlock.
- Uses Step-2 MultiHeadAttention and Step-1 scaled dot-product attention.
- Coach guide: deterministic seed, explicit shapes, micro-tests.

Note:
Mask semantics (causal/padding, broadcasting) are fully tested in:
  - scaled_dot_product_attention.py (Step-1)
  - multihead_attention.py (Step-2)
Here we focus on block-level behavior: shapes + residual identity sanity.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from multihead_attention import MultiHeadAttention

# Deterministic seed for repeatable tests
torch.manual_seed(0)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: MHA + FFN with residual connections."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:   [B, T, d_model]
            mask (optional):
                 [T, T], [1, T, T], [B, 1, T, T], or [B, T, T]
                 with 1 = keep, 0 = mask (MHA handles normalization).
        Returns:
            [B, T, d_model]
        """
        assert x.ndim == 3, "x must be [B,T,D]"
        B, T, D = x.shape

        # --- MHA path (pre-LN + residual) ---
        x_norm = self.ln1(x)                 # [B,T,D]
        attn_out = self.mha(x_norm, mask)    # [B,T,D]
        x = x + attn_out                     # residual 1

        # --- FFN path (pre-LN + residual) ---
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)           # [B,T,D]
        x = x + ffn_out                      # residual 2

        return x


# -------------------------- Tests -------------------------- #

def _test_shape_round_trip() -> None:
    """Block preserves [B,T,D] shape."""
    x = torch.randn(2, 4, 8)
    block = TransformerBlock(d_model=8, num_heads=2, d_ff=16)
    y = block(x)
    assert y.shape == (2, 4, 8)
    print("Test A (shape round-trip) passed:", y.shape)


def _test_identity_sanity() -> None:
    """
    If all projections are zeroed, FFN≈0 and MHA≈0,
    so the block should behave like the identity (residual dominates).
    """
    x = torch.randn(2, 3, 4)
    block = TransformerBlock(d_model=4, num_heads=1, d_ff=8)

    with torch.no_grad():
        # Zero all MHA linear weights
        for lin in (block.mha.q_proj, block.mha.k_proj, block.mha.v_proj, block.mha.o_proj):
            lin.weight.zero_()
            if lin.bias is not None:
                lin.bias.zero_()
        # Zero all FFN linear weights/biases
        for layer in block.ffn:
            if isinstance(layer, nn.Linear):
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()

    y = block(x)
    assert torch.allclose(y, x, atol=1e-6)
    print("Test B (identity residual) passed")


if __name__ == "__main__":
    _test_shape_round_trip()
    _test_identity_sanity()
    print("\nAll Step 3 tests passed.")
