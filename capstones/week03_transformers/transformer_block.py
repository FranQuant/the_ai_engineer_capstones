"""
Step 3: Transformer Block (pre-LN) with MHA + FFN + residuals + (optional) dropout.

Grounding:
- Handout Sections 6–7: pre-layernorm, residual paths, feedforward, dropout locations.
- Reference: llmcode code/ch08_transformer.py::TransformerBlock.
- Uses Step-2 MultiHeadAttention and Step-1 scaled dot-product attention.
- Coach guide: deterministic seed, explicit shapes, micro-tests.

Note:
Mask semantics (boolean causal/padding mask, broadcastable to [B,H,T,T])
are handled upstream in MiniTransformerLM; here we simply pass `attn_mask` to MHA.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from multihead_attention import MultiHeadAttention

# Deterministic seed for repeatable tests
torch.manual_seed(0)


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer block with:
        - LayerNorm
        - Multi-Head Attention
        - Residual connection
        - Feedforward (FFN)
        - Residual connection
        - Optional dropout (default p=0.0 for deterministic Week-3 behavior)
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()

        # LayerNorms (pre-norm architecture)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Multi-Head Attention block
        self.mha = MultiHeadAttention(d_model, num_heads)

        # Position-wise FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        # Dropout hooks (default no-op when p=0.0)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
            attn_mask: broadcastable boolean mask (True=keep, False=mask);
                       shapes allowed: [1,1,T,T], [B,1,T,T], [1,T,T], [B,T,T]
        Returns:
            [B, T, d_model]
        """
        assert x.ndim == 3, "x must be [B,T,D]"
        B, T, D = x.shape

        # --- Multi-Head Attention path (Pre-LN + residual + dropout) ---
        x_norm = self.ln1(x)                          # [B,T,D]
        attn_out = self.mha(x_norm, attn_mask)        # [B,T,D]
        attn_out = self.attn_dropout(attn_out)        # dropout (p=0.0 by default)
        x = x + attn_out                              # residual 1

        # --- Feedforward path (Pre-LN + residual + dropout) ---
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)                    # [B,T,D]
        ffn_out = self.ffn_dropout(ffn_out)           # dropout (p=0.0 by default)
        x = x + ffn_out                               # residual 2

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
