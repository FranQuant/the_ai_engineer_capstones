"""
Step 4: Tiny decoder-only Transformer LM (GPT-style) with sinusoidal positions.

Grounding:
- Handout Sections 6–8: transformer blocks, decoder-only LM.
- Reference: llmcode code/ch08_transformer.py and code/ch09_gpt.py.
- Uses Step-3 TransformerBlock (pre-LN, residuals).
- Coach guide: deterministic seeds, explicit shapes, runnable micro-tests.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from transformer_block import TransformerBlock

# Deterministic seed for repeatable tests
torch.manual_seed(0)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encodings as in the Attention & Tiny Transformers handout.

    Shapes:
        - buffer pe: (1, max_len, d_model)
        - forward(x): x is (B, T, d_model) -> (B, T, d_model)
    """

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T_max, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (T_max, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T_max, d_model)
        # Not a parameter; moves with .to(device) but is not trained
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, T, d_model]
        Returns:
            x + positional encodings of shape [B, T, d_model]
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


class MiniTransformerLM(nn.Module):
    """Decoder-only transformer language model with sinusoidal positional encodings."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_seq_len: int,
        dropout: float = 0.0,  # hooks for later experiments; default 0.0 for capstone
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Sinusoidal positional encodings (fixed, non-trainable)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)

        # Optional embedding dropout (default p=0.0, i.e. no effect)
        self.embed_dropout = nn.Dropout(dropout)

        # Stack of Transformer blocks (pre-LN, residuals)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout=dropout) for _ in range(num_layers)]
        )

        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: lm_head shares weights with token embeddings
        self.lm_head.weight = self.token_embed.weight

    def _build_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Build a default causal mask of shape [1, 1, T, T] with boolean dtype:
            True  = allowed
            False = masked out
        This broadcasts correctly across (B, H, T, T) in attention.
        """
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        return causal.view(1, 1, T, T)


    def _normalize_attn_mask(
        self, attn_mask: torch.Tensor, B: int, T: int, device: torch.device
    ) -> torch.Tensor:
        """
        Normalize various attn_mask input shapes/dtypes to a broadcastable bool mask.

        Accepted forms (as in the original file / tests):
            - [T, T]
            - [1, T, T]
            - [B, T, T]
            - [B, 1, T, T]

        Semantics:
            - non-zero / True  => keep
            - zero / False     => mask out
        """
        # Ensure mask is on the correct device
        attn_mask = attn_mask.to(device)

        if attn_mask.ndim == 2:
            # [T, T] -> [1, 1, T, T]
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        elif attn_mask.ndim == 3:
            # Could be [1, T, T] or [B, T, T]; add a "head" dimension
            attn_mask = attn_mask.unsqueeze(1)  # -> [*, 1, T, T]
        elif attn_mask.ndim == 4:
            # [B, 1, T, T] or [1, 1, T, T] already fine
            pass
        else:
            raise ValueError(f"Unsupported attn_mask ndim={attn_mask.ndim}")

        # Convert to boolean "keep" mask
        if attn_mask.dtype == torch.bool:
            keep = attn_mask
        else:
            keep = attn_mask != 0

        return keep

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            input_ids: LongTensor [B, T]
            attn_mask: optional mask
                Allowed shapes:
                    [T, T], [1, T, T], [B, T, T], or [B, 1, T, T]
                Semantics:
                    non-zero / True => keep
                    zero / False    => mask out
        Returns:
            logits: [B, T, vocab_size]
        """
        assert input_ids.ndim == 2, "input_ids must be [B,T]"
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        device = input_ids.device

        # Token embeddings: [B, T, D]
        x = self.token_embed(input_ids)

        # Add sinusoidal positional encodings
        x = self.pos_encoding(x)

        # Optional embedding dropout (p=0.0 by default => no effect)
        x = self.embed_dropout(x)

        # Build or normalize attention mask
        if attn_mask is None:
            mask = self._build_causal_mask(T, device)
        else:
            mask = self._normalize_attn_mask(attn_mask, B, T, device)

        # Pass through transformer blocks with mask
        for block in self.blocks:
            x = block(x, attn_mask=mask)

        x = self.final_ln(x)
        logits = self.lm_head(x)  # [B, T, V]
        return logits


# -------------------------- Tests -------------------------- #

def _config():
    return dict(
        vocab_size=17,
        d_model=16,
        num_heads=4,
        d_ff=32,
        num_layers=2,
        max_seq_len=8,
    )


def _test_shapes() -> None:
    cfg = _config()
    model = MiniTransformerLM(**cfg)
    input_ids = torch.randint(0, cfg["vocab_size"], (2, 5))
    logits = model(input_ids)
    assert logits.shape == (2, 5, cfg["vocab_size"])
    print("Test 1 (shape) passed:", logits.shape)


def _test_determinism() -> None:
    cfg = _config()
    torch.manual_seed(0)
    model1 = MiniTransformerLM(**cfg)
    input_ids = torch.randint(0, cfg["vocab_size"], (2, 5))
    logits1 = model1(input_ids)

    torch.manual_seed(0)
    model2 = MiniTransformerLM(**cfg)
    logits2 = model2(input_ids)

    assert torch.allclose(logits1, logits2, atol=1e-6)
    print("Test 2 (determinism) passed")


def _test_weight_tying() -> None:
    cfg = _config()
    model = MiniTransformerLM(**cfg)
    assert model.lm_head.weight.data_ptr() == model.token_embed.weight.data_ptr()
    assert model.token_embed.weight.shape == (cfg["vocab_size"], cfg["d_model"])
    assert model.lm_head.weight.shape == (cfg["vocab_size"], cfg["d_model"])
    print("Test 3 (weight tying) passed")


def _test_mask_smoke() -> None:
    cfg = _config()
    model = MiniTransformerLM(**cfg)
    input_ids = torch.randint(0, cfg["vocab_size"], (1, 5))
    # Original test: 2D mask [T, T] with 0/1 values
    causal = torch.tril(torch.ones(5, 5))
    logits = model(input_ids, attn_mask=causal)
    assert logits.shape == (1, 5, cfg["vocab_size"])
    print("Test 4 (mask smoke) passed:", logits.shape)


if __name__ == "__main__":
    _test_shapes()
    _test_determinism()
    _test_weight_tying()
    _test_mask_smoke()
    print("\nAll Step 4 tests passed.")
