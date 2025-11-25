"""
Step 4: Tiny decoder-only Transformer LM (GPT-style) with learned positions.

Grounding:
- Handout Sections 6–8: transformer blocks, decoder-only LM.
- Reference: llmcode code/ch08_transformer.py and code/ch09_gpt.py.
- Uses Step-3 TransformerBlock (pre-LN, residuals).
- Coach guide: deterministic seeds, explicit shapes, runnable micro-tests.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from transformer_block import TransformerBlock

# Deterministic seed for repeatable tests
torch.manual_seed(0)


class MiniTransformerLM(nn.Module):
    """Decoder-only transformer language model with learned positional embeddings."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embed.weight

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            input_ids: LongTensor [B, T]
            attn_mask: optional [T,T], [1,T,T], [B,T,T], or [B,1,T,T] (1=keep, 0=mask)
        Returns:
            logits: [B, T, vocab_size]
        """
        assert input_ids.ndim == 2, "input_ids must be [B,T]"
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        device = input_ids.device

        positions = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
        x = self.token_embed(input_ids) + self.pos_embed(positions)  # [B, T, D]

        for block in self.blocks:
            x = block(x, attn_mask)

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
