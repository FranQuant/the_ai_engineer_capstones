from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mini_transformer import MiniTransformerLM

torch.manual_seed(0)
random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------- Tokenizer and Dataset ---------------------- #

text = """
Attention is all you need.
Transformers use self-attention to model dependencies.
This is a tiny training corpus for the Week 03 capstone.
"""

chars = sorted(list(set(text)))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}


def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]


def decode(ids: list[int]) -> str:
    return "".join([itos[i] for i in ids])


ids = torch.tensor(encode(text), dtype=torch.long)


class CharDataset(Dataset):
    def __init__(self, data_ids: torch.Tensor, block_size: int):
        self.data = data_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


# Train/val split
split = int(0.9 * len(ids))
train_ids = ids[:split]
val_ids = ids[split:]


# ---------------------- Mask Builder ---------------------- #

def build_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    mask = torch.tril(torch.ones(T, T, device=device))
    return mask


# ---------------------- Config ---------------------- #

@dataclass
class TrainConfig:
    d_model: int = 128
    num_heads: int = 4
    d_ff: int = 256
    num_layers: int = 2
    max_seq_len: int = 128
    block_size: int = 64
    batch_size: int = 32
    max_iters: int = 500
    eval_interval: int = 50
    eval_iters: int = 20
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_lr_steps: int = 500
    beta1: float = 0.9
    beta2: float = 0.95


def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_lr_steps - cfg.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr * cosine


# ---------------------- Loss Estimation ---------------------- #

@torch.no_grad()
def estimate_loss(model: MiniTransformerLM, train_loader, val_loader, cfg: TrainConfig, device: torch.device, vocab_size: int):
    model.eval()
    train_losses = []
    val_losses = []
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    for _ in range(cfg.eval_iters):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        mask = build_causal_mask(x.size(1), device)
        logits = model(x, attn_mask=mask)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        train_losses.append(loss.item())

        try:
            x, y = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            x, y = next(val_iter)
        x, y = x.to(device), y.to(device)
        mask = build_causal_mask(x.size(1), device)
        logits = model(x, attn_mask=mask)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        val_losses.append(loss.item())

    model.train()
    return sum(train_losses) / len(train_losses), sum(val_losses) / len(val_losses)


# ---------------------- Sampling ---------------------- #

@torch.no_grad()
def generate(model: MiniTransformerLM, idx: torch.Tensor, max_new_tokens: int, cfg: TrainConfig, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg.max_seq_len :]
        T = idx_cond.size(1)
        mask = build_causal_mask(T, idx.device)
        logits = model(idx_cond, attn_mask=mask)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, ix = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


# ---------------------- Main Training Loop ---------------------- #


def main() -> None:
    cfg = TrainConfig()

    # Ensure block size fits both splits
    effective_block = min(cfg.block_size, len(train_ids) - 1, max(2, len(val_ids) - 1))

    train_ds = CharDataset(train_ids, effective_block)
    val_ds = CharDataset(val_ids, effective_block)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    vocab_size = len(chars)
    model = MiniTransformerLM(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        num_layers=cfg.num_layers,
        max_seq_len=cfg.max_seq_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)

    global_step = 0
    t0 = time.time()
    train_iter = iter(train_loader)
    for step in range(cfg.max_iters):
        model.train()
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        lr_now = get_lr(global_step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        mask = build_causal_mask(x.size(1), device)
        logits = model(x, attn_mask=mask)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % cfg.eval_interval == 0:
            train_loss, val_loss = estimate_loss(model, train_loader, val_loader, cfg, device, vocab_size)
            elapsed = time.time() - t0
            print(f"step {step:4d} | train {train_loss:.4f} | val {val_loss:.4f} | lr {lr_now:.6f} | time {elapsed:.1f}s")

        global_step += 1

    # Save checkpoint
    torch.save(model.state_dict(), "mini_gpt.pt")
    print("Saved checkpoint to mini_gpt.pt")

    # Sampling
    start = torch.tensor([[random.randint(0, vocab_size - 1)]], device=device)
    out = generate(model, start, max_new_tokens=100, cfg=cfg, temperature=0.8, top_k=8)
    print("\nSampled text:\n")
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
