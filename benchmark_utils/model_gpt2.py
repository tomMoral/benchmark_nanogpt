"""
GPT-2 model matching modded-nanogpt commit
844e5fdb2334ff83324e6f1f900ce443dd9e1226.

Reference architecture:
- RoPE (rotary positional embeddings) applied to Q/K in attention
- RMSNorm (parameter-free) before attention and MLP, and at the output
- Attention output scaled by 1/sqrt(2 * n_layer) before residual add
- No learnable positional embeddings (no wpe)
- No biases anywhere; weight tying between wte and lm_head
- MLP: c_fc -> gelu -> c_proj
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# -----------------------------------------------------------------------------
# Rotary positional embeddings


class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return (
            self.cos_cached[None, :, None, :],
            self.sin_cached[None, :, None, :],
        )


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


# -----------------------------------------------------------------------------
# Transformer blocks


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config.n_embd)
        self.attn_scale = 1.0 / (2 * config.n_layer) ** 0.5

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x


# -----------------------------------------------------------------------------
# Main model


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying.
        self.transformer.wte.weight = self.lm_head.weight

        # Hook kept for compatibility with solvers that override init_func
        # (e.g. sinusoidal init for the AdamW solver).
        self.init_func = None
        self.initialize_weights()

    def initialize_weights(self, seed=42):
        # The reference uses default PyTorch initialization. We keep a hook so
        # solvers can override init_func (used by sin_init experiments).
        if self.init_func is None:
            return
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(seed)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            self.init_func(
                module.weight, mean=0.0, std=0.02, generator=self.init_rng
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            self.init_func(
                module.weight, mean=0.0, std=0.02, generator=self.init_rng
            )

    def to(self, **kwargs):
        if "device" in kwargs:
            self.device = kwargs["device"]
        return super().to(**kwargs)

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, "
            f"block size is only {self.config.block_size}"
        )
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            logits = self.lm_head(x).float()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(x[:, [-1], :]).float()
            loss = None

        return loss, logits

    def optim_param_groups(self):
        """Structural parameter groups, so solvers stay architecture-agnostic.

        - "matrix": 2D body weights (Muon/Scion-friendly; AdamW with decay).
        - "embed_head": token embedding / output head (AdamW, no decay).
        - "scalar": 1D params such as norms and biases (AdamW, no decay).
        """
        groups = {"matrix": [], "embed_head": [], "scalar": []}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() < 2:
                groups["scalar"].append(p)
            elif any(k in name for k in ("wte", "wpe", "lm_head")):
                groups["embed_head"].append(p)
            else:
                groups["matrix"].append(p)
        return groups
