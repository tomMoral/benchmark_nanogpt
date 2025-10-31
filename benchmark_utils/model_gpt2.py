"""
Reference code for GPT-2 training and inference.
Will save the model weights into files, to be read from C as initialization.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
# from torch.distributed.optim import ZeroRedundancyOptimizer


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class NewGELU(nn.Module):
    """Versions of GeLU used by OpenAI"""
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi)
            * (input + 0.044715 * torch.pow(input, 3.0))
        ))


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        exponent = torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = (1 / max_seq_len) ** exponent
        angular_freq = torch.cat(
            [angular_freq, angular_freq.new_zeros(dim//4)]
        )
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)

        # save the embeddings in buffers so they are moved to the GPU
        # automatically but don't save them in the model state_dict
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos = self.cos[None, :x_BTHD.size(-3), None, :]
        sin = self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.d_embd = config.n_embd // config.n_head
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=False
        )
        self.rotary = Rotary(self.d_embd, config.block_size)
        self.norm = nn.RMSNorm((self.d_embd,))

        # output projection
        self.c_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=False
        )
        self.c_proj.INIT_ZERO = 1

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, n_embd
        # calculate query, key, values for all heads in batch
        # and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=2)
        # Reshape q, k, v for multi-head attention with shape (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k = self.norm(q), self.norm(k)  # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)  # Rotarty Positional Embedding

        # TODO: use the flex attention from torch.nn.attention.flex_attention?
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = NewGELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.c_proj.INIT_ZERO = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# -----------------------------------------------------------------------------
# The main GPT-2 model

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

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))

        # For the lm_head, we use tied weigths with the transformer.wte
        # See https://paperswithcode.com/method/weight-tying
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.SKIP_INIT = True
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def to(self, **kwargs):
        if 'device' in kwargs:
            self.device = kwargs['device']
        return super().to(**kwargs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, 'INIT_ZERO'):
                # this is a residual projection, initialize to zero
                torch.nn.init.zeros_(module.weight)
            elif not hasattr(module, 'SKIP_INIT'):
                # we want to skip initializing lm_head, which shares parameters
                # with wte initialized below during the Embedding's init
                std = 0.02 / math.sqrt(2 * self.config.n_layer)
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=std, generator=self.init_rng
                )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02, generator=self.init_rng
            )

    def forward(self, idx, targets=None, return_logits=True):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, "
            f"block size is only {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        # token embeddings of shape (b, t, n_embd)
        tok_emb = self.transformer.wte(idx)
        # position embeddings of shape (t, n_embd)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on
            # the very last position (using list [-1] to preserve the time dim)
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return loss, logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t))
        and complete the sequence max_new_tokens times, feeding the predictions
        back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of
        operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must
            # crop it at block_size
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size:]
            )
            # forward the model to get the logits for the index in the sequence
            _, logits = self(idx_cond)
            # pluck the logits at the final step and scale
            # by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
