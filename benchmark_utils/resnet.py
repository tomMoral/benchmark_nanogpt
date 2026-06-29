"""ResNet-18 for CIFAR, wrapped to match the benchmark model contract.

The benchmark expects a model that:
- returns ``(loss, logits)`` from ``forward(images, targets)`` (loss first),
- exposes ``optim_param_groups()`` -> {"matrix", "embed_head", "scalar"} so
  solvers stay architecture-agnostic,
- records its device in ``to(device=...)`` and keeps an ``init_func`` hook.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet18


class ResNet(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        net = resnet18(num_classes=num_classes)
        # CIFAR stem: 3x3 stride-1 conv and no initial max-pool, since the
        # 32x32 inputs are too small for the 7x7/stride-2 ImageNet stem.
        net.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        net.maxpool = nn.Identity()
        self.net = net

        # Hook kept for compatibility with solvers that override init_func
        # (e.g. the sin_init experiment); torchvision already inits at build.
        self.init_func = None
        self.device = None

    def to(self, **kwargs):
        if "device" in kwargs:
            self.device = kwargs["device"]
        return super().to(**kwargs)

    def initialize_weights(self, seed=42):
        # torchvision applies sensible init at construction. Keep the hook so
        # solvers that set init_func don't crash; only Linear is re-inited.
        if self.init_func is None:
            return
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(seed)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                self.init_func(
                    module.weight, mean=0.0, std=0.02, generator=self.init_rng
                )
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, images, targets=None):
        logits = self.net(images)
        if targets is None:
            return None, logits
        if self.training:
            # Differentiable loss for optimization.
            loss = F.cross_entropy(logits, targets)
        else:
            # In eval the reported metric is the top-1 error, so the objective
            # can stay metric-agnostic (it just averages this per batch).
            loss = (logits.argmax(dim=-1) != targets).float().mean()
        return loss, logits

    def optim_param_groups(self):
        """Structural parameter groups, so solvers stay architecture-agnostic.

        - "matrix": conv (4D) and classifier (2D) weights (Muon/Scion-friendly;
          AdamW with decay). ResNet has no embedding, so "embed_head" is empty.
        - "scalar": 1D params such as norm weights/biases (AdamW, no decay).
        """
        groups = {"matrix": [], "embed_head": [], "scalar": []}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() < 2:
                groups["scalar"].append(p)
            else:
                groups["matrix"].append(p)
        return groups
