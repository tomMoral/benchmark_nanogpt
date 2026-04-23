import torch

from benchmark_utils.optimizers.orthogonalization import (
    zeropower_via_polar_express,
)

class Muon(torch.optim.Optimizer):
    """Muon (MomentUm Orthogonalized) optimizer.

    Applies Polar Express orthogonalization to gradients of matrix
    parameters, combined with Nesterov momentum. Uses a separate momentum
    buffer (unlike ScionLight which stores momentum in p.grad).

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate (default: 0.02).
        momentum: Nesterov momentum factor (default: 0.95).
        nesterov: Whether to use Nesterov momentum (default: True).
    """

    _compiled_orthogonalize = None

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        if Muon._compiled_orthogonalize is None:
            Muon._compiled_orthogonalize = torch.compile(
                zeropower_via_polar_express
            )

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue

                g_matrix = g.reshape(g.size(0), -1)
                g_orth = Muon._compiled_orthogonalize(g_matrix).view(g.shape)

                # Scale so that the update has unit Frobenius norm per
                # "fan-in" dimension, matching the convention from
                # modded-nanogpt.
                d_out, d_in = g_matrix.shape
                g_orth = g_orth * max(1, (d_out / d_in) ** 0.5)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g_orth)

                if nesterov:
                    update = g_orth + mu * buf
                else:
                    update = buf

                p.add_(update, alpha=-lr)
