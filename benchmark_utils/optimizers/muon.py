# flake8: noqa
# Muon optimizer implementation
import torch


def zeropower_via_newtonschulz5(G, steps=5):
    """Newton-Schulz iteration to compute the zeroth power/orthogonalize G."""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon (MomentUm Orthogonalized) optimizer.

    Applies Newton-Schulz orthogonalization to gradients of matrix
    parameters, combined with Nesterov momentum. Uses a separate momentum
    buffer (unlike ScionLight which stores momentum in p.grad).

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate (default: 3.6e-4).
        momentum: Nesterov momentum factor (default: 0.95).
        nesterov: Whether to use Nesterov momentum (default: True).
        ns_steps: Number of Newton-Schulz iteration steps (default: 5).
    """

    def __init__(
            self, params, lr=3.6e-4, momentum=0.95, weight_decay=0.0,
            nesterov=True, ns_steps=5,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            nesterov=nesterov, ns_steps=ns_steps,
        )
        super().__init__(params, defaults)
        # Compile Newton-Schulz at init time (class-level to avoid
        # recompilation across param groups).
        self._newton_schulz = torch.compile(zeropower_via_newtonschulz5)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            mu = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue

                # 1) Nesterov momentum on the *raw* gradient.
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)
                g = g.add(buf, alpha=mu) if nesterov else buf

                # 2) Orthogonalize the momentum-mixed gradient.
                g = self._newton_schulz(g, steps=ns_steps)
                # scale so update.square().mean() == 1
                scale = max(g.size(0), g.size(1)) ** 0.5

                if wd > 0:
                    p.add_(p, alpha=-wd * lr)
                p.add_(g, alpha=-lr * scale)
