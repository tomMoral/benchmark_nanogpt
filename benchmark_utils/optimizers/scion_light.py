# Scion optimizer implementation
import torch


def zeropower_via_newtonschulz5(G, steps=5):
    """Newton-Schulz iteration to compute the 0-th power/orthogonalize G.
    """
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


class Norm(object):
    def lmo(self, g):
        raise NotImplementedError


class Spectral(Norm):
    newton_schultz5 = None

    def __init__(self, steps=5):
        self.steps = steps
        # need to compile only at runtime to avoid issues with distributed
        # training setup. Using a class attribute to avoid recompiling for
        # each instantiation.
        if self.newton_schultz5 is None:
            self.newton_schultz5 = torch.compile(zeropower_via_newtonschulz5)

    def lmo(self, g):
        g = self.newton_schultz5(g.reshape(len(g), -1), steps=self.steps).view(
            g.shape
        )
        d_out, d_in = g.shape
        g *= (d_out / d_in) ** 0.5
        return g


class Sign(Norm):
    def __init__(self, zero_init=False):
        self.zero_init = zero_init

    def lmo(self, g):
        _, d_in = g.shape
        return (1 / d_in) * torch.sign(g)


norm_dict = {"Spectral": Spectral, "Sign": Sign}


class ScionLight(torch.optim.Optimizer):
    """Memory-efficient variant of the Scion optimizer from
    https://github.com/LIONS-EPFL/scion/blob/main/examples/modded-nanogpt/train_gpt_scionlight.py

    This implementation saves memory by storing only the averaged gradient
    instead of both the gradient and its average. Note that gradients should
    not be zeroed since p.grad is used directly to store the gradient average.

    Args:
        params:
            Iterable of parameters to optimize or dicts defining parameter
            groups
        lr: float, optional (default: 1e-3)
            Learning rate for the optimizer
        momentum: float, optional (default: 1.0)
            One minus the traditional momentum factor.
            For example, a traditional momentum of 0.9 would be specified as
            momentum=0.1 here
        norm: str, optional (default: 'Auto')
            Choice of norm for gradient projection in
            ('Auto', 'Spectral', or 'Sign')
        norm_kwargs: dict, optional (default: None)
            Additional arguments for the norm projection
        scale: float, optional (default: 1.0)
            Scale factor for updates
        unconstrained: bool, optional (default: False)
            Whether to use unconstrained updates

    Example:
        >>> radius = 50.0
        >>> optim_groups = [{
        ...     'params': model.transformer.h.parameters(),
        ...     'norm': 'Spectral',
        ...     'norm_kwargs': {},
        ...     'scale': radius,
        ... }, {
        ...     'params': model.lm_head.parameters(),
        ...     'norm': 'Sign',
        ...     'norm_kwargs': {},
        ...     'scale': radius*60.0,
        ... }]
        >>> optimizer = ScionLight(optim_groups, lr=2**-12, momentum=0.1)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=1.0,
        norm: str = "Auto",
        norm_kwargs: dict = None,
        scale=1.0,
        unconstrained=False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if norm_kwargs is None:
            norm_kwargs = {}
        defaults = dict(
            lr=lr,
            momentum=momentum,
            scale=scale,
            unconstrained=unconstrained,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            scale = group["scale"]
            unconstrained = group["unconstrained"]
            norm_backend = norm_dict[group["norm"]](**group["norm_kwargs"])
            for p in group["params"]:
                G = p.grad
                if G is None:
                    continue

                update = scale * norm_backend.lmo(G)
                if not unconstrained:
                    p.data.mul_(1 - lr)
                p.data.add_(update, alpha=-lr)

                if momentum != 1:
                    G.mul_(1 - momentum)
