import torch
import torch.optim as optim

from benchmark_utils.optimizers.shampoo_preconditioner import (
    ShampooPreconditioner,
)


class Shampoo(optim.Optimizer):
    """Classic Shampoo with per-parameter momentum and decoupled weight decay."""

    def __init__(
        self,
        params,
        lr=1e-2,
        momentum=0.9,
        shampoo_beta=0.95,
        eps=1e-8,
        weight_decay=0.0,
        precondition_frequency=10,
        max_precond_dim=10000,
        merge_dims=False,
        precondition_1d=False,
        data_format="channels_first",
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            merge_dims=merge_dims,
            precondition_1d=precondition_1d,
        )
        super().__init__(params, defaults)
        self.preconditioner = ShampooPreconditioner(data_format=data_format)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            loss = None
        else:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                if "Q" not in state:
                    self.preconditioner.init_preconditioner(
                        grad,
                        state,
                        precondition_frequency=group["precondition_frequency"],
                        shampoo_beta=group["shampoo_beta"],
                        max_precond_dim=group["max_precond_dim"],
                        precondition_1d=group["precondition_1d"],
                        merge_dims=group["merge_dims"],
                    )

                state["step"] += 1
                self.preconditioner.update_preconditioner(
                    grad,
                    state,
                    max_precond_dim=group["max_precond_dim"],
                    merge_dims=group["merge_dims"],
                    precondition_1d=group["precondition_1d"],
                )

                preconditioned_grad = self.preconditioner.precondition(
                    grad,
                    state,
                    eps=group["eps"],
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )

                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(group["momentum"]).add_(preconditioned_grad)
                p.add_(momentum_buffer, alpha=-group["lr"])

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss
