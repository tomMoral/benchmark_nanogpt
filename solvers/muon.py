from contextlib import nullcontext

import torch
from benchmark_utils.distributed_tools import setup_distributed
from benchmark_utils.lr_scheduler import get_lr
from benchopt import BaseSolver
from torch.optim import AdamW
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# Muon optimizer implementation


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
        lr: Learning rate (default: 0.02).
        momentum: Nesterov momentum factor (default: 0.95).
        nesterov: Whether to use Nesterov momentum (default: True).
        ns_steps: Number of Newton-Schulz iteration steps (default: 5).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
        # Compile Newton-Schulz at init time (class-level to avoid
        # recompilation across param groups).
        self._newton_schulz = torch.compile(zeropower_via_newtonschulz5)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue

                # Orthogonalize: reshape to 2D, apply Newton-Schulz, reshape
                # back
                g_orth = self._newton_schulz(
                    g.reshape(g.size(0), -1), steps=ns_steps
                ).view(g.shape)

                # Scale so that the update has unit Frobenius norm per
                # "fan-in" dimension, matching the convention from
                # modded-nanogpt.
                d_out, d_in = g.shape[0], g[0].numel()
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


class Solver(BaseSolver):
    name = "Muon"

    parameters = {
        "muon_lr": [0.02],
        "muon_momentum": [0.95],
        "adam_lr": [3e-4],
        "adam_weight_decay": [0.0],
        "num_steps": [6200],
        "batch_size": [64],
        "slurm_nodes": [1, 2],
    }
    slurm_params = {
        "slurm_gres": "gpu:4",
        "slurm_ntasks_per_node": 4,
    }

    sampling_strategy = "callback"

    def set_objective(self, train_dataloader, model):
        self.dist, self.rank, self.world_size, device = setup_distributed()

        model = model.to(device=device)
        model.device = device
        self.train_dataloader = train_dataloader

        self.ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else nullcontext()
        )

        self.model = torch.compile(model, dynamic=False, fullgraph=True)
        Muon.step = torch.compile(torch.no_grad(Muon.step))
        AdamW.step = torch.compile(torch.no_grad(AdamW.step))

    def __del__(self):
        if getattr(self, "dist", None) is not None:
            self.dist.destroy_process_group()

    def get_next(self, stop_val):
        return stop_val + 250

    def warm_up(self):
        n_iter = self.num_steps
        self.num_steps = 10
        self.run_once(stop_val=10)
        self.num_steps = n_iter

    def run(self, cb):
        # Split parameters into Muon group (internal 2D matrices) and
        # AdamW group (embeddings, lm_head, biases, layernorms).
        muon_params = []
        adam_decay_params = []
        adam_nodecay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Embeddings and lm_head go to AdamW, everything else 2D goes
            # to Muon.
            if (
                param.dim() >= 2
                and "wte" not in name
                and "wpe" not in name
                and "lm_head" not in name
            ):
                muon_params.append(param)
            elif param.dim() >= 2:
                adam_decay_params.append(param)
            else:
                adam_nodecay_params.append(param)

        self.muon_optimizer = Muon(
            [{"params": muon_params}],
            lr=torch.tensor(self.muon_lr),
            momentum=self.muon_momentum,
        )

        self.adam_optimizer = AdamW(
            [
                {"params": adam_decay_params, "weight_decay": self.adam_weight_decay},
                {"params": adam_nodecay_params, "weight_decay": 0.0},
            ],
            lr=torch.tensor(self.adam_lr),
            betas=(0.9, 0.95),
            fused=True,
        )

        train_loader = self.train_dataloader.get_distributed_data_generator(
            batch_size=self.batch_size,
            world_size=self.world_size,
            rank=self.rank,
        )

        if self.dist is not None:
            self.dist.barrier()

        step = 0
        with tqdm(total=self.num_steps, desc="Training") as progress:
            while cb():
                self.model.train()
                self.muon_optimizer.zero_grad(set_to_none=True)
                self.adam_optimizer.zero_grad(set_to_none=True)

                step += 1
                progress.update()
                if step == self.num_steps:
                    break

                data = next(train_loader)
                with self.ctx:
                    loss, *_ = self.model(*data)
                loss.backward()

                if self.dist is not None:
                    for param in self.model.parameters():
                        self.dist.all_reduce(param.grad, op=self.dist.ReduceOp.AVG)

                # Scale learning rates with the schedule
                scale_lr = get_lr(step, self.num_steps)
                for param_group in self.muon_optimizer.param_groups:
                    param_group["lr"] = torch.tensor(self.muon_lr * scale_lr)
                for param_group in self.adam_optimizer.param_groups:
                    param_group["lr"] = torch.tensor(self.adam_lr * scale_lr)

                self.muon_optimizer.step()
                self.adam_optimizer.step()

    def get_result(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return dict(model=self.model, dist=self.dist)
