from contextlib import nullcontext

import torch
from benchmark_utils.distributed_tools import (
    setup_distributed, wrap_ddp, unwrap_model
)
from benchmark_utils.lr_scheduler import get_lr
from benchmark_utils.optimizers.muon import Muon
from benchopt import BaseSolver
from torch.optim import AdamW
from tqdm.auto import tqdm


class Solver(BaseSolver):
    # DDP variant of the Muon solver, to test the speedup from overlapping
    # the gradient all-reduce with the backward pass. Differences from `Muon`:
    #   - the model is wrapped in DDP (which also broadcasts rank-0 weights at
    #     construction, so broadcast_model is not needed);
    #   - no manual all-reduce loop (DDP syncs gradients during backward);
    #   - torch.compile drops fullgraph (DDP inserts graph breaks to overlap);
    #   - get_result returns the unwrapped module, so the shared distributed
    #     eval never goes through the DDP wrapper.
    name = "Muon-DDP"

    parameters = {
        "muon_lr": [3.6e-4],
        "muon_momentum": [0.95],
        "adam_lr": [3.6e-3],
        "num_steps": [6200],
        "batch_size": [64],
        "slurm_nodes": [2],
    }

    sampling_strategy = "callback"

    def set_objective(self, train_dataloader, model):
        self.dist, self.rank, self.world_size, device = setup_distributed()

        model = model.to(device=device)
        model.device = device
        self.train_dataloader = train_dataloader
        self.train_loss = None

        self.ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else nullcontext()
        )

        model = torch.compile(model, dynamic=False, fullgraph=True)
        self.model = wrap_ddp(self.dist, model, device)

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
        # Muon on the 2D body matrices; AdamW on the embedding/head and any
        # 1D params. Build groups from the unwrapped module.
        groups = unwrap_model(self.model).optim_param_groups()

        self.muon_optimizer = Muon(
            groups["matrix"],
            lr=torch.tensor(self.muon_lr),
            momentum=self.muon_momentum,
        )

        # Embedding/head and 1D params are never weight-decayed.
        self.adam_optimizer = AdamW(
            groups["embed_head"] + groups["scalar"],
            lr=torch.tensor(self.adam_lr),
            betas=(0.9, 0.95),
            weight_decay=0.0,
            fused=True,
        )

        train_loader = self.train_dataloader.get_distributed_data_generator(
            batch_size=self.batch_size,
            world_size=self.world_size,
            rank=self.rank,
        )

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
                # DDP averages the gradients across ranks during backward.
                loss.backward()

                # Track a smoothed train loss (on-device, no per-step sync).
                ema = loss.detach()
                self.train_loss = (
                    ema if self.train_loss is None
                    else 0.9 * self.train_loss + 0.1 * ema
                )

                # Scale learning rates with the schedule (see Muon solver).
                scale_lr = get_lr(step, self.num_steps, cooldown_frac=0.29)
                for param_group in self.muon_optimizer.param_groups:
                    param_group["lr"] = torch.tensor(self.muon_lr * scale_lr)
                for param_group in self.adam_optimizer.param_groups:
                    param_group["lr"] = torch.tensor(self.adam_lr * scale_lr)

                self.muon_optimizer.step()
                self.adam_optimizer.step()

    def get_result(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_loss = (
            float(self.train_loss) if self.train_loss is not None else None
        )
        return dict(
            model=unwrap_model(self.model),
            dist=self.dist,
            train_loss=train_loss,
        )
