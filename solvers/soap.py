from benchopt import BaseSolver

from contextlib import nullcontext

from tqdm.auto import tqdm

import torch

from benchmark_utils.optimizers.soap import SOAP
from benchmark_utils.lr_scheduler import get_lr
from benchmark_utils.torch_utils import compile_step
from benchmark_utils.distributed_tools import (
    setup_distributed, broadcast_model
)


class Solver(BaseSolver):
    name = "SOAP"

    parameters = {
        "learning_rate": [2e-3],
        "weight_decay": [1e-4],
        "num_steps": [6200],
        "batch_size": [64],
        "cooldown_frac": [0.3],
        "slurm_nodes": [2],
    }
    slurm_params = {
        "slurm_gres": "gpu:4",
        "slurm_ntasks_per_node": 4,
    }

    sampling_strategy = "callback"

    def set_objective(self, train_dataloader, model):
        # Setup distributed training if needed
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

        self.model = torch.compile(model, dynamic=False, fullgraph=True)
        compile_step(SOAP)

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
        # Weight-decay the 2D weights (body + embedding/head); not 1D params.
        groups = self.model.optim_param_groups()
        optim_groups = [
            {"params": groups["matrix"] + groups["embed_head"],
             "weight_decay": self.weight_decay},
            {"params": groups["scalar"], "weight_decay": 0.0},
        ]

        self.optimizer = SOAP(
            optim_groups,
            lr=torch.tensor(self.learning_rate),
            betas=(0.95, 0.95),
        )

        train_loader = self.train_dataloader.get_distributed_data_generator(
            batch_size=self.batch_size,
            world_size=self.world_size,
            rank=self.rank,
        )

        broadcast_model(self.dist, self.model)

        step = 0
        with tqdm(total=self.num_steps, desc="Training") as progress:
            while cb():
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                step += 1
                progress.update()
                if step == self.num_steps:
                    break

                data = next(train_loader)
                with self.ctx:
                    loss, *_ = self.model(*data)
                loss.backward()

                # Track a smoothed train loss (on-device, no per-step sync).
                ema = loss.detach()
                self.train_loss = (
                    ema if self.train_loss is None
                    else 0.9 * self.train_loss + 0.1 * ema
                )

                if self.dist is not None:
                    for param in self.model.parameters():
                        self.dist.all_reduce(
                            param.grad, op=self.dist.ReduceOp.AVG
                        )

                scale_lr = get_lr(
                    step,
                    self.num_steps,
                    cooldown_frac=self.cooldown_frac,
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = torch.tensor(
                        self.learning_rate * scale_lr
                    )

                self.optimizer.step()

    def get_result(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_loss = (
            float(self.train_loss) if self.train_loss is not None else None
        )
        return dict(model=self.model, dist=self.dist, train_loss=train_loss)
