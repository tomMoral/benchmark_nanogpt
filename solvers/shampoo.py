from contextlib import nullcontext

import torch
from benchopt import BaseSolver
from tqdm.auto import tqdm

from benchmark_utils.distributed_tools import setup_distributed
from benchmark_utils.lr_scheduler import get_lr
from benchmark_utils.optimizers.shampoo import Shampoo


class Solver(BaseSolver):
    name = "Shampoo"

    parameters = {
        "learning_rate": [1e-2],
        "momentum": [0.9],
        "weight_decay": [1e-4],
        "precondition_frequency": [10],
        "shampoo_beta": [0.95],
        "eps": [1e-8],
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
        Shampoo.step = torch.compile(torch.no_grad(Shampoo.step))

    def __del__(self):
        if getattr(self, "dist", None) is not None:
            self.dist.destroy_process_group()

    def get_next(self, stop_val):
        return stop_val + 250

    def warm_up(self):
        self.run_once(stop_val=10)

    def run(self, cb):
        param_dict = {
            pn: p for pn, p in self.model.named_parameters() if p.requires_grad
        }
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        self.optimizer = Shampoo(
            optim_groups,
            lr=torch.tensor(self.learning_rate),
            momentum=self.momentum,
            shampoo_beta=self.shampoo_beta,
            eps=self.eps,
            precondition_frequency=self.precondition_frequency,
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
                self.optimizer.zero_grad(set_to_none=True)

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
                        self.dist.all_reduce(
                            param.grad, op=self.dist.ReduceOp.AVG
                        )

                scale_lr = get_lr(step, self.num_steps)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = torch.tensor(
                        self.learning_rate * scale_lr
                    )

                self.optimizer.step()

    def get_result(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return dict(model=self.model, dist=self.dist)
