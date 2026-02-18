from contextlib import nullcontext

import torch
from benchmark_utils.distributed_tools import setup_distributed
from benchmark_utils.lr_scheduler import get_lr
from benchmark_utils.optimizers.muon import Muon
from benchopt import BaseSolver
from torch.optim import AdamW
from tqdm.auto import tqdm


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
            muon_params,
            lr=torch.tensor(self.muon_lr),
            momentum=self.muon_momentum,
        )

        self.adam_optimizer = AdamW(
            [
                {
                    "params": adam_decay_params,
                    "weight_decay": self.adam_weight_decay
                },
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
                        self.dist.all_reduce(
                            param.grad, op=self.dist.ReduceOp.AVG
                        )

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
