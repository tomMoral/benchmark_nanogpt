from benchopt import BaseSolver

from contextlib import nullcontext

import torch
from tqdm.auto import tqdm

from benchmark_utils.lr_scheduler import get_lr
from benchmark_utils.optimizers.scion_light import ScionLight
from benchmark_utils.torch_utils import compile_step
from benchmark_utils.distributed_tools import (
    setup_distributed, broadcast_model
)


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "Scion"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "learning_rate": [0.00026],
        "momentum": [0.1],
        "hidden_radius": [50.0],
        "lm_head_radius": [3000.0],
        "num_steps": [6200],
        "batch_size": [64],
        "cooldown_frac": [0.5],
        "slurm_nodes": [2],
    }

    # List of packages needed to run the solver.
    requirements = []

    sampling_strategy = "callback"

    def set_objective(self, train_dataloader, model):

        # Setup distributed training if needed
        self.dist, self.rank, self.world_size, device = setup_distributed()

        model = model.to(device=device)
        model.device = device  # store the device in the model
        self.train_dataloader = train_dataloader
        self.train_loss = None

        # use mixed precision if cuda is available
        self.ctx = (
            torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
            if torch.cuda.is_available() else nullcontext()
        )

        # Torch compile the model and the optimizer step function
        self.model = torch.compile(model, dynamic=False, fullgraph=True)
        compile_step(ScionLight)

    def __del__(self):
        # Clean up communication resources
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
        # Spectral norm on the 2D body matrices; Sign norm on the
        # embedding/head (1D params, if any, ride with the head group).
        groups = self.model.optim_param_groups()
        optim_groups = [
            {
                "params": groups["matrix"],
                "norm": "Spectral",
                "norm_kwargs": {},
                "scale": self.hidden_radius,
            },
            {
                "params": groups["embed_head"] + groups["scalar"],
                "norm": "Sign",
                "norm_kwargs": {},
                "scale": self.lm_head_radius,
            },
        ]

        # Create ScionLight optimizer
        self.optimizer = ScionLight(
            optim_groups,
            lr=torch.tensor(self.learning_rate),
            momentum=self.momentum
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

                # Initialize gradients to zero on first step only
                if step == 0:
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

                # determine and set the learning rate for this iteration.
                scale_lr = get_lr(
                    step,
                    self.num_steps,
                    cooldown_frac=self.cooldown_frac,
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = torch.tensor(
                        self.learning_rate * scale_lr
                    )

                # step the optimizer
                # Note: ScionLight uses gradients to store the momentum,
                # so don't zero them
                self.optimizer.step()

    def get_result(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # wait for all operations to finish
        train_loss = (
            float(self.train_loss) if self.train_loss is not None else None
        )
        return dict(model=self.model, dist=self.dist, train_loss=train_loss)
