from benchopt import BaseSolver

from contextlib import nullcontext

import torch
from tqdm.auto import tqdm

from benchmark_utils.lr_scheduler import get_lr
from benchmark_utils.optimizers.scion_light import ScionLight
from benchmark_utils.distributed_tools import setup_distributed


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "Scion"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "learning_rate": [0.00036],
        "momentum": [0.1],
        "hidden_radius": [50.0],
        "lm_head_radius": [3000.0],
        "num_steps": [6200],
        "batch_size": [64],
        "slurm_nodes": [1, 2],
    }
    slurm_params = {
        "slurm_gres": "gpu:4",
        "slurm_ntasks_per_node": 4,
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

        # use mixed precision if cuda is available
        self.ctx = (
            torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
            if torch.cuda.is_available() else nullcontext()
        )

        # Torch compile the model and the optimizer step function
        self.model = torch.compile(model, dynamic=False, fullgraph=True)
        ScionLight.step = torch.compile(torch.no_grad(ScionLight.step))

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
        # Configure the optimizer with different groups for transformer and
        # lm_head (Spectral norm for transformer, Sign for lm_head)
        transformer_params = []
        lm_head_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.dim() >= 2:
                if "lm_head" in name:
                    lm_head_params.append(param)
                else:
                    transformer_params.append(param)

        optim_groups = [
            {
                "params": transformer_params,
                "norm": "Spectral",
                "norm_kwargs": {},
                "scale": self.hidden_radius,
            },
            {
                "params": lm_head_params,
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

        if self.dist is not None:
            self.dist.barrier()  # wait for all processes to be ready

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

                if self.dist is not None:
                    for param in self.model.parameters():
                        self.dist.all_reduce(
                            param.grad, op=self.dist.ReduceOp.AVG
                        )

                # determine and set the learning rate for this iteration
                scale_lr = get_lr(step, self.num_steps)
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
        return dict(model=self.model, dist=self.dist)
