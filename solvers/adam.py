from benchopt import BaseSolver

from contextlib import nullcontext

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from benchmark_utils.lr_scheduler import get_lr_trapezoidal
from benchmark_utils.torch_utils import compile_step
from benchmark_utils.distributed_tools import (
    setup_distributed, broadcast_model
)


class Solver(BaseSolver):

    name = 'Adam'

    # Defaults follow modded-nanogpt commit
    # 844e5fdb2334ff83324e6f1f900ce443dd9e1226 (run.sh):
    # lr=0.0018, betas=(0.9, 0.98), wd=0 (the reference's custom Adam
    # ignores weight decay), 9536 iterations with 256 warmup / 2048 warmdown,
    # batch_size=64,
    # sequence_length=1024 over 8 GPUs (=> global batch = 512).
    parameters = {
        'learning_rate': [1.8e-3],
        'weight_decay': [0.1],
        'num_steps': [8600],
        'batch_size': [64],
        'warmup_iters': [256],
        'cooldown_frac': [0.5],
        "slurm_nodes": [2],
        "sin_init": [False],
    }

    sampling_strategy = 'callback'

    def set_objective(self, train_dataloader, model):

        # Setup distributed training if needed
        self.dist, self.rank, self.world_size, device = setup_distributed()

        if self.sin_init:
            print("Using sinusoidal initialization")
            from benchmark_utils.sin_init import sinusoidal_
            model.init_func = sinusoidal_
            model.initialize_weights(seed=42)

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
        compile_step(AdamW)

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

        # Weight-decay the 2D body matrices; not the embedding/head or 1D
        # params (decaying the tied embedding hurts the final loss).
        groups = self.model.optim_param_groups()
        optim_groups = [
            {'params': groups["matrix"], 'weight_decay': self.weight_decay},
            {'params': groups["embed_head"] + groups["scalar"],
             'weight_decay': 0.0},
        ]

        # Create AdamW optimizer. Betas (0.9, 0.98) match the reference.
        self.optimizer = AdamW(
            optim_groups,
            lr=torch.tensor(self.learning_rate),
            betas=(0.9, 0.98),
            fused=True
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

                # determine and set the learning rate for this iteration
                scale_lr = get_lr_trapezoidal(
                    step, self.num_steps,
                    warmup_iters=self.warmup_iters,
                    cooldown_frac=self.cooldown_frac,
                )
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = torch.tensor(
                        self.learning_rate * scale_lr
                    )
                # step the self.optimizer
                self.optimizer.step()

    def get_result(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # wait for all operations to finish
        train_loss = (
            float(self.train_loss) if self.train_loss is not None else None
        )
        return dict(model=self.model, dist=self.dist, train_loss=train_loss)
