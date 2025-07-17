from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import os

    from tqdm.auto import tqdm

    import torch
    import torch.distributed as dist
    from torch.optim import AdamW


# learning rate schedule: stable then decay
def get_lr(step, num_iterations, cooldown_frac=0.4):
    x = step / num_iterations  # progress in training
    assert 0 <= x < 1
    if x < 1 - cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / cooldown_frac
        return w * 1.0 + (1 - w) * 0.1


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'Adam'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'learning_rate': [1e-3],
        'weight_decay': [1e-4],
        'num_steps': [3000],
        'batch_size': [16],
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = []

    sampling_strategy = 'callback'

    def set_objective(self, train_dataloader, model):

        # Use submitit helpers to setup distributed training easily.
        import submitit
        submitit.helpers.TorchDistributedEnvironment().export()
        ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
        if ddp:
            print("Running in Distributed Data Parallel (DDP) mode")
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            assert torch.cuda.is_available()
            # TorchDistributedEnvironment sets the visible devices to the
            # current rank, so we can use the default device.
            device = torch.device("cuda", 0)
            torch.cuda.set_device(device)
            dist.init_process_group(backend="nccl", device_id=device)
            self.dist = dist
        else:
            self.rank = 0
            self.world_size = 1
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.dist = None
        model = model.to(device=device)
        self.model = torch.compile(model)
        self.model.device = device  # store the device in the model
        self.train_dataloader = train_dataloader

    def get_next(self, stop_val):
        return stop_val + 250

    def warm_up(self):
        self.run_once(stop_val=10)

    def run(self, cb):

        # configure the optimizer
        # List all parameters that require gradients
        param_dict = {
            pn: p for pn, p in self.model.named_parameters()
            if p.requires_grad
        }

        # create optim groups. Any parameters that is 2D will be weight
        # decayed, otherwise no. i.e. all weight tensors in
        # matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Create AdamW optimizer
        # TODO: consider using a ZeroRedundancyOptimizer
        self.optimizer = AdamW(
            optim_groups, lr=self.learning_rate, betas=(0.9, 0.95), fused=True
        )

        train_loader = self.train_dataloader.get_distributed_data_generator(
            batch_size=self.batch_size * 1024 * self.world_size,
            rank=self.rank, world_size=self.world_size
        )

        if self.dist is not None:
            self.dist.barrier()  # wait for all processes to be ready

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
                    param_group['lr'] = self.learning_rate * scale_lr
                # step the self.optimizer
                self.optimizer.step()

    def get_result(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # wait for all operations to finish
        return dict(model=self.model, dist=self.dist)
