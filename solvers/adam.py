from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import os

    import torch
    import torch.distributed as dist
    from torch.optim import AdamW

    from benchmark_utils.dataloading import distributed_data_generator


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
        'learning_rate': [1e-4],
        'weight_decay': [1e-4],
        'num_steps': [2000],
        'batch_size': [64],
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = []

    sampling_strategy = 'callback'

    def set_objective(self, train_files, model):
        self.train_files = train_files
        self.model = torch.compile(model)

        # configure the optimizer
        # List all parameters that require gradients
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
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

    def get_next(self, stop_val):
        return stop_val + 125

    def run(self, cb):

        # torchrun sets these env variables
        ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
        if ddp:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            # assert world_size == 8 # this code is designed for 8xH100
            # assert torch.cuda.is_available()
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            torch.cuda.set_device(device)
            dist.init_process_group(backend="nccl", device_id=device)
            dist.barrier()
            master_process = (rank == 0) # this process will do logging, checkpointing etc.
        else:
            rank = 0
            world_size = 1
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            master_process = True

        train_loader = distributed_data_generator(
            self.train_files, batch_size=self.batch_size * 1024,
            rank=rank, world_size=world_size
        )
        self.model = self.model.to(device)

        step = 0
        while cb():
            print(f"Step {step} (rank {rank})\r", end='', flush=True)

            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            step += 1
            inputs, targets = next(train_loader)
            self.model(inputs, targets, return_logits=False)[1].backward()
            if ddp:
                for param in self.model.parameters():
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

            # determine and set the learning rate for this iteration
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate * get_lr(step, self.num_steps)
            # step the self.optimizer
            self.optimizer.step()

    def get_result(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # wait for all operations to finish
        return dict(model=self.model)
