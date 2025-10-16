from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import os

    from tqdm.auto import tqdm

    import torch
    import torch.distributed as dist


# -----------------------------------------------------------------------------
# Scion optimizer implementation


@torch.compile
def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
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


class Norm(object):
    def lmo(self, g):
        raise NotImplementedError


class Spectral(Norm):
    def __init__(self, steps=5):
        self.steps = steps

    def lmo(self, g):
        g = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(
            g.shape
        )
        d_out, d_in = g.shape
        g *= (d_out / d_in) ** 0.5
        return g


class Sign(Norm):
    def __init__(self, zero_init=False):
        self.zero_init = zero_init

    def lmo(self, g):
        _, d_in = g.shape
        return (1 / d_in) * torch.sign(g)


norm_dict = {"Spectral": Spectral, "Sign": Sign}


class ScionLight(torch.optim.Optimizer):
    """Memory-efficient variant of the Scion optimizer from https://github.com/LIONS-EPFL/scion/blob/main/examples/modded-nanogpt/train_gpt_scionlight.py

    This implementation saves memory by storing only the averaged gradient instead of
    both the gradient and its average. Note that gradients should not be zeroed since
    p.grad is used directly to store the gradient average.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): Learning rate (default: 1e-3)
        momentum (float, optional): One minus the traditional momentum factor. For example,
            a traditional momentum of 0.9 would be specified as momentum=0.1 here (default: 1.0)
        norm (str, optional): Choice of norm for gradient projection ('Auto', 'SpectralConv',
            'ColNorm', 'RowNorm', 'BiasRMS', 'Spectral', or 'Sign') (default: 'Auto')
        norm_kwargs (dict, optional): Additional arguments for the norm projection (default: None)
        scale (float, optional): Scale factor for updates (default: 1.0)
        unconstrained (bool, optional): Whether to use unconstrained updates (default: False)

    Example:
        >>> radius = 50.0
        >>> optim_groups = [{
        ...     'params': model.transformer.h.parameters(),
        ...     'norm': 'Spectral',
        ...     'norm_kwargs': {},
        ...     'scale': radius,
        ... }, {
        ...     'params': model.lm_head.parameters(),
        ...     'norm': 'Sign',
        ...     'norm_kwargs': {},
        ...     'scale': radius*60.0,
        ... }]
        >>> optimizer = ScionLight(optim_groups, lr=2**-12, momentum=0.1)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=1.0,
        norm: str = "Auto",
        norm_kwargs: dict = None,
        scale=1.0,
        unconstrained=False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if norm_kwargs is None:
            norm_kwargs = {}
        defaults = dict(
            lr=lr,
            momentum=momentum,
            scale=scale,
            unconstrained=unconstrained,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            scale = group["scale"]
            unconstrained = group["unconstrained"]
            norm_backend = norm_dict[group["norm"]](**group["norm_kwargs"])
            for p in group["params"]:
                G = p.grad
                if G is None:
                    continue

                update = scale * norm_backend.lmo(G)
                if not unconstrained:
                    p.data.mul_(1 - lr)
                p.data.add_(update, alpha=-lr)

                if momentum != 1:
                    G.mul_(1 - momentum)


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
    name = "ScionLight"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "learning_rate": [2**-12],  # ~0.000244
        "momentum": [0.1],
        "hidden_radius": [50.0],
        "lm_head_radius": [3000.0],
        "num_steps": [3000],
        "batch_size": [32],
    }

    # List of packages needed to run the solver.
    requirements = []

    sampling_strategy = "callback"

    def set_objective(self, train_dataloader, model):
        # Use submitit helpers to setup distributed training easily.
        try:
            import submitit

            submitit.helpers.TorchDistributedEnvironment().export()
            ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
        except (ImportError, RuntimeError):
            ddp = False
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
        # Configure the optimizer with different groups for transformer and lm_head
        # Get transformer parameters (use Spectral norm)
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
            optim_groups, lr=self.learning_rate, momentum=self.momentum
        )

        train_loader = self.train_dataloader.get_distributed_data_generator(
            batch_size=self.batch_size * 1024 * self.world_size,
            rank=self.rank,
            world_size=self.world_size,
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
                loss, *_ = self.model(*data)
                loss.backward()

                if self.dist is not None:
                    for param in self.model.parameters():
                        self.dist.all_reduce(param.grad, op=self.dist.ReduceOp.AVG)

                # determine and set the learning rate for this iteration
                scale_lr = get_lr(step, self.num_steps)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate * scale_lr

                # step the optimizer
                # Note: ScionLight uses gradients for momentum, so don't zero them
                self.optimizer.step()

    def get_result(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # wait for all operations to finish
        return dict(model=self.model, dist=self.dist)
