from benchopt import BaseObjective

import torch

from benchmark_utils.stopping import TargetStoppingCriterion


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "Deep Learning Optimization with NanoGPT"

    # URL of the main repo for this benchmark.
    url = "https://github.com/tomMoral/benchmark_nanogpt"

    requirements = ["pytorch", "tqdm"]

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.7"

    # Budget is each solver's num_steps; stop early only if the dataset's
    # target metric is reached (target read off this objective, see set_data).
    stopping_criterion = TargetStoppingCriterion(strategy="callback")

    def set_data(self, train_dataloader, val_dataloader, model, target=None):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        # Static per-dataset target for the stopping criterion (None => run the
        # full budget). Read by TargetStoppingCriterion via the solver handle.
        self.target = target

    def evaluate_result(self, model, dist=None, train_loss=None):
        model.eval()

        if dist is not None:
            # Solvers all-reduce gradients but not buffers, so models with
            # BatchNorm (ResNet) keep per-rank running stats. Average the float
            # buffers so every rank evaluates the same model. This is a no-op
            # for the GPT model, whose only buffer is identical across ranks.
            for buf in model.buffers():
                if buf.is_floating_point():
                    dist.all_reduce(buf, op=dist.ReduceOp.AVG)

        val_batch_size = 64  # Batch of 64 for validation
        if dist is not None:
            # In distributed mode, we use the distributed data generator
            rank, size = dist.get_rank(), dist.get_world_size()
            val_loader = self.val_dataloader.get_distributed_data_generator(
                batch_size=val_batch_size, rank=rank, world_size=size
            )
        else:
            # In non-distributed mode, we use the regular data generator
            val_loader = self.val_dataloader.get_distributed_data_generator(
                batch_size=val_batch_size, rank=0, world_size=1
            )

        with torch.no_grad():
            # Sample-weighted average of the per-batch eval metric returned by
            # the model. Cope with uneven batch effects.
            total_loss, n_samples = 0.0, 0
            for data in val_loader:
                loss, *_ = model(*data)
                bs = data[-1].shape[0]
                total_loss += loss.item() * bs
                n_samples += bs

            if dist is not None:
                # Reduce sums to accomodate with partial batches.
                stats = torch.tensor(
                    [total_loss, n_samples], device=model.device
                )
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                total_loss, n_samples = stats[0].item(), stats[1].item()

            val_loss = total_loss / n_samples

        del val_loader

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        # `train_loss` is optionally reported by solvers (e.g. Muon-debug) so
        # the train/val curves can be compared offline.
        return dict(
            value=val_loss,
            train_loss=train_loss,
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        return dict(model=self.model)

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        return dict(
            train_dataloader=self.train_dataloader,
            model=self.model,
        )
