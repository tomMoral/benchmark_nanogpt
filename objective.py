from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import torch
    from benchmark_utils.dataloading import distributed_data_generator


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "Ordinary Least Squares"

    # URL of the main repo for this benchmark.
    url = "https://github.com/tomMoral/benchmark_nanogpt"

    requirements = ["torch"]

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.6"

    def set_data(self, train_files, val_files, model):
        self.train_files = train_files
        self.val_files = val_files
        self.model = model

    def evaluate_result(self, model):
        model.eval()
        val_loader = distributed_data_generator(
            self.val_files, batch_size=2*128, rank=0, world_size=1
        )

        with torch.no_grad():
            # Compute the validation loss
            val_loss, n_batches = 0.0, 0
            for i, batch in enumerate(val_loader):
                if i > 0: break
                inputs, targets = batch
                _, loss = self.model(inputs, targets, return_logits=False)
                val_loss += loss.item()
                n_batches += 1
            val_loss /= n_batches

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            value=val_loss,
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
            train_files=self.train_files,
            model=self.model,
        )
