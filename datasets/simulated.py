from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import torch


class DataLoader:
    def __init__(self, train=True):
        self.train = train
        self.X = torch.randn(32, 10)

    def get_distributed_data_generator(self, batch_size, rank, world_size):
        # Simulate a data loader that yields batches of data
        if self.train:
            # For training, we yield the data in batches
            while True:
                yield self.X,
        else:
            # For validation, we yield the entire dataset
            yield self.X,


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    def get_data(self):

        model = torch.nn.Linear(10, 1)

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(
            train_dataloader=DataLoader(train=True),
            val_dataloader=DataLoader(train=False),
            model=model
        )
