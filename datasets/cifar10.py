from benchopt import BaseDataset
from benchopt.config import get_data_path

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, DistributedSampler

from benchmark_utils.resnet import ResNet


class DistributedDataLoader:
    """Expose CIFAR batches through the benchmark dataloader contract.

    ``get_distributed_data_generator`` yields ``(images, targets)`` batches
    already moved to the GPU when available. The training stream is infinite
    and reshuffled every epoch; the validation stream is a single finite pass
    (the objective iterates it until exhaustion).
    """

    def __init__(self, dataset, train):
        self.dataset = dataset
        self.train = train

    def get_distributed_data_generator(self, batch_size, world_size, rank):
        # Explicit num_replicas/rank => no default process group required,
        # so this also works in the non-distributed (world_size=1) case.
        sampler = DistributedSampler(
            self.dataset, num_replicas=world_size, rank=rank,
            shuffle=self.train, drop_last=self.train,
        )
        loader = DataLoader(
            self.dataset, batch_size=batch_size, sampler=sampler,
            num_workers=4, pin_memory=torch.cuda.is_available(),
            persistent_workers=False, drop_last=self.train,
        )

        def _to_device(batch):
            images, targets = batch
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
            return images, targets

        if self.train:
            epoch = 0
            while True:
                sampler.set_epoch(epoch)
                for batch in loader:
                    yield _to_device(batch)
                epoch += 1
        else:
            for batch in loader:
                yield _to_device(batch)


class Dataset(BaseDataset):

    name = "CIFAR-10"

    # Normalization statistics from
    # https://github.com/kuangliu/pytorch-cifar/blob/master/main.py#L34
    normalization_mean = (0.4914, 0.4822, 0.4465)
    normalization_std = (0.2023, 0.1994, 0.2010)

    requirements = ["torchvision"]

    def get_data(self):
        # Augmentation is always on for training (random crop + horizontal
        # flip), matching the standard CIFAR ResNet recipe.
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                self.normalization_mean, self.normalization_std
            ),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                self.normalization_mean, self.normalization_std
            ),
        ])

        data_dir = get_data_path("cifar10")
        train_dataset = CIFAR10(
            root=data_dir, train=True, download=True,
            transform=train_transform,
        )
        val_dataset = CIFAR10(
            root=data_dir, train=False, download=True,
            transform=val_transform,
        )

        model = ResNet(num_classes=10)

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(
            train_dataloader=DistributedDataLoader(train_dataset, train=True),
            val_dataloader=DistributedDataLoader(val_dataset, train=False),
            model=model,
        )
