from torchvision import datasets, transforms
from torchvision.transforms import v2
from utils.sampling import iid, noniid
import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple


class DatasetSplit(torch.utils.data.Dataset):
    """Custom Dataset class that returns a subset of another dataset based on indices.

    Args:
        dataset: The base dataset to sample from
        idxs: Indices to use for sampling from the base dataset
    """

    def __init__(self, dataset: torch.utils.data.Dataset, idxs: List[int]) -> None:
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[self.idxs[item]]
        return image, label


class CachedTensorDataset(torch.utils.data.Dataset):
    """Keep CIFAR images as raw uint8 tensors to avoid repeated PIL decoding."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        transform: Optional[Callable] = None,
    ) -> None:
        image_array = np.asarray(dataset.data)
        self.images = (
            torch.from_numpy(image_array).permute(0, 3, 1, 2).contiguous()
        )
        self.targets = list(dataset.targets)
        self.transform = transform

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        image = self.images[item]
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.targets[item])


trans_mnist = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
trans_cifar10_train = v2.Compose(
    [
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
trans_cifar10_val = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
trans_cifar100_train = v2.Compose(
    [
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ]
)
trans_cifar100_val = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ]
)


def get_data(
    args: Any,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, Dict, Dict, np.ndarray]:
    """Get train and test datasets and user splits for federated learning.

    Args:
        args: Arguments containing dataset configuration

    Returns:
        dataset_train: Training dataset
        dataset_test: Test dataset
        dict_users_train: Dictionary mapping users to training data indices
        dict_users_test: Dictionary mapping users to test data indices
        rand_set_all: Random set assignments for non-iid splitting
    """
    dataset_train = CachedTensorDataset(
        datasets.CIFAR10("data/cifar10", train=True, download=True),
        transform=trans_cifar10_train,
    )
    dataset_test = CachedTensorDataset(
        datasets.CIFAR10("data/cifar10", train=False, download=True),
        transform=trans_cifar10_val,
    )
    if args.iid:
        dict_users_train = iid(dataset_train, args.num_users)
        dict_users_test = iid(dataset_test, args.num_users)
        rand_set_all = np.array([])
    else:
        dict_users_train, rand_set_all = noniid(
            dataset_train,
            args.num_users,
            args.shard_per_user,
            args.server_data_ratio,
            size=args.num_samples,
        )
        dict_users_test, rand_set_all = noniid(
            dataset_test,
            args.num_users,
            args.shard_per_user,
            args.server_data_ratio,
            size=args.test_size,
            rand_set_all=rand_set_all,
        )

    return dataset_train, dataset_test, dict_users_train, dict_users_test, rand_set_all


def prepare_dataloaders(
    dataset_train: torch.utils.data.Dataset,
    dict_users_train: Dict,
    dataset_test: torch.utils.data.Dataset,
    dict_users_test: Dict,
    args: Any,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Prepare train and test data loaders for a user.

    Args:
        dataset_train: Training dataset
        dict_users_train: Dictionary mapping users to training data indices
        dataset_test: Test dataset
        dict_users_test: Dictionary mapping users to test data indices
        args: Arguments containing batch size configuration

    Returns:
        ldr_train: Training data loader
        ldr_test: Test data loader
    """
    ldr_train = torch.utils.data.DataLoader(
        DatasetSplit(dataset_train, dict_users_train),
        batch_size=args.local_bs,
        shuffle=True,
    )
    ldr_test = torch.utils.data.DataLoader(
        DatasetSplit(dataset_test, dict_users_test),
        batch_size=args.local_bs,
        shuffle=False,
    )
    return ldr_train, ldr_test
