from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def load_mnist_data(
    data_dir: str | Path,
    batch_size: int,
    train: bool = True,
    download: bool = True,
    subset_fraction: float = 0.2,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,),
                (0.3081,),
            ),
        ]
    )

    dataset = datasets.MNIST(
        str(data_dir), train=train, download=download, transform=transform
    )

    if subset_fraction < 1.0:
        num_samples = int(len(dataset) * subset_fraction)
        indices: NDArray[np.int64] = np.random.choice(
            len(dataset), num_samples, replace=False
        )
        subset_indices: list[int] = indices.tolist()
        dataset = Subset(dataset, subset_indices)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=2
    )
