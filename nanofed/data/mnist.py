from pathlib import Path
from typing import Sequence

import numpy as np
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
        indices_array = np.random.choice(
            a=len(dataset), size=num_samples, replace=False
        )
        subset_indices: Sequence[int] = list(map(int, indices_array))
        dataset = Subset(dataset, subset_indices)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=2
    )
