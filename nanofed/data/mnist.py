from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore[import]


def load_mnist_data(
    data_dir: str | Path,
    batch_size: int,
    train: bool = True,
    download: bool = True,
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

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=2
    )
