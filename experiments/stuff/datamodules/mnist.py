from .core import ClassificationDataModule
from typing import Any, List
import os
import torchvision


class MNIST(ClassificationDataModule):

    def __init__(self, *, datasets_root: str, **kwargs: Any) -> None:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Pad(2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        super(MNIST, self).__init__(
            fit_dataset=torchvision.datasets.MNIST(os.path.join(datasets_root, 'MNIST'), train=True, download=True, transform=transform),
            test_dataset=torchvision.datasets.MNIST(os.path.join(datasets_root, 'MNIST'), train=False, download=True, transform=transform),
            **kwargs
        )

    @classmethod
    def test_dataset_name(cls) -> List[str]:
        return ['clean']
