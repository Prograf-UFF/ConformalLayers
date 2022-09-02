from . import utils
from .core import ClassificationDataModule
from PIL import Image
from typing import Any, Callable, List, Optional, Tuple
import numpy as np
import os, shutil
import torchvision


class CIFAR10(ClassificationDataModule):

    _CORRUPTIONS = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur']

    def __init__(self, *, datasets_root: str, **kwargs: Any) -> None:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ])
        cifar10c_datasets = []
        for corruption in self._CORRUPTIONS:
            for severity in range(1, 5 + 1):
                cifar10c_datasets.append(CIFAR10C(os.path.join(datasets_root, 'CIFAR-10-C'), corruption=corruption, severity=severity, download=True, transform=transform))
        super(CIFAR10, self).__init__(
            fit_dataset=torchvision.datasets.CIFAR10(os.path.join(datasets_root, 'CIFAR-10'), train=True, download=True, transform=transform),
            test_dataset=[torchvision.datasets.CIFAR10(os.path.join(datasets_root, 'CIFAR-10'), train=False, download=True, transform=transform), *cifar10c_datasets],
            **kwargs
        )

    @classmethod
    def test_dataset_name(cls) -> List[str]:
        result = ['clean']
        for corruption in cls._CORRUPTIONS:
            for severity in range(1, 5 + 1):
                result.append(f'{corruption}_{severity}')
        return result


class CIFAR10C(torchvision.datasets.VisionDataset):

    def __init__(self, root: str, corruption: str, severity: int, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False):
        super(CIFAR10C, self).__init__(root, transform=transform, target_transform=target_transform)
        if download:
            self._download(root)
        self.severity = severity
        self.images = np.load(os.path.join(root, f'{corruption}.npy'), mmap_mode='r')
        self.targets = np.load(os.path.join(root, 'labels.npy'), mmap_mode='r')

    def _download(self, root: str) -> None:
        with utils.checkpoint(os.path.join(root, '__install_check')) as exists:
            if not exists:
                # Download the zip file.
                utils.download('https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1', root)
                # Unpack the zip file.
                utils.unpack(os.path.join(root, 'CIFAR-10-C.tar'), root)
                for filename in os.listdir(os.path.join(root, 'CIFAR-10-C')):
                    shutil.move(os.path.join(root, 'CIFAR-10-C', filename), os.path.join(root, filename))
                os.removedirs(os.path.join(root, 'CIFAR-10-C'))
                # Delete the zip file.
                os.remove(os.path.join(root, 'CIFAR-10-C.tar'))
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        index += 10000 * (self.severity - 1)
        image, target = self.images[index], self.targets[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target
    
    def __len__(self):
        return 10000
