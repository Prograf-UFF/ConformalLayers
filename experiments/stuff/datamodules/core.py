from abc import ABC, abstractclassmethod
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Any, List, Optional, Sequence, Union
import pytorch_lightning as pl
import sklearn.model_selection
import torch


class ClassificationDataModule(pl.LightningDataModule, ABC):

    def __init__(self, *, batch_size: int, fit_dataset: Dataset, num_workers: int, test_dataset: Union[Dataset, Sequence[Dataset]], train_size: float, **kwargs: Any) -> None:
        super(ClassificationDataModule, self).__init__()
        self.batch_size = batch_size
        self.fit_dataset = fit_dataset
        self.num_workers = num_workers
        self.test_dataset = list(test_dataset) if isinstance(test_dataset, Sequence) else [test_dataset]
        self.train_size = train_size
        self.train_subset: Optional[Subset] = None
        self.val_subset: Optional[Subset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, 'fit'):
            # Get entries' indices realted to each class/target.
            subsets = {}
            for index, (_, target) in enumerate(self.fit_dataset):
                indices = subsets.get(target, None)
                if indices is not None:
                    indices.append(index)
                else:
                    subsets[target] = [index]
            # Perform balanced split.
            train_indices = []
            val_indices = []
            for indices in subsets.values():
                train_indices_split, val_indices_split = sklearn.model_selection.train_test_split(indices, test_size=(1.0 - self.train_size), train_size=self.train_size)
                train_indices.extend(train_indices_split)
                val_indices.extend(val_indices_split)
            train_indices.sort()
            val_indices.sort()
            # Define the subsets.
            self.train_subset = torch.utils.data.Subset(self.fit_dataset, train_indices)
            self.val_subset = torch.utils.data.Subset(self.fit_dataset, val_indices)

    def test_dataloader(self) -> List[DataLoader]:
        return [DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=False) for dataset in self.test_dataset]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_subset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=not isinstance(self.train_subset, torch.utils.data.IterableDataset), pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_subset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)

    @abstractclassmethod
    def test_dataset_name(cls) -> List[str]:
        raise NotImplementedError


class RandomDataModule(pl.LightningDataModule):

    def __init__(self, *, batch_size: int, num_trials: int, num_workers: int, **kwargs: Any) -> None:
        super(RandomDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = torch.utils.data.ConcatDataset([torch.utils.data.TensorDataset(torch.rand(1, 3, 32, 32, dtype=torch.float32).expand(batch_size, -1, -1, -1), torch.randint(10, (batch_size,), dtype=torch.long))] * num_trials)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
