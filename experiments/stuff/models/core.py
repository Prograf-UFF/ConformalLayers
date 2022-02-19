from ..typing import ImageBatch, Logits, TargetBatch
from abc import ABC
from pytorch_lightning.callbacks.base import Callback
from torch.nn import Module
from typing import Any, Dict, List, Optional, Sequence, Tuple
import codecarbon
import operator, os
import pytorch_lightning as pl
import torch
import torchmetrics


class ClassificationModel(pl.LightningModule, ABC):

    def __init__(self, *, depth: int, early_stopping_patience: Optional[int] = None, learning_rate: Optional[float] = None, net: Module, optimizer: Optional[str] = None, run_dir: str, scheduler_patience: Optional[int] = None, test_dataset_name: Optional[Sequence[str]] = None, **kwargs: Any) -> None:
        super(ClassificationModel, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.depth = depth
        self.early_stopping_patience = early_stopping_patience
        self.learning_rate = learning_rate
        self.net = net
        self.optimizer = optimizer
        self.run_dir = run_dir
        self.scheduler_patience = scheduler_patience
        self.test_dataset_name = test_dataset_name

    def _fit_step(self, batch: Tuple[ImageBatch, TargetBatch], mode: str) -> Dict[str, Any]:
        # Evaluate the model on the given batch.
        images, targets = batch
        logits = self(images)
        # Compute and log the loss.
        loss = self.criterion(logits, targets)
        self.log(f'Loss/{mode}/Step', loss)
        # Return the computed values.
        return {'loss': loss, 'logits': logits.detach(), 'targets': targets.detach()}
    
    def _fit_epoch_end(self, step_outputs: List[Any], mode: str) -> None:
        # Compute and log the mean loss.
        loss = torch.stack(list(map(operator.itemgetter('loss'), step_outputs)))
        self.log(f'Loss/{mode}', loss.mean())
        # Compute and log metrics.
        logits = torch.cat(list(map(operator.itemgetter('logits'), step_outputs)))
        targets = torch.cat(list(map(operator.itemgetter('targets'), step_outputs)))
        self.log(f'Accuracy/{mode}', torchmetrics.functional.accuracy(logits, targets))

    def configure_callbacks(self) -> List[Callback]:
        # Apply early stopping.
        early_stopping = pl.callbacks.early_stopping.EarlyStopping(monitor='Accuracy/Val', mode='max', patience=self.early_stopping_patience)
        # Return the list of callbacks.
        return [early_stopping]

    def configure_optimizers(self) -> Dict[str, Any]:
        # Set the optimizer.
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.net.parameters(), momentum=0.9, lr=self.learning_rate)
        else:
            raise ValueError(f'Unknown optimizer: {self.optimizer}')
        # Set the learning rate scheduler.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.scheduler_patience)
        # Return the configuration.
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'Loss/Val'}}

    def forward(self, images: ImageBatch) -> Logits:
        return self.net(images)

    def predict_step(self, batch: Tuple[ImageBatch, TargetBatch], batch_idx: int, dataloader_idx: int = None) -> Dict[str, Any]:
        images, _ = batch
        # Estimate elapsed time and batch_size.
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self(images)
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
        max_memory = torch.cuda.max_memory_allocated(device=self.device)
        # Estimate CO2 emission in Kgs.
        emission_tracker = codecarbon.EmissionsTracker(measure_power_secs=1, output_dir=self.run_dir, save_to_file=False)
        emission_tracker.start()
        self(images)
        emission = emission_tracker.stop()
        # Return the estimated values.
        return {'batch_idx': batch_idx, 'depth': self.depth, 'batch_size': len(images), 'elapsed_time': elapsed_time, 'max_memory': max_memory, 'emission': emission}

    def test_step(self, batch: Tuple[ImageBatch, TargetBatch], batch_idx: int, dataloader_idx: int) -> Dict[str, Any]:
        # Evaluate the model on the given batch.
        images, targets = batch
        logits = self(images)
        # Return the computed values.
        return {'logits': logits.detach(), 'targets': targets.detach()}

    def test_epoch_end(self, step_outputs: List[List[Dict[str, Any]]]) -> None:
        # Compute and log metrics for each test dataset.
        for dataloader_step_outputs, dataset_name in zip(step_outputs, self.test_dataset_name):
            logits = torch.cat(list(map(operator.itemgetter('logits'), dataloader_step_outputs)))
            targets = torch.cat(list(map(operator.itemgetter('targets'), dataloader_step_outputs)))
            self.log(f'Accuracy/Test/{dataset_name}', torchmetrics.functional.accuracy(logits, targets))

    def training_step(self, batch: Tuple[ImageBatch, TargetBatch], batch_idx: int) -> Dict[str, Any]:
        return self._fit_step(batch, 'Train')

    def training_epoch_end(self, step_outputs: List[Dict[str, Any]]) -> None:
        self._fit_epoch_end(step_outputs, 'Train')

    def validation_step(self, batch: Tuple[ImageBatch, TargetBatch], batch_idx: int) -> Dict[str, Any]:
        return self._fit_step(batch, 'Val')

    def validation_epoch_end(self, step_outputs: List[Dict[str, Any]]) -> None:
        self._fit_epoch_end(step_outputs, 'Val')

    @classmethod
    def wandb_metric(cls) -> Dict[str, Any]:
        return {'name': 'Accuracy/Val', 'goal': 'maximize'}
