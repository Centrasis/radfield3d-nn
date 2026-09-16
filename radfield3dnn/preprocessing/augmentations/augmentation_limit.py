from RadFiled3D.pytorch.datasets.processing import DataProcessing
from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer
from typing import Any
from rich import print


class LimitedAugmentation(LightningModule):
    def __init__(self, augmentation: DataProcessing, end_epoch: int, start_epoch: int = 0):
        """
        Apply an augmentation only for a limited number of epochs.
        :param augmentation: The augmentation to apply.
        :param end_epoch: The epoch at which to stop applying the augmentation.
        :param start_epoch: The epoch at which to start applying the augmentation.
        """
        super().__init__()
        self.augmentation = augmentation
        self.end_epoch = end_epoch
        self.start_epoch = start_epoch
        self._is_active = False

    def forward(self, x):
        new_is_active = self.current_epoch >= self.start_epoch and self.current_epoch <= self.end_epoch
        if new_is_active != self._is_active:
            print(f"[blue]LimitedAugmentation: {'Activating' if new_is_active else 'Deactivating'} augmentation {self.augmentation.get_name()} at epoch {self.current_epoch}[/blue]")
        self._is_active = new_is_active
        if self.training and self._is_active:
            # Drive any annealable augmentation: fraction of the active window
            # elapsed (0 at start_epoch → 1 at end_epoch). Generic hook — the
            # wrapper does not need to know what the augmentation does with it.
            if hasattr(self.augmentation, "set_schedule_progress"):
                span = max(1, self.end_epoch - self.start_epoch)
                self.augmentation.set_schedule_progress((self.current_epoch - self.start_epoch) / span)
            x = self.augmentation(x)
        return x
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.augmentation.to(*args, **kwargs)
        return self

    def dataset_multiplier(self) -> float:
        if not self.training:
            return min(1.0, self.augmentation.dataset_multiplier())
        return self.augmentation.dataset_multiplier() if self._is_active else 1.0

    def get_parameters(self) -> dict[str, Any]:
        return self.augmentation.get_parameters()

    def get_name(self) -> str:
        return self.augmentation.__class__.__name__
