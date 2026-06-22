from lightning.pytorch.callbacks import ModelCheckpoint as LightningModelCheckpoint
from lightning.pytorch import Trainer
import lightning.pytorch as pl
import warnings
import json
import os


class ModelCheckpoint(LightningModelCheckpoint):
    """
    ModelCheckpoint callback that handles missing metrics gracefully.
    """
    
    def _metric_available(self, trainer: pl.Trainer) -> bool:
        """Check if the monitored metric is available in the logged metrics."""
        if self.monitor is None:
            return True  # No metric to monitor
        
        if not hasattr(trainer, 'logged_metrics') or trainer.logged_metrics is None:
            return False
        return self.monitor in trainer.logged_metrics
    
    def _should_save_checkpoint(self, trainer: pl.Trainer) -> bool:
        """Check if we should save checkpoint, considering metric availability."""
        if not self._metric_available(trainer):
            warnings.warn(
                f"ModelCheckpoint metric '{self.monitor}' not available yet. "
                f"This is normal for validation metrics in the first epoch. "
                f"Skipping checkpoint save for epoch {trainer.current_epoch}.",
                UserWarning
            )
            return False
        return True
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Override to handle missing metrics."""
        if not self._should_save_checkpoint(trainer):
            return
        
        super().on_validation_end(trainer, pl_module)
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Override to handle missing metrics."""
        if not self._should_save_checkpoint(trainer):
            return
        
        super().on_train_epoch_end(trainer, pl_module)

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        config_path = filepath.replace(".ckpt", ".config")
        if os.path.exists(config_path):
            os.remove(config_path)
        with open(config_path, "w") as f:
            # trainer.lightning_module unwraps any DDP/strategy wrapper (trainer.model can be the
            # wrapper, which has no get_model_config()).
            json.dump(trainer.lightning_module.get_model_config(), f)
