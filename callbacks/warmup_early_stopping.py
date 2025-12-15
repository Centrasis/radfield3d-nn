from lightning.pytorch.callbacks import EarlyStopping
import lightning.pytorch as pl
import warnings


class WarmupEarlyStopping(EarlyStopping):
    """
    Early stopping callback that only becomes active after a warmup period.
    """
    
    def __init__(self, warmup_epochs: int = 50, *args, **kwargs):
        # The monitor parameter is in kwargs and will be set by the parent class
        super().__init__(*args, **kwargs)
        self.warmup_epochs = warmup_epochs
    
    def _should_skip_check(self, trainer: pl.Trainer) -> bool:
        """Skip early stopping check during warmup period."""
        return trainer.current_epoch < self.warmup_epochs
    
    def _metric_available(self, trainer: pl.Trainer) -> bool:
        """Check if the monitored metric is available in the logged metrics."""
        if not hasattr(trainer, 'logged_metrics') or trainer.logged_metrics is None:
            return False
        return self.monitor in trainer.logged_metrics
    
    def _run_early_stopping_check(self, trainer: pl.Trainer) -> None:
        """Only run early stopping check after warmup period and when metric is available."""
        if self._should_skip_check(trainer):
            return
        
        if not self._metric_available(trainer):
            warnings.warn(
                f"Early stopping metric '{self.monitor}' not available yet. "
                f"This is normal for validation metrics in the first epoch. "
                f"Skipping early stopping check for epoch {trainer.current_epoch}.",
                UserWarning
            )
            return
        
        super()._run_early_stopping_check(trainer)
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Override to skip during warmup and when metric is not available."""
        if self._should_skip_check(trainer):
            return
        
        if not self._metric_available(trainer):
            warnings.warn(
                f"Early stopping metric '{self.monitor}' not available yet. "
                f"This is normal for validation metrics in the first epoch. "
                f"Skipping early stopping check for epoch {trainer.current_epoch}.",
                UserWarning
            )
            return
        
        super().on_validation_end(trainer, pl_module)
