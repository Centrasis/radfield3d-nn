import os

import lightning.pytorch as pl
from rich import print

from radfield3dnn.deploy import ModelPackager


class PackageExportCallback(pl.Callback):
    """After the test stage, write a self-contained RF3M deployment package (ONNX + validity
    domain + provenance + test metrics) next to the checkpoints. Best-effort: a packaging error
    is reported but never fails the run."""

    def __init__(self, out_dir: str, model_name: str, dataset_path: str,
                 max_energy_eV: float = 1.5e5, spectra_bins: int = 32):
        super().__init__()
        self.out_dir = out_dir
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.max_energy_eV = max_energy_eV
        self.spectra_bins = spectra_bins

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        try:
            datamodule = getattr(trainer, "datamodule", None)
            metrics = {k: v for k, v in trainer.callback_metrics.items() if str(k).startswith("test")}
            packager = ModelPackager(
                pl_module, datamodule, metrics,
                dataset_path=self.dataset_path,
                max_energy_eV=self.max_energy_eV,
                spectra_bins=self.spectra_bins,
            )
            packager.save(os.path.join(self.out_dir, f"{self.model_name}.rf3m"))
        except Exception as e:  # never break a finished training run over packaging
            print(f"[red]PackageExportCallback: failed to write model package ({e})[/red]")
