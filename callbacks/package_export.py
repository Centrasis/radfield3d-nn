import os

import lightning.pytorch as pl
from rich import print

from radfield3dnn.deploy import ModelPackager


class PackageExportCallback(pl.Callback):
    """After the test stage, write a self-contained RF3M deployment package (ONNX + validity
    domain + provenance + test metrics) next to the checkpoints. Best-effort: a packaging error
    is reported but never fails the run."""

    def __init__(self, out_dir: str, model_name: str, dataset_path: str,
                 max_energy_eV: float = 1.5e5, spectra_bins: int = 32,
                 export_fp16: bool | None = None):
        super().__init__()
        self.out_dir = out_dir
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.max_energy_eV = max_energy_eV
        self.spectra_bins = spectra_bins
        # None -> auto: export fp16 weights iff the run trained in mixed precision (its fp32 master
        # weights carry no more than fp16 of signal). True/False forces the choice.
        self.export_fp16 = export_fp16

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        try:
            datamodule = getattr(trainer, "datamodule", None)
            metrics = {k: v for k, v in trainer.callback_metrics.items() if str(k).startswith("test")}
            export_fp16 = self.export_fp16
            if export_fp16 is None:   # auto-detect from the trainer's precision ("16-mixed", "16", ...)
                export_fp16 = "16" in str(getattr(trainer, "precision", "")).lower()
            packager = ModelPackager(
                pl_module, datamodule, metrics,
                dataset_path=self.dataset_path,
                max_energy_eV=self.max_energy_eV,
                spectra_bins=self.spectra_bins,
                export_fp16=export_fp16,
            )
            packager.save(os.path.join(self.out_dir, f"{self.model_name}.rf3m"))
        except Exception as e:  # never break a finished training run over packaging
            print(f"[red]PackageExportCallback: failed to write model package ({e})[/red]")
