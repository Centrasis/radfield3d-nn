import lightning.pytorch as pl
from RadFiled3D.pytorch.radiationfieldloader import DataLoaderBuilder
from RadFiled3D.pytorch.datasets.radfield3d import RadField3DDataset
from typing import Type
from radfield3dnn.rftypes import TrainingInputData, RadiationField, rf3RadiationField
from RadFiled3D.pytorch.datasets.processing import DataProcessing
from radfield3dnn.preprocessing.augmentations.augmentation_limit import LimitedAugmentation
from radfield3dnn.datasets.prefetcher import CudaStreamPrefetcher


class RadiationFieldDataModule(pl.LightningDataModule):
    def __init__(self, zip_directory, dataset_cls: Type[RadField3DDataset], batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, num_workers=None, data_processings: list[DataProcessing]=None, prefetch_to_device: bool = True, max_fields: int = None, cache_to_ram: bool = False, cache_ram_gb: float = None):
        super().__init__()
        self.zip_directory = zip_directory
        # Overlap the H2D upload + GPU preprocessing of the next batch with the current step's
        # compute via a side CUDA stream (see prefetcher.py). Only engages when a CUDA device is
        # attached; _prefetch_active flips true once we actually wrap a dataloader, telling the
        # transfer hooks below to short-circuit (the prefetcher already did their work).
        self._prefetch_to_device = prefetch_to_device
        self._prefetch_active = False
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.cpu_count = num_workers
        self._max_hits_per_voxel_per_file_per_stage = {}
        self._current_stage_creation = None
        self._dataset_cls = dataset_cls
        self._fields_count = 0
        self._train_count = 0
        self.data_processings: list[DataProcessing] = data_processings
        self.uploaded_processings = False
        self._dataloader_builder = None
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

        def set_augmentations(dataset: RadField3DDataset):
            dataset.data_processings = self.data_processings

        self._dataloader_builder = DataLoaderBuilder(
            self.zip_directory,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            dataset_class=self._dataset_cls,
            on_dataset_created=set_augmentations
        )

        # Optional field-count cap for fast iteration / coverage ablations. Subsample the file list
        # (seeded deterministically by the run's global seed) and re-split into train/val/test, so a
        # reduced run still has held-out fields drawn from the same beam-parameter coverage.
        if max_fields is not None and max_fields < len(self._dataloader_builder.file_paths):
            import random
            from torch.utils.data import random_split
            fp = list(self._dataloader_builder.file_paths)
            random.Random(1337).shuffle(fp)
            fp = fp[:int(max_fields)]
            self._dataloader_builder.file_paths = fp
            self._dataloader_builder.train_files, self._dataloader_builder.val_files, self._dataloader_builder.test_files = \
                random_split(fp, [self.train_ratio, self.val_ratio, self.test_ratio])
            print(f"[yellow]max_fields={max_fields}: using {len(fp)} of the available fields "
                  f"(train≈{int(self.train_ratio*len(fp))}).[/yellow]")

        self._fields_count = len(self._dataloader_builder.file_paths)
        self._train_count = int(self.train_ratio * self._fields_count)

        self._train_dataset = self.dataloader_builder.build_train_dataset()
        self._val_dataset = self.dataloader_builder.build_val_dataset()
        self._test_dataset = self.dataloader_builder.build_test_dataset()

        # Optional RAM cache of the RAW decoded fields (the expensive deterministic decode). The
        # per-epoch stochastic augmentations still run fresh downstream in on_after_batch_transfer.
        if cache_to_ram:
            import psutil
            from radfield3dnn.datasets.ram_cache import RamCachedDataset, compute_ram_budget
            # Dynamic soft budget from the live system. cache_ram_gb overrides the auto sizing.
            total = compute_ram_budget(override_gb=cache_ram_gb)
            workers = max(1, int(num_workers or 0))
            per_proc = total / workers
            # HARD guard against swap: stop caching (in every process) once live free RAM would drop
            # below this floor — a startup byte-budget can't foresee the training process's own
            # growing footprint (worker replicas, pinned buffers, the model). Keep ~20% of RAM free.
            try:
                min_free = int(0.20 * psutil.virtual_memory().total)
            except Exception:
                min_free = 0
            train_bytes, val_bytes = per_proc * 0.8, per_proc * 0.2
            self._train_dataset = RamCachedDataset(self._train_dataset, max_bytes=train_bytes, min_free_bytes=min_free)
            self._val_dataset = RamCachedDataset(self._val_dataset, max_bytes=val_bytes, min_free_bytes=min_free)
            print(f"[green]RAM cache enabled (memory-pressure-aware): soft budget ~{total/1e9:.1f} GB "
                  f"({per_proc/1e9:.1f} GB/proc × {workers}); caching stops if free RAM < "
                  f"{min_free/1e9:.1f} GB. Raw decoded fields cached, augmentations stay per-epoch.[/green]")

    def __len__(self):
        return self._fields_count
    
    def get_train_count(self):
        return self._train_count

    def setup(self, stage) -> None:
        pass

    def prepare_dataset(self) -> None:
        pass

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            self.prepare_data()
        return self._train_dataset
    
    @property
    def val_dataset(self):
        if self._val_dataset is None:
            self.prepare_data()
        return self._val_dataset
    
    @property
    def test_dataset(self):
        if self._test_dataset is None:
            self.prepare_data()
        return self._test_dataset

    @property
    def dataloader_builder(self) -> DataLoaderBuilder:
        if self._dataloader_builder is None:
            self.prepare_data()
        return self._dataloader_builder

    def _resolve_device(self):
        """The CUDA device the trainer runs on, or None when there is no CUDA trainer attached
        (CPU tests / manual datamodule use → prefetcher disabled, original path runs)."""
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None
        try:
            device = trainer.strategy.root_device
        except (AttributeError, RuntimeError):
            return None
        return device if device.type == "cuda" else None

    def _maybe_wrap(self, dl, is_training: bool):
        """Wrap a dataloader in the side-stream prefetcher when prefetch is enabled and a CUDA
        device is available; otherwise return it untouched. ``is_training`` is propagated to the
        processings so train-only augmentations/samplers are skipped on the val/test loaders."""
        if not self._prefetch_to_device:
            return dl
        device = self._resolve_device()
        if device is None:
            return dl
        self._prefetch_active = True
        return CudaStreamPrefetcher(dl, device, self.data_processings, is_training=is_training)

    def train_dataloader(self):
        dl = self.dataloader_builder.build_dataloader(self.train_dataset, batch_size=self.batch_size, shuffle=True, worker_count=self.cpu_count)
        return self._maybe_wrap(dl, is_training=True)

    def val_dataloader(self):
        dl = self.dataloader_builder.build_dataloader(self.val_dataset, batch_size=self.batch_size, shuffle=False, worker_count=self.cpu_count)
        return self._maybe_wrap(dl, is_training=False)

    def test_dataloader(self):
        dl = self.dataloader_builder.build_dataloader(self.test_dataset, batch_size=self.batch_size, shuffle=False, worker_count=self.cpu_count)
        return self._maybe_wrap(dl, is_training=False)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # When the prefetcher is active the batch is already resident on the target device; skip
        # Lightning's redundant move (this is what removes the ~22% batch_to_device cost).
        if self._prefetch_active:
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def on_after_batch_transfer(self, batch: TrainingInputData, dataloader_idx):
        # The prefetcher already ran the processings (and the one-time .to(device) upload) on the
        # side stream, so this becomes a pass-through when active.
        if self._prefetch_active:
            return batch
        if not self.uploaded_processings:
            for i in range(len(self.data_processings)):
                if isinstance(batch.ground_truth, RadiationField) or isinstance(batch.ground_truth, rf3RadiationField):
                    self.data_processings[i] = self.data_processings[i].to(batch.ground_truth.scatter_field.flux.device)
                else:
                    self.data_processings[i] = self.data_processings[i].to(batch.ground_truth.flux.device)
            self.uploaded_processings = True

        # Non-prefetch path: propagate the trainer's train/val state to each processing (same role
        # as CudaStreamPrefetcher.is_training) so train-only augmentations/samplers self-gate off
        # during validation, then apply.
        is_training = (self.trainer is None) or self.trainer.training
        for process in self.data_processings:
            process.train(is_training)
            batch = process(batch)
        return batch
