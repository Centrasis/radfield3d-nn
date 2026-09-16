import lightning.pytorch as pl
from RadFiled3D.pytorch.radiationfieldloader import DataLoaderBuilder
from RadFiled3D.pytorch.datasets.radfield3d import RadField3DDataset
from typing import Type
from radfield3dnn import TrainingInputData, RadiationField, rf3RadiationField
from RadFiled3D.pytorch.datasets.processing import DataProcessing
from threading import Lock


class RadiationFieldDataModule(pl.LightningDataModule):
    def __init__(self, zip_directory, dataset_cls: Type[RadField3DDataset], batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, num_workers=None, data_processings: list[DataProcessing]=None):
        super().__init__()
        self.zip_directory = zip_directory
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.cpu_count = num_workers
        self._max_hits_per_voxel_per_file_per_stage = {}
        self._current_stage_creation = None
        self._dataset_cls = dataset_cls
        self.max_log_flux = 1.0
        self.is_prepared = False
        self._fields_count = 0
        self._train_count = 0
        self.data_processings: list[DataProcessing] = data_processings
        self.uploaded_processings = False
        self._dataloader_builder = None
        self._lock = Lock()
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def setup(self, stage=None):
        pass

    def __len__(self):
        return self._fields_count
    
    def get_train_count(self):
        return self._train_count

    def prepare_data(self) -> None:
        with self._lock:
            if self.is_prepared:
                return

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

            self._fields_count = len(self._dataloader_builder.file_paths)
            self._train_count = int(self.train_ratio * self._fields_count)

            self._train_dataset = self.dataloader_builder.build_train_dataset()
            self._val_dataset = self.dataloader_builder.build_val_dataset()
            self._test_dataset = self.dataloader_builder.build_test_dataset()

            self.is_prepared = True

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

    def train_dataloader(self):
        dl = self.dataloader_builder.build_dataloader(self.train_dataset, batch_size=self.batch_size, shuffle=True, worker_count=self.cpu_count)
        return dl

    def val_dataloader(self):
        dl = self.dataloader_builder.build_dataloader(self.val_dataset, batch_size=self.batch_size, shuffle=False, worker_count=self.cpu_count)
        return dl

    def test_dataloader(self):
        dl = self.dataloader_builder.build_dataloader(self.test_dataset, batch_size=self.batch_size, shuffle=False, worker_count=self.cpu_count)
        return dl

    def on_after_batch_transfer(self, batch: TrainingInputData, dataloader_idx):
        if not self.uploaded_processings:
            for i in range(len(self.data_processings)):
                if isinstance(batch.ground_truth, RadiationField) or isinstance(batch.ground_truth, rf3RadiationField):
                    self.data_processings[i] = self.data_processings[i].to(batch.ground_truth.scatter_field.flux.device)
                else:
                    self.data_processings[i] = self.data_processings[i].to(batch.ground_truth.flux.device)
            self.uploaded_processings = True

        for process in self.data_processings:
            batch = process(batch)
        return batch
