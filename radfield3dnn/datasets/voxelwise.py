from typing import List
from radfield3dnn import TrainingInputData
from RadFiled3D.pytorch.datasets.processing import DataProcessing
from RadFiled3D.pytorch.datasets.radfield3d import RadField3DVoxelwiseDataset


class AugmentedVoxelwiseDataset(RadField3DVoxelwiseDataset):
    def __init__(self, file_paths: str = None, zip_file: bool = None, dataprocessings: list[DataProcessing] = None, max_spectrum_energy_eV: float = 150000.0, input_spectra_bins: int = 150):
        super().__init__(file_paths=file_paths, zip_file=zip_file)
        self.dataprocessings = dataprocessings
        self.max_spectrum_energy_eV = max_spectrum_energy_eV
        self.input_spectra_bins = input_spectra_bins

    def prefetch_data(self):
        pass

    def __len__(self):
        dataset_size = super().__len__()
        multiplicator = 1.0
        if self.dataprocessings is not None:
            for aug in self.dataprocessings:
                multiplicator *= aug.dataset_multiplier()
        return int(dataset_size * multiplicator)

    def __getitem__(self, idx: int) -> TrainingInputData:
        raise NotImplementedError("This dataset does not support direct indexing. Use __getitems__ instead.")

    def __getitems__(self, indices: List[int]) -> TrainingInputData:
        inputs = super().__getitems__(indices)
        return inputs
