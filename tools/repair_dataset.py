from RadFiled3D.pytorch.datasets.radfield3d import RadField3DDataset
from RadFiled3D.pytorch.radiationfieldloader import DataLoaderBuilder
from pathlib import Path
import os
from rich import print
from rich.progress import track


class FilteringDataset(RadField3DDataset):
    def _get_field(self, idx):
        field = super()._get_field(idx)
        if not field.has_channel("geometry"):
            raise ValueError("Field is missing geometry channel")
        return field

    def __init__(self, file_paths, zip_file=None):
        super().__init__(file_paths=file_paths, zip_file=zip_file)
        field = self._get_field(0)
        metadata = self._get_metadata(0)
        self.default_ti = self.transform2training_input(field, metadata)

    def __getitem__(self, idx):
        try:
            return super().__getitem__(idx)
        except Exception as e:
            file_name = self.file_paths[idx] if isinstance(idx, int) else self.file_paths[idx.item()]
            print(f"[yellow]Failed to load dataset item at index {idx} from file {file_name}[/yellow]")
            print(f"[red]Error details: {e}[/red]")
            os.remove(file_name)
            return self.default_ti


if __name__ == "__main__":
    dataset_path = Path("D:/Datasets/RAF-QArtis-DynGeometry")
    if "fields" in os.listdir(dataset_path):
        dataset_path = dataset_path / "fields"
    dl = DataLoaderBuilder(dataset_path, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0, dataset_class=FilteringDataset)
    fields = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.rf3')]
    original_files_count = len(fields)
    
    for field in track(dl.build_train_dataloader(worker_count=8), description="Processing fields"):
        pass
    files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.rf3')]
    repaired_files_count = len(files)
    print(f"Original files count: {original_files_count}")
    print(f"Repaired files count: {repaired_files_count}")
    print(f"[green]Removed corrupted files: {original_files_count - repaired_files_count}[/green]")
