import torch
from datasets.dataloader import RadiationFieldDataModule
from pathlib import Path
from rich import print
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn, BarColumn, TextColumn, TaskProgressColumn, SpinnerColumn, MofNCompleteColumn

PROGRESSBAR = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            MofNCompleteColumn()
        )

if __name__ == "__main__":
    datamodule = RadiationFieldDataModule(
        Path("C:/Users/lehner04/Documents/Datasets") / "H-100-Alderson-2_5m",
        batch_size=64,
        num_workers=4
    )
    datamodule.setup(stage="fit")

    min_fluence = 1.0
    max_fluence = 0.0
    min_x, min_y, min_z = 1.0, 1.0, 1.0
    max_x, max_y, max_z = -1.0, -1.0, -1.0
    dl = datamodule.train_dataloader()
    with PROGRESSBAR as progress:
        task = progress.add_task("Testing data loader...", total=len(dl))
        for batch in dl:
            data, metadata = batch
            spectrum, flux = data[:, :-1], data[:, -1]
            batch_min_fluence = torch.min(flux, dim=-1)[0]
            batch_max_fluence = torch.max(flux, dim=-1)[0]
            min_fluence = min(min_fluence, batch_min_fluence)
            max_fluence = max(max_fluence, batch_max_fluence)
            summed = torch.sum(spectrum, dim=-1)
            # test if all entries are close to 1.0
            if not torch.allclose(summed, torch.ones_like(summed), atol=1e-3):
                print(f"Summed spectrum is not close to 1.0 at all positions: {summed}")
                assert False, "Summed spectrum is not close to 1.0 at all positions"
            assert min_fluence >= 0.0, f"Error: Min Fluence: {min_fluence}"
            assert max_fluence <= 1.0, f"Error: Max Fluence: {max_fluence}"

            xyz = metadata[:, :3]
            abc = metadata[:, 3:]
            min_x = min(min_x, torch.min(xyz[:, 0]))
            min_y = min(min_y, torch.min(xyz[:, 1]))
            min_z = min(min_z, torch.min(xyz[:, 2]))
            max_x = max(max_x, torch.max(xyz[:, 0]))
            max_y = max(max_y, torch.max(xyz[:, 1]))
            max_z = max(max_z, torch.max(xyz[:, 2]))

            assert min_x >= -1.0, f"Error: Min X: {min_x}"
            assert min_y >= -1.0, f"Error: Min Y: {min_y}"
            assert min_z >= -1.0, f"Error: Min Z: {min_z}"

            assert max_x <= 1.0, f"Error: Max X: {max_x}"
            assert max_y <= 1.0, f"Error: Max Y: {max_y}"
            assert max_z <= 1.0, f"Error: Max Z: {max_z}"

            progress.update(task, advance=1)

    print(f"Min Fluence: {min_fluence}, Max Fluence: {max_fluence}")
    print("[green]Data loader passed tests!")
