import json
from pathlib import Path
import torch
from radfiled3d.pytorch.datasets.radfield3d import RadField3DDataset, RadField3DDatasetWithGeometry, RadField3DTranslationDataset
from radfiled3d.store import FieldStore
from radfiled3d import CartesianRadiationField
from enum import Enum
from radfiled3d.pytorch.datasets.processing import DataProcessing
from .crop_dataset import CropDataset
from .mc_floor_cut import MCFloorCut
from radfield3dnn.rftypes import TrainingInputData, rf3TrainingInputData, RadiationField, rf3RadiationField, RadiationFieldChannel, AirKermaField
import os
from radfield3dnn.preprocessing.normalizations.beam_parameters import BeamParametersNormalization
from rich import print
from .dataloader import RadiationFieldDataModule


class DatasetType(Enum):
    Voxelwise = 0
    Layerwise = 1


# Dataset feature composition lives in radfield3dnn.datasets.decorators: the plain
# RadField3DDataset is wrapped in input decorators (translation, geometry) so
# use_geometry / use_translation / use_beam_parameters combine freely — no per-combination
# subclasses.


class OriginalGroundTruthPreservation(DataProcessing):
    def clone_channel(self, channel: RadiationFieldChannel) -> RadiationFieldChannel:
        return RadiationFieldChannel(
            flux=channel.flux.clone() if channel.flux is not None else None,
            spectrum=channel.spectrum.clone() if channel.spectrum is not None else None,
            error=channel.error.clone() if channel.error is not None else None
        )

    def clone_radfield(self, field: RadiationField) -> RadiationField:
        return RadiationField(
            scatter_field=self.clone_channel(field.scatter_field) if field.scatter_field is not None else None,
            direct_beam=self.clone_channel(field.direct_beam) if field.direct_beam is not None else None,
            geometry=field.geometry.clone() if "geometry" in field._fields and field.geometry is not None else None
        )
    
    def clone(self, x: RadiationField | RadiationFieldChannel | AirKermaField | torch.Tensor) -> RadiationField | RadiationFieldChannel | AirKermaField | torch.Tensor:
        if isinstance(x, (RadiationField, rf3RadiationField)):
            return self.clone_radfield(x)
        elif isinstance(x, RadiationFieldChannel):
            return self.clone_channel(x)
        elif isinstance(x, torch.Tensor):
            return x.clone()
        elif isinstance(x, AirKermaField):
            return AirKermaField(
                air_kerma=x.air_kerma.clone(),
                geometry=x.geometry.clone() if x.geometry is not None else None
            )
        else:
            raise ValueError("Input must be of type RadiationField, RadiationFieldChannel, AirKermaField, or torch.Tensor")

    @staticmethod
    def canonicalize(gt):
        """Map RadFiled3D's native field types onto the repo's rftypes.

        ANTI-CORRUPTION BOUNDARY. RadFiled3D yields its own RadiationField (scatter_field,
        direct_beam) while the repo's RadiationField adds `geometry`; they are distinct
        NamedTuple classes, so every downstream `isinstance(gt, RadiationField)` in the losses,
        metrics, samplers and channel transforms silently evaluates False for a foreign instance
        and falls through to a wrong branch. Converting once, here at the dataset edge, is what
        keeps the rest of the pipeline single-typed.
        """
        if isinstance(gt, rf3RadiationField) and not isinstance(gt, RadiationField):
            return RadiationField(scatter_field=gt.scatter_field, direct_beam=gt.direct_beam,
                                  geometry=None)
        return gt

    def forward(self, x: TrainingInputData | RadiationField | RadiationFieldChannel | AirKermaField | torch.Tensor) -> TrainingInputData:
        if isinstance(x, (rf3TrainingInputData, TrainingInputData)):
            gt = self.canonicalize(x.ground_truth)
            return TrainingInputData(
                input=x.input,
                ground_truth=gt,
                original_ground_truth=self.clone(gt),
                # The directional target rides beside the channel (rftypes); rebuilding the tuple
                # here must not drop it, and a foreign item simply has none.
                lobes=getattr(x, "lobes", None),
            )
        else:
            raise ValueError("Input must be of type TrainingInputData")

    @classmethod
    def create_from_config(cls, config: dict) -> "OriginalGroundTruthPreservation":
        return OriginalGroundTruthPreservation()


def is_zipped_dataset(dataset_path: str) -> bool:
    """A dataset is either a directory of `.rf3` files or a single zip holding them."""
    return str(dataset_path).endswith(".zip")


def dataset_statistics_path(dataset_path: str) -> str:
    """Where `statistics.json` lives for this dataset.

    A zipped dataset keeps it BESIDE the archive (`<name>.statistics.json`) rather than inside:
    the file is regenerated and hand-edited, and rewriting a multi-gigabyte zip to change a few
    hundred bytes is not viable.
    """
    if is_zipped_dataset(dataset_path):
        return os.path.splitext(str(dataset_path))[0] + ".statistics.json"
    return os.path.join(str(dataset_path), "statistics.json")


def list_dataset_field_names(dataset_path: str) -> list[str]:
    """Every field in the dataset, as names `load_dataset_field` accepts."""
    if is_zipped_dataset(dataset_path):
        import zipfile
        with zipfile.ZipFile(str(dataset_path), "r") as z:
            return sorted(n for n in z.namelist() if n.endswith(".rf3"))
    fields_dir = os.path.join(str(dataset_path), "fields")
    search_dir = fields_dir if os.path.isdir(fields_dir) else str(dataset_path)
    return sorted(os.path.join(search_dir, f) for f in os.listdir(search_dir) if f.endswith(".rf3"))


def load_dataset_field(dataset_path: str, name: str) -> CartesianRadiationField:
    """Load one field by a name from `list_dataset_field_names`, from a directory or a zip."""
    if is_zipped_dataset(dataset_path):
        import zipfile
        with zipfile.ZipFile(str(dataset_path), "r") as z:
            with z.open(name) as f:
                return FieldStore.load_from_buffer(f.read())
    return FieldStore.load(name)


def get_dataset_dimensions_and_voxel_size(dataset: str | RadiationFieldDataModule) -> tuple[tuple[int, int, int], float]:
    dataset_path = None
    if isinstance(dataset, str):
        dataset_path = str(dataset)
        datamodule = RadiationFieldDataModule(
            Path(dataset_path),
            batch_size=1,
            num_workers=0,
            dataset_cls=RadField3DDataset,
            val_ratio=0.0,
            test_ratio=1.0,
            train_ratio=0.0
        )
        datamodule.prepare_data()
        dataset = datamodule
    elif not isinstance(dataset, RadiationFieldDataModule):
        raise TypeError(f"dataset must be a string path or a RadiationFieldDataModule instance, but got {type(dataset)}")
    
    if dataset_path is None:
        dataset_path = str(getattr(dataset, "zip_directory", ""))
    test_dl = dataset.test_dataloader()
    test_ds = test_dl.dataset
    test_files = test_ds.file_paths
    source = getattr(test_ds, "zip_file", None) or dataset_path
    test_field: CartesianRadiationField = load_dataset_field(source, test_files[0])
    field_dim = test_field.get_field_dimensions()
    vx_size_x = field_dim.x / test_field.get_voxel_counts().x
    vx_size_y = field_dim.y / test_field.get_voxel_counts().y
    vx_size_z = field_dim.z / test_field.get_voxel_counts().z
    assert vx_size_x == vx_size_y and vx_size_x == vx_size_z, f"Voxels are not isotropic! Voxel sizes: {vx_size_x}, {vx_size_y}, {vx_size_z}"
    return (field_dim.x, field_dim.y, field_dim.z), vx_size_x


def construct_datamodule(dataset_path: str, batch_size: int, num_workers: int, use_geometry: bool, use_beam_parameters: bool, dataprocessings: list[DataProcessing] = None, voxel_resolution: tuple[int, int, int] = None, prefetch_to_device: bool = True, max_fields: int = None, cache_to_ram: bool = False, cache_ram_gb: float = None, use_translation: bool = False, dataset_definition: str | dict = None, scan_translation_metadata: bool = True, skip_fields_without_translation: bool = False, attach_geometry_mask: bool = False, exclude_files: set = None, use_lobes: bool = False, translation_output_range: tuple = (0.0, 1.0), split: dict = None, geometry_layers: tuple = ("density",)) -> RadiationFieldDataModule:
    """dataset_definition: optional path to (or parsed dict of) the dataset definition JSON the
    dataset was GENERATED with (training config: ``dataset: definition_file``). When given, it is
    the authoritative source for the beam-parameter ranges (instead of statistics.json) and for
    the patient-translation ranges the TranslationNormalization needs with use_translation.

    exclude_files: fields to drop before the train/val/test split. Normally computed here by the
    translation-metadata preflight; passing it (together with scan_translation_metadata=False)
    replays a previous scan's verdict instead of re-reading every field's metadata.

    use_lobes: also load the radiation channel's von Mises-Fisher lobes onto every item
    (`TrainingInputData.lobes`) — the directional target, for a model with a lobe head.

    split: a fixed {"train", "val", "test"} assignment of dataset-relative field paths instead of
    the random split; fields it does not list are not used.

    geometry_layers: the geometry-channel layers use_geometry / attach_geometry_mask read, stacked
    channel-first onto ``input.geometry`` (default: the ``density`` layer only)."""
    if dataprocessings is None:
        dataprocessings = []
    # The datamodule carries the recipe that built it (see RadiationFieldDataModule.construction_spec)
    # so another PROCESS can rebuild an identical one — the Optuna tuner runs each trial in a fresh
    # subprocess. Captured HERE, before the derived processings below are prepended/appended, so a
    # replay adds them exactly once.
    _spec = dict(dataset_path=dataset_path, batch_size=batch_size, num_workers=num_workers,
                 use_geometry=use_geometry, use_beam_parameters=use_beam_parameters,
                 dataprocessings=list(dataprocessings), voxel_resolution=voxel_resolution,
                 prefetch_to_device=prefetch_to_device, max_fields=max_fields,
                 cache_to_ram=cache_to_ram, cache_ram_gb=cache_ram_gb,
                 use_translation=use_translation, dataset_definition=dataset_definition,
                 scan_translation_metadata=scan_translation_metadata,
                 skip_fields_without_translation=skip_fields_without_translation,
                 attach_geometry_mask=attach_geometry_mask, exclude_files=exclude_files,
                 use_lobes=use_lobes, translation_output_range=translation_output_range, split=split,
                 geometry_layers=tuple(geometry_layers))
    if dataset_definition is not None and not isinstance(dataset_definition, dict):
        definition_path = dataset_definition
        assert os.path.exists(definition_path), f"dataset definition file not found: {definition_path}"
        with open(definition_path, "r") as f:
            dataset_definition = json.load(f)
        print(f"[green]Loaded dataset definition from {definition_path}[/green]")
    # ── Input decorators: any combination of dataset features ─────────────────
    # The plain dataset provides the core input (direction, origin, spectrum, beam shape — which
    # is all use_beam_parameters needs); each decorator initializes exactly one optional
    # FieldInput member, so the flags compose freely. attach_geometry_mask loads the geometry
    # occupancy mask for GeometryVoxelExclusion even when use_geometry is off.
    from radfield3dnn.datasets.decorators import GeometryInputDecorator, TranslationInputDecorator
    if use_translation:
        print("[yellow]Input decorator: patient translation (from field metadata).")
    if use_geometry or attach_geometry_mask:
        print("[yellow]Input decorator: geometry occupancy mask (binary, from the geometry channel).")

    from radfield3dnn.datasets.lobes import LobedFieldDataset, PerFileAccessorDataset
    if use_lobes:
        print("[yellow]Loading the directional lobes (vmf_lobes) as a training target.")

    def dataset_cls(file_paths: list[str] = None, zip_file: str = None, data_processings: list["DataProcessing"] = None):
        # Per-file accessor always: identical output on a dataset with one byte layout, and the
        # only correct read of one whose generator wrote some metadata only for some fields.
        base = LobedFieldDataset if use_lobes else PerFileAccessorDataset
        ds = base(file_paths=file_paths, zip_file=zip_file, data_processings=data_processings)
        if use_translation:
            ds = TranslationInputDecorator(ds)
        if use_geometry or attach_geometry_mask:
            ds = GeometryInputDecorator(ds, binary=True, layers=geometry_layers)
        return ds

    stats = {}
    if os.path.exists(dataset_statistics_path(dataset_path)):
        statistics_path = dataset_statistics_path(dataset_path)
        with open(statistics_path, "r") as f:
            stats = json.load(f)
        print(f"[green]Loaded dataset statistics from {statistics_path}[/green]")
    else:
        print(f"[yellow]No dataset statistics found at {dataset_statistics_path(dataset_path)}[/yellow]")

    field_dim, vx_size = get_dataset_dimensions_and_voxel_size(dataset_path)
    vx_counts = (int(field_dim[0] / vx_size), int(field_dim[1] / vx_size), int(field_dim[2] / vx_size))
    assert abs(field_dim[0] - vx_size * vx_counts[0]) < 1e-8 and abs(field_dim[1] - vx_size * vx_counts[1]) < 1e-8 and abs(field_dim[2] - vx_size * vx_counts[2]) < 1e-8, f"Voxel dimensions do not match calculated voxel size! {field_dim} vs. {vx_size}"
    print(f"[blue]Dataset field dimensions: {field_dim}, voxel size: {vx_size} m, voxel counts: {vx_counts}[/blue]")

    if voxel_resolution is not None:
        print(f"[blue]Testing dataset with voxel resolution {voxel_resolution}[/blue]")
        assert vx_counts[0] >= voxel_resolution[0] and vx_counts[1] >= voxel_resolution[1] and vx_counts[2] >= voxel_resolution[2], f"Voxel resolution of dataset {vx_counts} does not match enforced resolution {voxel_resolution}"
        
        dataprocessings = [CropDataset(voxel_resolution)] + dataprocessings

        print(f"[green]Voxel resolution of dataset matches enforced resolution {voxel_resolution}!")

    if use_beam_parameters:
        if dataset_definition is not None:
            # The definition file the dataset was generated with is authoritative — it holds the
            # exact sampling ranges, so no statistics.json is required.
            print("[yellow]Using beam parameters normalization (ranges from dataset definition)!")
            beam_normalizer = BeamParametersNormalization.from_dataset_definition(
                dataset_definition, size_per_voxel_m=vx_size, is_origin_centered=False
            )
        else:
            # A missing statistics.json is only FATAL together with use_beam_parameters: the beam
            # normalizer needs the dataset's opening-angle / distance ranges. Without those keys, fail
            # loudly with a fixable message instead of a raw KeyError.
            required = ("tube_opening_angles_deg", "tube_distances_m")
            missing = [k for k in required if k not in stats]
            if missing:
                raise ValueError(
                    f"use_beam_parameters=True requires dataset statistics, but "
                    f"{dataset_statistics_path(dataset_path)} is missing or lacks keys {missing}. "
                    f"Generate it with scripts/compute_dataset_statistics.py, provide the dataset "
                    f"definition file (training config: dataset.definition_file), or disable use_beam_parameters."
                )
            print("[yellow]Using beam parameters normalization!")
            beam_normalizer = BeamParametersNormalization(
                opening_angle_range_deg=(
                    stats["tube_opening_angles_deg"]["Min"],
                    stats["tube_opening_angles_deg"]["Max"]
                ),
                size_per_voxel_m=vx_size,
                is_origin_centered=False,
                distance_range_m=(
                    stats["tube_distances_m"]["Min"],
                    stats["tube_distances_m"]["Max"]
                ),
                half_field_size=(field_dim[0]/2, field_dim[1]/2, field_dim[2]/2)
            )
        dataprocessings.append(beam_normalizer)

    if use_translation:
        # Normalize the patient translation to [0, 1] from the generation ranges. Requires the
        # dataset definition file — without it the translation reaches the network in raw metres
        # (previous behavior), which trains but couples the checkpoint to the dataset's extent.
        from radfield3dnn.preprocessing.normalizations.translation import TranslationNormalization
        if dataset_definition is not None and "GeometryTransformations" in dataset_definition:
            translation_normalizer = TranslationNormalization.from_dataset_definition(
                dataset_definition, output_range=translation_output_range)
            print(f"[yellow]Using translation normalization: {translation_normalizer.get_parameters()['translation_ranges_m']}")
            dataprocessings.append(translation_normalizer)
        else:
            print("[yellow]WARNING: use_translation without a dataset definition file "
                  "(dataset.definition_file in the training config) — patient translation stays "
                  "UN-normalized (raw metres).")

    # ── Translation-metadata preflight ────────────────────────────────────────
    # RadField3DTranslationDataset raises when a field carries no patient_translation entry. In a
    # dataset that mixes generation runs that happens at a RANDOM step inside a DataLoader worker,
    # minutes into training, naming no file. Scan the metadata (cheap: no field decode) up front.
    if use_translation and scan_translation_metadata:
        from radfield3dnn.datasets.translation_scan import (
            resolve_missing_translation, scan_fields_missing_translation, write_missing_report)
        # Read the file list the same way DataLoaderBuilder will (directory, or its fields/ subdir).
        field_dir = os.path.join(dataset_path, "fields")
        search_dir = field_dir if os.path.isdir(field_dir) else dataset_path
        candidates = [os.path.join(search_dir, f) for f in sorted(os.listdir(search_dir)) if f.endswith(".rf3")]
        print(f"[blue]Scanning {len(candidates)} field(s) for patient_translation metadata…[/blue]")
        missing = scan_fields_missing_translation(candidates, workers=num_workers)
        if missing:
            # Report EVERY offender: printed in full, and written one-per-line to the run's working
            # directory so the list can be fed straight to a cleanup command. Written BEFORE the
            # decision so it exists even when the run then fails.
            report_path = write_missing_report(missing)
            print(f"[red]{len(missing)} of {len(candidates)} field(s) carry NO patient_translation "
                  f"metadata (unreadable fields are listed here too):[/red]")
            for path in missing:
                print(f"  {path}")
            if report_path:
                print(f"[blue]Full list written to {report_path} "
                      f"(e.g. `xargs rm < {os.path.basename(report_path)}`).[/blue]")
            # Raises unless the run explicitly opts into training on the valid subset.
            exclude_files = resolve_missing_translation(
                dataset_path, candidates, missing, skip_fields_without_translation,
                report_path=report_path)
            print(f"[yellow]Those {len(missing)} field(s) are EXCLUDED from this run; "
                  f"{len(candidates) - len(missing)} remain.[/yellow]")
        else:
            print(f"[green]All {len(candidates)} fields carry patient_translation metadata.[/green]")

    datamodule = RadiationFieldDataModule(
        Path(dataset_path),
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_cls=dataset_cls,
        data_processings=dataprocessings,
        prefetch_to_device=prefetch_to_device,
        max_fields=max_fields,
        cache_to_ram=cache_to_ram,
        cache_ram_gb=cache_ram_gb,
        exclude_files=exclude_files,
        split=split,
    )
    # A replay resolves the definition to the parsed dict above; store THAT, so the second replay
    # does not re-read the file (and works when only the parsed ranges are still reachable).
    _spec["dataset_definition"] = dataset_definition
    _spec["exclude_files"] = exclude_files
    _spec["scan_translation_metadata"] = False
    datamodule.set_construction_spec(_spec)
    datamodule.prepare_data()

    return datamodule
