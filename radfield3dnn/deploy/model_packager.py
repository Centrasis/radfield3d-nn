"""Produce a self-contained RF3M deployment package from a trained model.

The package binds the exported PyTorch->ONNX graphs to the model's *validity domain* (the valid
ranges of the beam parameters and the physical meaning, in metric units, of the normalised
inputs/outputs), lightweight provenance and the test metrics — everything a deployment (e.g. UE5)
needs to run the model and interpret its I/O without the training stack.

The container format is owned by the `radfiled3d-nn` module (`RadFiled3D::nn::deploy`), reached
through its `ModelBuilder`; nothing about the byte layout is known here. This module's job is the
part only the training stack can do: gather the metadata, export the graphs, and say what each
tensor MEANS.

Two facts decide how a package is assembled:

* **The model declares its tensors, the graphs decide which exist.** `deploy_interface()` maps a
  graph tensor name to its semantic/shape/unit; tracing decides which of them the graph actually
  carries. The declaration is the intersection, and a graph tensor no model described raises —
  the runtime binds BY NAME, so a package that declares a name no graph provides loads and then
  cannot be driven.
* **A range is the dataset's, not the model's.** The distances, energies and collimations the model
  generalises over come from `statistics.json`, so they are attached here rather than by the model.

A model generalises over a *range* of beam parameters, so that range and the normalisation mappings
(what the [0..1] source-distance scalar means in metres, what the spectrum bins mean in eV) are
stored rather than any single simulation's X-ray-tube settings.
"""

from __future__ import annotations

import datetime
import glob
import json
import os
from typing import Mapping, Sequence

import numpy as np
import torch

from radfield3dnn.models import ModelExporter, staged_encoders


def _metric_value(v) -> float:
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return float(v)


class ModelPackager:
    """Gather metadata + export ONNX + write an RF3M package.

    Parameters
    ----------
    model            : the trained LightningModule (a BaseNeuralRadFieldModel).
    datamodule       : the data module used for training/testing (for a sample .rf3 + provenance).
    test_metrics     : mapping of metric name -> value (floats or 0-dim tensors), e.g. trainer.callback_metrics.
    dataset_path     : dataset root (holds statistics.json with the parameter envelope).
    max_energy_eV    : top of the spectrum energy range (bin i spans [i, i+1)*max/bins eV).
    spectra_bins     : number of spectrum histogram bins.

    The field GEOMETRY — the metric box and the voxel grid the fields were sampled on — is stored,
    and both halves are mandatory. The grid is a property of the model: the network was fitted to
    voxel values at one resolution, with nothing learned about what lies between two of them, so a
    box on its own does not say what the weights are meaningful at. The container refuses a package
    that carries only one of the two.
    """

    def __init__(self, model, datamodule, test_metrics: Mapping[str, object], *,
                 dataset_path: str, max_energy_eV: float = 1.5e5, spectra_bins: int = 32,
                 export_fp16: bool = False,
                 field_geometry: tuple[Sequence[float], Sequence[int]] | None = None):
        self.model = model
        self.datamodule = datamodule
        self.metrics = {str(k): _metric_value(v) for k, v in dict(test_metrics or {}).items()}
        self.dataset_path = dataset_path
        # ([x, y, z] metres, [nx, ny, nz] voxels) for a caller that cannot be read off the dataset —
        # a model packaged away from its fields, or a test with no .rf3 at all. Default: read from
        # the dataset, which is the only thing that KNOWS what the model was trained on, so pass
        # this only when that read is impossible, never to override it.
        self.field_geometry = field_geometry
        self.max_energy_eV = float(max_energy_eV)
        self.spectra_bins = int(getattr(model, "out_spectra_dim", spectra_bins) or spectra_bins)
        # Store the exported ONNX graphs with fp16 weights (half the file size). The graph I/O is
        # kept fp32 (Cast nodes at the boundary), so the deploy runtime contract is unchanged — it
        # still feeds/reads float32. Intended for models trained in mixed precision, whose fp32
        # master weights carry no more than fp16 of useful signal anyway.
        self.export_fp16 = bool(export_fp16)

    # ── metadata gathering ────────────────────────────────────────────────────
    def _dataset_layer_unit(self, layer: str) -> str:
        """The unit the training fields give `layer`, read from one of them; '' when unreadable."""
        path = self._sample_rf3()
        if path is None:
            return ""
        try:
            from radfiled3d.store import FieldStore
            field = FieldStore.load(path)
            for channel in ("scatter_field", "direct_beam"):
                if channel in field.get_channel_names():
                    ch = field.get_channel(channel)
                    if ch.has_layer(layer):
                        return ch.get_layer_unit(layer)
        except Exception as e:
            print(f"[yellow]ModelPackager: could not read the unit of `{layer}` from {path} ({e})[/yellow]")
        return ""

    def _output_definitions(self, declared_out: dict) -> dict:
        """Say what each output's numbers ARE, in its unit — the graph emits them as they are.

        The flux is in the dataset's own unit when the export took it back there (a normalizer
        with one `dataset_scale`, see `models.in_final_units`), and relative to the field's maximum
        when the model was trained on per-field normalized targets: no constant can undo those, so
        the honest unit is the definition. A per-field normalizer other than `linear0_1` leaves
        numbers nothing here can describe, and is refused rather than exported unlabelled.

        The spectrum is the per-voxel fluence distribution over energy bins, normalized to sum 1 —
        a histogram output whose bins are as wide as the training spectra's (in eV).
        """
        out = {name: dict(desc) for name, desc in declared_out.items()}
        normalizer = getattr(self.model, "_normalizer", None)
        kind = normalizer.get_type() if normalizer is not None else "identity"
        if "flux" in out:
            if getattr(normalizer, "dataset_scale", None) is not None or kind == "identity":
                unit = self._dataset_layer_unit("flux")
                if not unit:
                    raise ValueError(
                        "the exported flux is in the dataset's units, but no training field says "
                        "which unit that is; the package cannot declare it.")
                out["flux"]["unit"] = unit
            elif kind == "linear0_1":
                out["flux"]["unit"] = "fraction of the field's maximum flux"
            else:
                raise ValueError(
                    f"the flux was trained under `{kind}` fitted per field, so the exported numbers "
                    f"are neither in the dataset's units nor a fraction of the field's maximum. "
                    f"Train with a dataset_scale, or with linear0_1, to export this model.")
        if "spectrum" in out:
            from radfiled3d.nn.records import HistogramOutput
            out["spectrum"]["unit"] = "fraction of the voxel's flux per energy bin"
            bins = int(out["spectrum"]["shape"][-1])
            _, max_ev = self._input_spectrum_layout()
            out["spectrum"]["output_type"] = HistogramOutput(float(max_ev) / max(1, bins), "eV")
        return out

    def _sample_rf3(self) -> str | None:
        try:
            ds = self.datamodule.test_dataloader().dataset
            fp = getattr(ds, "file_paths", None)
            if fp:
                return fp[0]
        except Exception:
            pass
        hits = glob.glob(os.path.join(self.dataset_path, "**", "*.rf3"), recursive=True)
        return hits[0] if hits else None

    def _statistics(self) -> dict:
        from radfield3dnn.datasets import dataset_statistics_path
        p = dataset_statistics_path(self.dataset_path)
        if os.path.exists(p):
            # These ranges become the DEPLOYED model's validity domain. A statistics.json older
            # than the newest field file means the dataset changed after the stats were computed
            # (e.g. densification) and the shipped envelope would be wrong.
            fields_dir = os.path.join(self.dataset_path, "fields")
            if os.path.isdir(fields_dir):
                newest_field = max((e.stat().st_mtime for e in os.scandir(fields_dir) if e.is_file()),
                                   default=0.0)
                if os.path.getmtime(p) < newest_field:
                    print(f"[red]{p} is OLDER than the newest field file — the packaged domain "
                          f"metadata is stale. Re-run scripts/compute_dataset_statistics.py.[/red]")
            with open(p, "r") as f:
                return json.load(f)
        return {}

    def _ranges(self, shapes: dict) -> dict:
        """Training envelope per SEMANTIC, from the dataset statistics.

        Keyed by semantic rather than tensor name because the range is a property of the physical
        quantity: whichever graph input carries the source distance gets the same metres.

        The tube spectrum's bin WIDTH is derived from the traced input width in ``shapes``, not
        from the dataset's bin count: the graph is what a consumer feeds, and a model that rebins
        internally still declares its own input contract.
        """
        stats = self._statistics()

        def _minmax(key):
            d = stats.get(key, {})
            return (float(d.get("Min", 0.0)), float(d.get("Max", 0.0)))

        in_bins, in_max_ev = self._input_spectrum_layout()
        # The LAST axis is the bin count: a per-field input is declared `[1, bins]`, and reading
        # the leading axis would call the whole spectrum one bin and hand a consumer a bin width of
        # the full energy span.
        in_bins = int(shapes.get("tube_spectrum", [in_bins])[-1])
        return {
            # Unit vector: each component is already its own range.
            "beam_direction": (-1.0, 1.0),
            "source_distance": _minmax("tube_distances_m"),
            "beam_collimation": _minmax("tube_opening_angles_deg"),
            # A histogram axis, not an interval: the third element is the bin width, so the bin
            # count follows from (max - min) / width. Distinct from the OUTPUT spectrum's bins.
            "tube_spectrum": (0.0, float(in_max_ev), float(in_max_ev) / max(1, in_bins)),
        }

    def _field_geometry(self) -> tuple[list, list]:
        """The training field's metric box and voxel grid: ([x, y, z] metres, [nx, ny, nz] voxels).

        Both come from the dataset — the only thing that knows them — unless the caller passed a
        ``field_geometry`` because there is no dataset to read. The box is what the normalised
        [0,1]^3 positions map into, and the grid is the resolution the model was trained at. A crop
        (``voxel_resolution``) replaces the grid with what the network actually saw, and the box
        shrinks with it: the voxel size is the dataset's either way, never hard-coded here.

        Raises rather than guessing. The container demands both halves, so an unknown geometry
        cannot be written as zeros any more — a package carrying them would be refused on load,
        which is a worse failure than this one and much further from its cause.
        """
        if self.field_geometry is not None:
            box, counts = [float(v) for v in self.field_geometry[0]], \
                          [int(v) for v in self.field_geometry[1]]
        else:
            box, counts = self._dataset_geometry()
        crop = self._training_voxel_resolution()
        if crop is not None and list(crop) != counts:
            # Cropping keeps the voxel SIZE and drops voxels, so the box the model spans is smaller
            # than the dataset's by exactly the voxels that were cut.
            voxel_m = [box[i] / counts[i] for i in range(3)]
            counts = [int(crop[i]) for i in range(3)]
            box = [voxel_m[i] * counts[i] for i in range(3)]
            print(f"[yellow]ModelPackager: trained on a {counts} crop, so the package states "
                  f"that grid over {box} m, not the dataset's full field[/yellow]")
        return box, counts

    def _dataset_geometry(self) -> tuple[list, list]:
        """The dataset's own box and grid, through the one reader that knows every dataset layout.

        `get_dataset_dimensions_and_voxel_size` is the datasets package's accessor — it handles a
        zipped dataset as well as a directory and asserts the voxels are isotropic — so the read is
        not repeated here. The counts follow from box / voxel size, which is the only direction this
        repo derives that relation in."""
        from radfield3dnn.datasets import get_dataset_dimensions_and_voxel_size
        try:
            dims, voxel_m = get_dataset_dimensions_and_voxel_size(self.datamodule)
        except Exception as from_datamodule:
            try:
                dims, voxel_m = get_dataset_dimensions_and_voxel_size(self.dataset_path)
            except Exception as from_path:
                raise RuntimeError(
                    "ModelPackager: the field box and the training resolution could not be read "
                    f"from the datamodule ({from_datamodule}) or from dataset_path="
                    f"{self.dataset_path!r} ({from_path}), and a package must state both. Point it "
                    "at the dataset the model was trained on, or pass field_geometry=(box, counts) "
                    "if there is none to read.") from from_path
        box = [float(v) for v in dims]
        counts = [int(round(v / voxel_m)) for v in box]
        for edge, n in zip(box, counts):
            if n <= 0 or abs(edge - n * voxel_m) > 1e-8:
                raise RuntimeError(
                    f"ModelPackager: a {box} m field does not divide into {voxel_m} m voxels "
                    f"({counts}); the dataset's geometry is inconsistent.")
        return box, counts

    def _training_voxel_resolution(self):
        """The enforced grid this run trained on (``voxel_resolution``), or None for the dataset's.

        Read from the datamodule's own construction spec rather than inferred from its processings:
        the spec is the recipe it was built from, and it is what another process replays."""
        try:
            # A method, and it raises for a datamodule built by hand rather than by
            # `construct_datamodule` — which is exactly the case where the crop is unknowable.
            spec = self.datamodule.construction_spec()
        except Exception:
            return None
        resolution = (spec or {}).get("voxel_resolution")
        return tuple(resolution) if resolution else None

    def _input_spectrum_layout(self) -> tuple[int, float]:
        """The model's INPUT tube-spectrum histogram size + energy span (eV) — distinct from the
        OUTPUT per-voxel histogram bins. Read from the dataset (input_spectra_bins /
        max_spectrum_energy_eV); fall back to the configured values."""
        bins, max_ev = None, None
        try:
            ds = self.datamodule.test_dataloader().dataset
            bins = getattr(ds, "input_spectra_bins", None)
            max_ev = getattr(ds, "max_spectrum_energy_eV", None)
        except Exception:
            pass
        return int(bins if bins is not None else self.spectra_bins), \
               float(max_ev if max_ev is not None else self.max_energy_eV)

    def _provenance(self, sample_rf3: str | None) -> dict:
        software_version, physics = "", ""
        if sample_rf3 is not None:
            try:
                from radfiled3d.store import FieldStore
                hdr = FieldStore.load_metadata(sample_rf3).get_header()
                software_version = str(getattr(getattr(hdr, "software", None), "version", "") or "")
                physics = str(getattr(getattr(hdr, "simulation", None), "physics_list", "") or "")
            except Exception as e:  # provenance is best-effort
                print(f"[yellow]ModelPackager: could not read provenance header ({e})[/yellow]")
        return dict(dataset_name=os.path.basename(os.path.normpath(self.dataset_path)),
                    software_version=software_version, physics=physics)

    # ── ONNX export ───────────────────────────────────────────────────────────
    def _export(self, export_fn):
        """Run an ONNX exporter on CPU and return its ExportedGraph.

        torch.export traces the ops the CURRENT device dispatches to, and CUDA-specialized kernels
        (cudnn_*) are not all fake-tensor-traceable. The export needs no GPU; weights are identical
        either way. The model is restored afterwards so the test-end callback leaves it where it
        found it.
        """
        was_training = self.model.training
        prior_device = next(self.model.parameters()).device
        self.model.eval().cpu()
        try:
            with torch.no_grad():
                return export_fn(self.model)
        finally:
            self.model.to(prior_device)
            if was_training:
                self.model.train()

    @staticmethod
    def _to_fp16(onnx_bytes: bytes) -> bytes:
        """Convert an exported fp32 ONNX graph to fp16 weights, keeping the graph I/O in fp32.

        ``keep_io_types=True`` makes the converter wrap the inputs/outputs with Cast nodes, so the
        external interface (the tensors the deploy runtime feeds and reads) stays float32 while the
        weights and internal compute drop to fp16 — halving the stored graph. ``max_finite_val`` is
        raised to the true fp16 ceiling (65504) so values that genuinely fit in fp16 are not clamped
        (the library's 1e4 default would needlessly truncate them)."""
        from onnxconverter_common.float16 import convert_float_to_float16  # optional, lazy import
        import onnx
        model = onnx.load_from_string(onnx_bytes)
        model16 = convert_float_to_float16(model, keep_io_types=True, max_finite_val=65504.0)
        return model16.SerializeToString()

    def _graphs(self) -> dict:
        """Named ONNX graphs composing the model, as {name: ExportedGraph}.

        Per-voxel NeRF models export a two-graph pair — a 'beam_encoder' (beam parameters -> latent)
        and a 'trunk' (position + latent -> flux/spectrum) — so a consumer encodes the beam once and
        reuses the latent across every voxel. Other models export a single 'trunk', which is
        mandatory in every package.

        With ``export_fp16`` each graph's weights are stored as fp16 (I/O stays fp32).
        """
        if ModelExporter.supports_two_graph_split(self.model):
            try:
                graphs = {
                    "beam_encoder": self._export(ModelExporter.onnx_export_beam_encoder),
                    "trunk":        self._export(ModelExporter.onnx_export_trunk),
                }
                # A resolution-aware location encoder gives the trunk a third input, `region_state`,
                # whose value depends on the grid the consumer picks. The graph that produces it
                # ships with the package, so the consumer can re-run it per LOD change instead of
                # having the scale baked in as a constant.
                if "region_state" in graphs["trunk"].input_names:
                    graphs["encoding_config"] = self._export(
                        ModelExporter.onnx_export_encoding_config)
                return self._maybe_fp16(graphs)
            except Exception as e:
                print(f"[yellow]ModelPackager: two-graph export failed ({e}); using single trunk[/yellow]")
        # A whole-volume model (the field-wise CNNs) ships its own beam->volume exporter; the
        # per-voxel wrapper does not apply to it.
        if hasattr(self.model, "export_onnx"):
            return self._maybe_fp16({"trunk": self._export(lambda m: m.export_onnx())})
        # The trunk is exported FIRST even though it runs last: a stage-source graph has to know
        # which of its inputs the trunk also takes, so it can relay those through buffers instead
        # of both stages claiming the same caller binding.
        trunk = self._export(ModelExporter.onnx_export)
        graphs = {}
        # An encoder with no ONNX lowering ships as a kernel stage, and a kernel stage has no
        # caller-bound tensors — so the graph computing its input has to ship too, ahead of it.
        for entry in staged_encoders(self.model):
            graphs[f"{entry.stage}_source"] = self._export(
                lambda m, e=entry: ModelExporter.onnx_export_stage_source(
                    m, e, relay=trunk.input_names))
        graphs["trunk"] = trunk
        return self._maybe_fp16(graphs)

    def _maybe_fp16(self, graphs: dict) -> dict:
        if not self.export_fp16:
            return graphs
        try:
            out = {name: g._replace(onnx=self._to_fp16(g.onnx)) for name, g in graphs.items()}
            print(f"[green]ModelPackager: exported graphs as fp16 "
                  f"({', '.join(f'{n} {len(graphs[n].onnx)//1024}->{len(out[n].onnx)//1024} KiB' for n in graphs)}).[/green]")
            return out
        except Exception as e:                              # never lose the package over fp16 conversion
            print(f"[yellow]ModelPackager: fp16 conversion failed ({e}); keeping fp32 graphs[/yellow]")
            return graphs

    def _composition(self, graphs: dict):
        """Buffers, stages and kernel blocks for the model's staged encoders.

        Returns `(buffers, stages, kernels, wired)`. `stages` is the EXECUTION ORDER the package
        records: every stage-source graph, then its kernel, and `trunk` last — a stage may only
        read what an earlier one wrote, and the trunk reads all of them.

        `wired` is the graph tensors that exist only between stages: the coordinate, the encoded
        features, and the RELAYED copies. Those are declared on neither side of the package. A
        relayed tensor's original name is NOT wired — the source stage still takes it from the
        caller, and that is the one binding the package declares.

        Empty for a model with no staged encoder, which is then packaged exactly as before.
        """
        staged = staged_encoders(self.model)
        if not staged:
            return [], [], [], set()

        buffers, stages, kernels, wired = [], [], [], set()
        trunk_reads = {}
        for entry in staged:
            source = graphs[f"{entry.stage}_source"]
            coordinate = source.output_names[0]
            features = entry.tensor
            buffers += [(coordinate, entry.encoder.in_dim),
                        (features, entry.encoder.encoded_dims)]
            writes = {coordinate: coordinate}
            shapes = self._input_shapes({f"{entry.stage}_source": source})
            for relayed in source.output_names[1:]:
                original = relayed[: -len("_relay")]
                buffers.append((relayed, int(np.prod(shapes[original]))))
                writes[relayed] = relayed
                trunk_reads[original] = relayed
                wired.add(relayed)
            stages.append((f"{entry.stage}_source", {}, writes))
            # The kernel's own parameter names, not the buffers': `positions` and `encoded` are
            # what hashgrid_kernel.py compiled, and the ports map one onto the other.
            stages.append((entry.stage, {"positions": coordinate}, {"encoded": features}))
            trunk_reads[features] = features
            payload, elements, dtype = entry.encoder.deploy_stage_weights()
            kernels.append(dict(stage=entry.stage,
                                blocks=entry.encoder.deploy_stage_blocks(),
                                launches=entry.encoder.deploy_stage_launch(),
                                weights=(payload, elements, dtype)))
            wired.update({coordinate, features})
        stages.append(("trunk", trunk_reads, {}))
        return buffers, stages, kernels, wired

    # ── assembly ──────────────────────────────────────────────────────────────
    def _build(self, graphs: dict):
        """Assemble the ModelBuilder: provenance, box, declared tensors, graphs, metrics."""
        from radfiled3d.nn import deploy as rf3m

        prov = self._provenance(self._sample_rf3())
        builder = rf3m.ModelBuilder(
            dataset=prov["dataset_name"],
            software=f"radfield3d-nn ({prov['software_version']})" if prov["software_version"]
                     else "radfield3d-nn",
            physics=prov["physics"],
            created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        )
        box_m, voxel_counts = self._field_geometry()
        builder.set_field_dimensions(box_m, voxel_counts=voxel_counts)

        declared_in, declared_out = self.model.deploy_interface()
        declared_out = self._output_definitions(declared_out)
        shapes = self._input_shapes(graphs)
        ranges = self._ranges(shapes)
        graph_in, graph_out = set(), set()
        for g in graphs.values():
            graph_in.update(g.input_names)
            graph_out.update(g.output_names)
        # A tensor one graph produces and another consumes (the beam latent, the region state) is
        # plumbing between the graphs, not a result, so it is not declared as an output.
        graph_out -= graph_in
        # A tensor the composition attaches to a buffer is carried between stages and never
        # reaches a consumer, so it is declared on neither side: the runtime fills it, and asking
        # the caller to bind it would be asking for something it cannot know.
        buffers, stages, kernels, wired = self._composition(graphs)

        # The graphs go on BEFORE the declarations, because what the declarations must leave out
        # depends on how the graphs connect, and working that out means reading them.
        for name, g in graphs.items():
            builder.add_graph(name, g.onnx)

        # A model of several graphs with no staged encoder — the beam encoder and the trunk — is
        # wired by asking the graphs: the container refuses a package that ships several graphs and
        # says nothing about how they connect, and a consumer of one could only run the trunk.
        #
        # The plan is stated, not guessed: an encoder describes the BEAM, which is constant over the
        # volume, so it runs once per field and the runtime repeats what it wrote across the queries.
        if not stages and len(graphs) > 1:
            plan = [(name, "once") for name in graphs if name != "trunk"] + [("trunk", "per_query")]
            edges = builder.compose(plan)
            wired = wired | {edge.consumed_tensor for edge in edges}
            for edge in edges:
                print(f"[green]ModelPackager: {edge.producer}.{edge.produced_tensor} -> "
                      f"[{edge.consumed_tensor}] {edge.width} wide -> {edge.consumer}."
                      f"{edge.consumed_tensor} (once per field)[/green]")

        graph_in -= wired
        graph_out -= wired

        def _declare(add, names, table, role):
            unknown = sorted(n for n in names if n not in table)
            if unknown:
                raise ValueError(
                    f"the exported graphs carry {role}s {unknown} that deploy_interface() does not "
                    f"describe. The runtime binds by name, so a consumer would have no way to know "
                    f"what to put there. Describe them on the model, or stop tracing them.")
            for name in sorted(names):
                desc = dict(table[name])
                stale = sorted(k for k in ("normalizer", "normalizer_params") if k in desc)
                if stale:
                    raise ValueError(
                        f"deploy_interface() declares {stale} for the {role} '{name}'. A network "
                        f"normalizes its inputs and emits final units itself, so a package states no "
                        f"normalizer; do the transform inside the exported graph and drop the keys.")
                desc.setdefault("range", ranges.get(desc["semantic"]))
                if role == "input":
                    desc["shape"] = shapes[name]
                add(name, **desc)

        _declare(builder.add_input, graph_in, declared_in, "input")
        _declare(builder.add_output, graph_out, declared_out, "output")

        for k, v in self.metrics.items():
            builder.add_metric(k, float(v))

        # Buffers first, then the stages IN EXECUTION ORDER.
        for name, elements in buffers:
            builder.add_buffer(name, elements)
        for name, reads, writes in stages:
            builder.add_stage(name, "per_query", reads=reads, writes=writes)
        for kernel in kernels:
            payload, elements, dtype = kernel["weights"]
            builder.add_stage_weights(kernel["stage"], payload, elements, dtype=dtype)
            for kind, blob in kernel["blocks"].items():
                builder.add_block(kind, kernel["stage"], blob)
            for kind, launch in kernel["launches"].items():
                builder.add_stage_launch(kernel["stage"], kind, **launch)
        return builder

    @staticmethod
    def _input_shapes(graphs: dict) -> dict:
        """Per-query shape of every graph input, read from the graphs themselves.

        The DYNAMIC leading axis is the batch and is dropped; what remains is what one query
        supplies. Dropping it unconditionally would be wrong for a per-GRID input like
        `region_state`, which is exported with no batch axis at all — its whole shape is the value.
        Taken from the graph rather than from the model because the graph is what the runtime will
        bind against; a shape the model believes and the trace disagrees with is a binding failure
        in the field.

        Outputs get their shape from the model instead: the dynamo exporter names the output
        dimensions dynamically and the graph reports no usable shape for them.
        """
        import onnx
        shapes = {}
        for g in graphs.values():
            model = onnx.load_from_string(g.onnx)
            for t in model.graph.input:
                dims = list(t.type.tensor_type.shape.dim)
                if dims and dims[0].WhichOneof("value") == "dim_param":
                    dims = dims[1:]
                shapes[t.name] = [int(d.dim_value) for d in dims] or [1]
        return shapes

    def to_bytes(self) -> bytes:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="rf3m-") as tmp:
            path = os.path.join(tmp, "model.rf3m")
            self._write(path)
            with open(path, "rb") as f:
                return f.read()

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._write(path)
        print(f"[green]Wrote RF3M model package -> {path} ({os.path.getsize(path)/1e6:.2f} MB)[/green]")
        return path

    def _write(self, path: str) -> None:
        graphs = self._graphs()
        builder = self._build(graphs)
        # Every declared tensor must appear in some graph. Checked across the graphs TOGETHER,
        # because a factored model splits its inputs between the encoder and the trunk.
        builder.verify_declarations()
        from radfiled3d.nn import deploy as rf3m
        rf3m.ModelStore.store(builder.build(), path)
        self._verify_readback(path, graphs)

    def _verify_readback(self, path: str, graphs: dict) -> None:
        """Read the package back and check it is what we meant to write.

        Writing an RF3M is the last step of training, so a package that is subtly wrong is not
        noticed until someone deploys it. Reading it back costs one metadata parse and catches the
        failure that matters: the graphs and the declared tensors on disk disagreeing with what the
        model announced, which a consumer only discovers when it cannot bind.
        """
        from radfiled3d.nn import deploy as rf3m
        meta = rf3m.ModelStore.load_metadata(path)
        stored_graphs = {b.name for b in meta.blocks if b.kind == "onnx"}
        if stored_graphs != set(graphs):
            raise RuntimeError(
                f"RF3M read-back failed for {path}: graphs on disk {sorted(stored_graphs)} != "
                f"{sorted(graphs)} as written.")
        stored_in = {t.name for t in meta.tensors if t.role == "input"}
        stored_out = {t.name for t in meta.tensors if t.role == "output"}
        graph_in, graph_out = set(), set()
        for g in graphs.values():
            graph_in.update(g.input_names)
            graph_out.update(g.output_names)
        missing = sorted((stored_in | stored_out) - (graph_in | graph_out))
        if missing:
            raise RuntimeError(
                f"RF3M read-back failed for {path}: the package declares {missing}, which no "
                f"shipped graph carries — the runtime binds by name and could not drive it.")
        print(f"[green]Verified RF3M read-back: graphs {sorted(stored_graphs)}, "
              f"{len(stored_in)} inputs, {len(stored_out)} outputs, "
              f"{len(meta.metrics)} metrics[/green]")
