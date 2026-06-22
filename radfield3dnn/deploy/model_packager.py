"""Produce a self-contained RF3M deployment package from a trained model.

The package binds the exported PyTorch->ONNX graph to the model's *validity domain* (the valid
ranges of the beam parameters and the physical meaning, in metric units, of the normalised
inputs/outputs), lightweight provenance and the test metrics — everything a deployment (e.g. UE5)
needs to run the model and interpret its I/O without the training stack.

The byte layout is owned by the C++ side `rfnn::io::V1::ModelStore::save_to_memory`
(include/radfield3d-nn/model_io.h / src/RadField3DNN/model_io.cpp) — the single source of truth.
This module gathers the metadata (domain / provenance / metrics) and exports the ONNX graphs, then
hands them to that serialiser through the `rfnn_deploy` python bindings, so the format is never
re-implemented here.

A model generalises over a *range* of beam parameters, so we deliberately store that range and
the normalisation mappings (what the model's [0..1] source-distance scalar maps to in metres,
what the spectrum bins mean in eV) rather than any single simulation's X-ray-tube settings.
"""

from __future__ import annotations

import glob
import json
import os
import tempfile
from typing import Mapping

import torch

from radfield3dnn.models import ModelExporter


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

    The predicted spatial resolution / voxel geometry is intentionally NOT stored — it is chosen
    at inference and may vary across a dataset, so it is not a property of the model.
    """

    def __init__(self, model, datamodule, test_metrics: Mapping[str, object], *,
                 dataset_path: str, max_energy_eV: float = 1.5e5, spectra_bins: int = 32,
                 export_fp16: bool = False):
        self.model = model
        self.datamodule = datamodule
        self.metrics = {str(k): _metric_value(v) for k, v in dict(test_metrics or {}).items()}
        self.dataset_path = dataset_path
        self.max_energy_eV = float(max_energy_eV)
        self.spectra_bins = int(getattr(model, "out_spectra_dim", spectra_bins) or spectra_bins)
        # Store the exported ONNX graphs with fp16 weights (half the file size). The graph I/O is
        # kept fp32 (Cast nodes at the boundary), so the deploy runtime contract is unchanged — it
        # still feeds/reads float32. Intended for models trained in mixed precision, whose fp32
        # master weights carry no more than fp16 of useful signal anyway.
        self.export_fp16 = bool(export_fp16)

    # ── metadata gathering ────────────────────────────────────────────────────
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
        p = os.path.join(self.dataset_path, "statistics.json")
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f)
        return {}

    def _domain(self) -> dict:
        stats = self._statistics()

        def _range(key):
            d = stats.get(key, {})
            return (float(d.get("Min", 0.0)), float(d.get("Max", 0.0)))

        dist = _range("tube_distances_m")
        ang = _range("tube_opening_angles_deg")
        in_bins, in_max_ev = self._input_spectrum_layout()
        bin_width_ev = float(in_max_ev) / max(1, int(in_bins))
        # Ordered beam-parameter descriptors: (name, typed-range spec). Most parameters are a
        # ("minmax", min, max, unit) interval; the tube spectrum is a ("spectrum", min, max, bin_width,
        # unit) histogram range — its bins = (max-min)/bin_width, distinct from spectrum_bins (the
        # OUTPUT per-voxel histogram size).
        beam_parameters = [
            ("direction",     ("minmax", -1.0, 1.0, "")),
            ("distance",      ("minmax", dist[0], dist[1], "m")),
            ("opening_angle", ("minmax", ang[0], ang[1], "deg")),
            ("spectrum",      ("spectrum", 0.0, float(in_max_ev), bin_width_ev, "eV")),
        ]
        return dict(
            spectrum_bins=self.spectra_bins,
            spectrum_max_energy_ev=self.max_energy_eV,
            field_dimensions_m=self._field_dimensions_m(),
            beam_parameters=beam_parameters,
        )

    def _field_dimensions_m(self) -> list:
        """The training dataset's physical field box (metres) the normalised [0,1]^3 positions map
        into. Read from a sample .rf3 field; falls back to [0,0,0] (unknown) on any failure."""
        rf3 = self._sample_rf3()
        if rf3:
            try:
                from RadFiled3D.utils import FieldStore
                field = FieldStore.load(rf3)
                d = field.get_field_dimensions()   # glm::vec3, metres
                return [float(d.x), float(d.y), float(d.z)]
            except Exception as e:  # best-effort — geometry stays "unknown" if unavailable
                print(f"[yellow]ModelPackager: could not read field dimensions ({e})[/yellow]")
        return [0.0, 0.0, 0.0]

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
                from RadFiled3D.utils import FieldStore
                hdr = FieldStore.load_metadata(sample_rf3).get_header()
                software_version = str(getattr(getattr(hdr, "software", None), "version", "") or "")
                physics = str(getattr(getattr(hdr, "simulation", None), "physics_list", "") or "")
            except Exception as e:  # provenance is best-effort
                print(f"[yellow]ModelPackager: could not read provenance header ({e})[/yellow]")
        return dict(dataset_name=os.path.basename(os.path.normpath(self.dataset_path)),
                    software_version=software_version, physics=physics)

    # ── ONNX export ───────────────────────────────────────────────────────────
    def _export_bytes(self, export_fn) -> bytes:
        """Run an ONNX exporter (model -> path) to a temp file and return the bytes."""
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        tmp.close()
        try:
            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                export_fn(self.model, tmp.name)
            if was_training:
                self.model.train()
            with open(tmp.name, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

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
        """Named ONNX graphs composing the model. Per-voxel NeRF models export a two-graph
        pair — a 'beam_encoder' (beam parameters -> latent) and a 'trunk' (position + latent ->
        flux/spectrum) — so the deploy runtime encodes the beam once and reuses the latent across
        every voxel. Other models export a single 'trunk'. Keys match the rfnn k*Graph names.

        With ``export_fp16`` each graph's weights are stored as fp16 (I/O stays fp32)."""
        if ModelExporter.supports_two_graph_split(self.model):
            try:
                graphs = {
                    "beam_encoder": self._export_bytes(ModelExporter.onnx_export_beam_encoder),
                    "trunk":        self._export_bytes(ModelExporter.onnx_export_trunk),
                }
                return self._maybe_fp16(graphs)
            except Exception as e:
                print(f"[yellow]ModelPackager: two-graph export failed ({e}); using single trunk[/yellow]")
        # Field-wise/volume models ship their own self-contained beam->volume exporter; the per-voxel
        # ModelExporter.onnx_export wrapper does not apply to them.
        if hasattr(self.model, "export_onnx"):
            return self._maybe_fp16({"trunk": self._export_bytes(lambda m, p: m.export_onnx(p))})
        return self._maybe_fp16({"trunk": self._export_bytes(ModelExporter.onnx_export)})

    def _maybe_fp16(self, graphs: dict) -> dict:
        if not self.export_fp16:
            return graphs
        try:
            out = {name: self._to_fp16(ob) for name, ob in graphs.items()}
            print(f"[green]ModelPackager: exported graphs as fp16 "
                  f"({', '.join(f'{n} {len(graphs[n])//1024}->{len(out[n])//1024} KiB' for n in graphs)}).[/green]")
            return out
        except Exception as e:                              # never lose the package over fp16 conversion
            print(f"[yellow]ModelPackager: fp16 conversion failed ({e}); keeping fp32 graphs[/yellow]")
            return graphs

    def _rf3m_metadata(self):
        """Assemble the C++ ModelDomain/ModelProvenance from the gathered metadata + ONNX graphs.
        Imported lazily via the deploy loader (which preloads the matching ONNX Runtime), so merely
        importing this module never requires the compiled deploy bindings."""
        from radfield3dnn.deploy.onnx_runtime import rfnn_deploy as rd
        domain = self._domain()
        prov = self._provenance(self._sample_rf3())
        def _mk_range(spec):
            kind = spec[0]
            if kind == "minmax":
                _, mn, mx, unit = spec
                return rd.ParameterRange.min_max(float(mn), float(mx), str(unit))
            if kind == "spectrum":
                _, mn, mx, bw, unit = spec
                return rd.ParameterRange.spectrum(float(mn), float(mx), float(bw), str(unit))
            raise ValueError(f"unknown range kind {kind!r}")

        rd_domain = rd.ModelDomain(
            spectrum_bins=int(domain["spectrum_bins"]),
            spectrum_max_energy_ev=float(domain["spectrum_max_energy_ev"]),
            field_dimensions_m=[float(x) for x in domain["field_dimensions_m"]],
            beam_parameters=[
                rd.BeamParameterSpec(str(name), _mk_range(spec))
                for name, spec in domain["beam_parameters"]
            ],
        )
        rd_prov = rd.ModelProvenance(dataset_name=prov["dataset_name"],
                                     software_version=prov["software_version"], physics=prov["physics"])
        metrics = {str(k): float(v) for k, v in self.metrics.items()}
        return rd, rd_domain, rd_prov, metrics

    def to_bytes(self) -> bytes:
        rd, domain, prov, metrics = self._rf3m_metadata()
        return rd.save_to_memory(self._graphs(), domain, prov, metrics)

    def save(self, path: str) -> str:
        rd, domain, prov, metrics = self._rf3m_metadata()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        rd.save(path, self._graphs(), domain, prov, metrics)   # C++ writes the bytes
        print(f"[green]Wrote RF3M model package -> {path} ({os.path.getsize(path)/1e6:.2f} MB)[/green]")
        return path
