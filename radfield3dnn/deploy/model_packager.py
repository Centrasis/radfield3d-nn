"""Produce a self-contained RF3M deployment package from a trained model.

The package binds the exported PyTorch->ONNX graph to the model's *validity domain* (the valid
ranges of the beam parameters and the physical meaning, in metric units, of the normalised
inputs/outputs), lightweight provenance and the test metrics — everything a deployment (e.g. UE5)
needs to run the model and interpret its I/O without the training stack.

The binary layout is defined and consumed by the C++ side `rfnn::io::V1::ModelFactory`
(include/radfield3d-nn/model_io.h / src/RadField3DNN/model_io.cpp) — that header is the
authoritative format. This module writes the identical layout (RF3M version 2).

A model generalises over a *range* of beam parameters, so we deliberately store that range and
the normalisation mappings (what the model's [0..1] source-distance scalar maps to in metres,
what the spectrum bins mean in eV) rather than any single simulation's X-ray-tube settings.
"""

from __future__ import annotations

import glob
import json
import os
import struct
import tempfile
from typing import Mapping

import torch

from radfield3dnn.models import ModelExporter

RF3M_MAGIC = b"RF3M"
RF3M_VERSION = 2


def _u32(v: int) -> bytes:
    return struct.pack("<I", int(v))


def _str(s: str) -> bytes:
    b = (s or "").encode("utf-8")
    return _u32(len(b)) + b


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
                 dataset_path: str, max_energy_eV: float = 1.5e5, spectra_bins: int = 32):
        self.model = model
        self.datamodule = datamodule
        self.metrics = {str(k): _metric_value(v) for k, v in dict(test_metrics or {}).items()}
        self.dataset_path = dataset_path
        self.max_energy_eV = float(max_energy_eV)
        self.spectra_bins = int(getattr(model, "out_spectra_dim", spectra_bins) or spectra_bins)

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
        # Ordered beam-parameter descriptors: (name, slot count in the input vector, range_min,
        # range_max, unit) — the layout of the model's beam-parameter input vector.
        beam_parameters = [
            ("direction",     3, -1.0, 1.0, ""),
            ("distance",      1, dist[0], dist[1], "m"),
            ("opening_angle", 1, ang[0], ang[1], "deg"),
            ("spectrum",      int(self.spectra_bins), 0.0, self.max_energy_eV, "eV"),
        ]
        return dict(
            spectrum_bins=self.spectra_bins,
            spectrum_max_energy_ev=self.max_energy_eV,
            beam_parameters=beam_parameters,
        )

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

    def _graphs(self) -> dict:
        """Named ONNX graphs composing the model. Per-voxel NeRF models export a two-graph
        pair — a 'beam_encoder' (beam parameters -> latent) and a 'trunk' (position + latent ->
        flux/spectrum) — so the deploy runtime encodes the beam once and reuses the latent across
        every voxel. Other models export a single 'trunk'. Keys match the rfnn k*Graph names."""
        if ModelExporter.supports_two_graph_split(self.model):
            try:
                return {
                    "beam_encoder": self._export_bytes(ModelExporter.onnx_export_beam_encoder),
                    "trunk":        self._export_bytes(ModelExporter.onnx_export_trunk),
                }
            except Exception as e:
                print(f"[yellow]ModelPackager: two-graph export failed ({e}); using single trunk[/yellow]")
        return {"trunk": self._export_bytes(ModelExporter.onnx_export)}

    # ── serialisation (mirrors rfnn::io::V1::ModelFactory::save_to_memory) ─────
    def to_bytes(self) -> bytes:
        sample = self._sample_rf3()
        domain = self._domain()
        prov = self._provenance(sample)
        graphs = self._graphs()

        out = bytearray()
        out += RF3M_MAGIC
        out += _u32(RF3M_VERSION)
        out += _str(prov["dataset_name"])
        out += _str(prov["software_version"])
        out += _str(prov["physics"])
        # ModelDomain block (must match put_domain in model_io.cpp byte-for-byte).
        d = domain
        out += struct.pack("<i", d["spectrum_bins"])
        out += struct.pack("<f", d["spectrum_max_energy_ev"])
        out += _u32(len(d["beam_parameters"]))
        for name, count, rmin, rmax, unit in d["beam_parameters"]:
            out += _str(name)
            out += struct.pack("<i", int(count))
            out += struct.pack("<f", float(rmin))
            out += struct.pack("<f", float(rmax))
            out += _str(unit)
        # metrics
        out += _u32(len(self.metrics))
        for k, v in self.metrics.items():
            out += _str(k)
            out += struct.pack("<f", float(v))
        # payload: named ONNX graphs
        out += _u32(len(graphs))
        for name, data in graphs.items():
            out += _str(name)
            out += struct.pack("<Q", len(data))
            out += data
        return bytes(out)

    def save(self, path: str) -> str:
        data = self.to_bytes()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        print(f"[green]Wrote RF3M model package -> {path} ({len(data)/1e6:.2f} MB)[/green]")
        return path
