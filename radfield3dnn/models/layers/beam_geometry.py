"""Beam / patient relational geometry derived INSIDE the model from its existing inputs.

The model keeps consuming the absolute query position, the beam direction, the normalised source
distance, the normalised rectangle collimation and (optionally) the normalised patient translation.
This module inverts those normalisations into one metric frame ("field units": the field box is the
unit cube centred on the isocentre) and expresses every query point relative to the beam and to the
patient, so the trunk no longer has to learn the coordinate transforms itself:

  * beam frame     — (axial depth from the source in units of the SID, lateral offsets in units of
                     the collimated half-width at that depth): the direct beam is a fixed shape here.
  * patient frame  — the query position minus the patient translation: the phantom is static here.
  * ray vs patient — per field: where the central ray passes the patient origin (closest-approach
                     depth, impact vector in the collimator axes, in half-width units).
  * direct prior   — the collimated free-air fluence (focal-spot penumbra × inverse-square), unit
                     peak at the near-source clamp for the reference collimation; a learned
                     amplitude and a learned transmission gate turn it into the direct-beam flux.

Conventions (all must match the dataset generator, RadField3DSimulation):
  direction points source → isocentre, the source sits at −d·SID; the collimator axes are the
  minimal rotation of (0,0,−1) onto d (RadiationSource::setTransform); the field rectangle (w, h) is
  the size AT THE ISOCENTRE and grows linearly with the axial distance from the source; the patient
  is translated by +t in the world frame. Every op is plain ONNX arithmetic.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from radfield3dnn.preprocessing.normalizations.translation import TranslationNormalization


_AXES = ("X", "Y", "Z")


def beam_rect_axes(direction: Tensor) -> tuple[Tensor, Tensor]:
    """Collimator axes (e1, e2) for unit beam directions ``direction`` (N, 3): the local X/Y axes
    rotated by the MINIMAL rotation taking the local beam axis (0, 0, −1) onto ``direction``
    (Rodrigues, (1−c)/s² = 1/(1+c)). The anti-parallel case (d = +z) degenerates to the identity."""
    dev, dt = direction.device, direction.dtype
    up = torch.tensor([0., 0., -1.], device=dev, dtype=dt).expand_as(direction)
    v = torch.cross(up, direction, dim=1)
    cdot = (up * direction).sum(1)
    k = 1.0 / (1.0 + cdot).clamp_min(1e-6)
    zero = torch.zeros_like(cdot)
    Vx = torch.stack([
        torch.stack([zero, -v[:, 2], v[:, 1]], dim=-1),
        torch.stack([v[:, 2], zero, -v[:, 0]], dim=-1),
        torch.stack([-v[:, 1], v[:, 0], zero], dim=-1),
    ], dim=1)
    R = torch.eye(3, device=dev, dtype=dt)[None] + Vx + torch.bmm(Vx, Vx) * k[:, None, None]
    return R[:, :, 0], R[:, :, 1]


class RayGeometry(NamedTuple):
    """Per-point relational features, all in field units (box = 1, isocentre at the origin)."""
    beam_frame: Tensor            # (N, 3): (s/SID − 1, l1/hw(s), l2/hh(s)), lateral terms soft-clamped to ±LATERAL_CLAMP
    direct_prior: Tensor          # (N,): free-air collimated fluence, ≤ reference_area/(w·h) at the clamp
    patient_xyz: Tensor | None    # (N, 3): query position in the patient frame (x − t); None without translation


def _rows(x: Tensor, n: int) -> Tensor:
    """A per-field (1, C) tensor broadcast to the (n, C) query rows; (n, C) passes through."""
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.shape[0] == n:
        return x
    assert x.shape[0] == 1, f"expected 1 or {n} rows, got {x.shape[0]}"
    return x.expand(n, *x.shape[1:])


class BeamPatientGeometry(nn.Module):
    """Computes :class:`RayGeometry` from the model's NORMALISED inputs. Holds no learnable state;
    the dataset's physical ranges (the normalisations' inverses) are fixed constants."""

    LATERAL_CLAMP = 4.0   # lateral coordinates are in half-width units; |l̃| > 4 is far outside the beam.
    # Saturated smoothly (c·tanh(l/c)): the half-width vanishes at the source, so a hard clamp would
    # pin every near-source voxel to one plateau value and hand the encoders a step.

    @classmethod
    def _soft_clamp(cls, l: Tensor) -> Tensor:
        c = cls.LATERAL_CLAMP
        return c * torch.tanh(l / c)

    def __init__(self, field_size_m: float, distance_range_m: tuple[float, float],
                 translation_ranges_m: dict[str, tuple[float, float]] | None,
                 position_frame_extent: float,
                 penumbra_m: float = 0.01, source_clamp_m: float = 0.02,
                 reference_area_m2: float = 0.01,
                 translation_output_range: tuple[float, float] = (0.0, 1.0)):
        super().__init__()
        assert field_size_m > 0, f"field_size_m must be positive, got {field_size_m}"
        lo, hi = (float(v) for v in distance_range_m)
        # hi == lo is a fixed tube distance: the normalised input is then a constant and SID == lo
        assert hi >= lo >= 0, f"distance_range_m must be a non-decreasing non-negative range, got {distance_range_m}"
        t_lo, t_hi = (float(v) for v in translation_output_range)
        assert t_hi > t_lo, f"translation_output_range must increase, got {translation_output_range}"
        assert position_frame_extent in (1.0, 2.0), \
            f"position_frame_extent must be 1.0 ([0,1] grid) or 2.0 ([-1,1] grid), got {position_frame_extent}"
        self.field_size_m = float(field_size_m)
        self.distance_range_m = (lo, hi)
        self.translation_ranges_m = (
            {axis: (float(a), float(b)) for axis, (a, b) in translation_ranges_m.items()}
            if translation_ranges_m else None)
        self.position_frame_extent = float(position_frame_extent)
        self.penumbra_m = float(penumbra_m)
        self.source_clamp_m = float(source_clamp_m)
        self.reference_area_m2 = float(reference_area_m2)
        self.translation_output_range = (t_lo, t_hi)

        f = self.field_size_m
        # distance: normalised od ∈ [0,1] → SID = lo + od·(hi−lo), in field units
        self.register_buffer("_dist_lo", torch.tensor(lo / f), persistent=False)
        self.register_buffer("_dist_span", torch.tensor((hi - lo) / f), persistent=False)
        # translation: normalised per axis over the generation range into translation_output_range
        # (TranslationNormalization); constant axes sit at its low end and carry span 0
        lows, spans = [], []
        for axis in _AXES:
            a, b = (self.translation_ranges_m or {}).get(axis, (0.0, 0.0))
            lows.append(a / f - t_lo * (b - a) / (f * (t_hi - t_lo)))
            spans.append((b - a) / (f * (t_hi - t_lo)))
        self.register_buffer("_trans_lo", torch.tensor(lows), persistent=False)
        self.register_buffer("_trans_span", torch.tensor(spans), persistent=False)
        self.register_buffer("_source_clamp", torch.tensor(self.source_clamp_m / f), persistent=False)
        self.register_buffer("_penumbra", torch.tensor(self.penumbra_m / f), persistent=False)
        self.register_buffer("_reference_area", torch.tensor(self.reference_area_m2 / (f * f)), persistent=False)

    # ── construction ────────────────────────────────────────────────────────────────────────────
    @classmethod
    def from_dataset_definition(cls, definition: str | Path | dict, position_frame_extent: float,
                                with_translation: bool, **kwargs) -> "BeamPatientGeometry":
        """Physical ranges from the dataset definition JSON the data was generated with
        (``Metaparameters.WorldDim``, ``Parameters.source_distance``,
        ``GeometryTransformations.<translated object>.Translation``). A fixed ``source_distance``
        (``value`` instead of ``range``) yields the degenerate range (v, v)."""
        if not isinstance(definition, dict):
            with open(definition, "r") as fh:
                definition = json.load(fh)
        world_dim = definition.get("Metaparameters", {}).get("WorldDim")
        assert world_dim, "Dataset definition lacks Metaparameters.WorldDim"
        field_size_m = float(max(world_dim))
        params = {p["name"]: p for p in definition.get("Parameters", [])}
        distance = params.get("source_distance", {})
        if "range" in distance:
            distance_range_m = tuple(float(v) for v in distance["range"])
        else:
            assert "value" in distance, "Dataset definition lacks a source_distance range or value"
            distance_range_m = (float(distance["value"]), float(distance["value"]))
        translation_ranges_m = None
        if with_translation:
            translation_ranges_m = TranslationNormalization.from_dataset_definition(definition).translation_ranges
        return cls(field_size_m, distance_range_m, translation_ranges_m, position_frame_extent, **kwargs)

    def get_parameters(self) -> dict:
        """The resolved physical ranges and prior constants; ``from_parameters`` rebuilds an
        identical module without the definition file."""
        return {
            "field_size_m": self.field_size_m,
            "distance_range_m": list(self.distance_range_m),
            "translation_ranges_m": ({k: list(v) for k, v in self.translation_ranges_m.items()}
                                     if self.translation_ranges_m else None),
            "penumbra_m": self.penumbra_m,
            "source_clamp_m": self.source_clamp_m,
            "reference_area_m2": self.reference_area_m2,
            "translation_output_range": list(self.translation_output_range),
        }

    @classmethod
    def from_parameters(cls, params: dict, position_frame_extent: float) -> "BeamPatientGeometry":
        tr = params.get("translation_ranges_m")
        return cls(
            field_size_m=params["field_size_m"],
            distance_range_m=tuple(params["distance_range_m"]),
            translation_ranges_m={k: tuple(v) for k, v in tr.items()} if tr else None,
            position_frame_extent=position_frame_extent,
            penumbra_m=params.get("penumbra_m", 0.01),
            source_clamp_m=params.get("source_clamp_m", 0.02),
            reference_area_m2=params.get("reference_area_m2", 0.01),
            translation_output_range=tuple(params.get("translation_output_range", (0.0, 1.0))),
        )

    @property
    def has_translation(self) -> bool:
        return self.translation_ranges_m is not None

    # ── unit conversions (normalised model inputs → field units) ────────────────────────────────
    def position_to_field(self, position: Tensor) -> Tensor:
        """Node-grid position in the encoder frame ([0,1] or [−1,1]) → field units centred on the
        isocentre. The node grid i/(N−1) spans the box edge to edge, so this is exact at the centre
        and half a voxel off at the box faces — below the penumbra's own width."""
        x = position / self.position_frame_extent
        if self.position_frame_extent == 1.0:
            x = x - 0.5
        return x

    def field_to_position(self, x: Tensor) -> Tensor:
        """Inverse of :meth:`position_to_field`: field units (isocentre at 0, box = 1) → encoder frame."""
        if self.position_frame_extent == 1.0:
            x = x + 0.5
        return x * self.position_frame_extent

    def translation_to_field(self, translation: Tensor) -> Tensor:
        return self._trans_lo + translation[:, :3] * self._trans_span

    def source_position(self, direction: Tensor, origin_distance: Tensor) -> Tensor:
        """(B, 3) focal-spot position in field units: the beam points along ``direction`` towards
        the isocentre from one source-isocentre distance behind it."""
        d = direction.float()
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return -d * self.source_distance_to_field(origin_distance.float())[:, None]

    def source_distance_to_field(self, origin_distance: Tensor) -> Tensor:
        return self._dist_lo + origin_distance.reshape(-1) * self._dist_span

    # ── geometry ────────────────────────────────────────────────────────────────────────────────
    def _lateral(self, rv: Tensor, d: Tensor, e1: Tensor, e2: Tensor, rect: Tensor, sid: Tensor):
        """Axial depth ``s`` of the vectors ``rv`` (from the source) along ``d``, the lateral
        offsets in units of the collimated half-widths at that depth, and those half-widths."""
        s = (rv * d).sum(-1)
        lat = rv - s[:, None] * d
        depth = s.clamp_min(self._source_clamp)
        hw = (rect[:, 0] * 0.5).clamp_min(1e-4) * depth / sid
        hh = (rect[:, 1] * 0.5).clamp_min(1e-4) * depth / sid
        l1 = (lat * e1).sum(-1) / hw
        l2 = (lat * e2).sum(-1) / hh
        return s, l1, l2, hw, hh

    def _coverage(self, l: Tensor, hw: Tensor, s: Tensor, sid: Tensor) -> Tensor:
        """Smoothstep edge: 1 inside, 0.5 at the geometric edge (|l| = 1), 0 one penumbra
        half-width outside. The focal-spot blur ``penumbra_m`` grows with the depth beyond the
        isocentre and is floored at its isocentre value nearer the source."""
        pen = self._penumbra * (s / sid).clamp_min(1.0)
        x = (0.5 + 0.5 * (1.0 - l.abs()) * hw / pen).clamp(0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _beam(self, direction: Tensor, origin_distance: Tensor, rect: Tensor, n: int):
        """Unit direction, source position, SID, rect (w, h) and collimator axes for n rows."""
        d = _rows(direction, n).float()
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        sid = self.source_distance_to_field(_rows(origin_distance, n).float())
        rect = _rows(rect, n).float()[:, :2]
        source = -d * sid[:, None]
        e1, e2 = beam_rect_axes(d)
        return d, source, sid, rect, e1, e2

    def forward(self, position: Tensor, direction: Tensor, origin_distance: Tensor, rect: Tensor,
                translation: Tensor | None = None) -> RayGeometry:
        """Per-point geometry. ``position`` (N,3) in the encoder frame; the beam inputs NORMALISED
        as the model receives them, with N rows or a single field row: ``direction`` (·,3),
        ``origin_distance`` (·,1) in [0,1], ``rect`` (·,2) = (w, h)/field extent, ``translation``
        (·,3) in ``translation_output_range`` or None."""
        n = position.shape[0]
        x = self.position_to_field(position.float())
        d, source, sid, rect, e1, e2 = self._beam(direction, origin_distance, rect, n)

        s, l1, l2, hw, hh = self._lateral(x - source, d, e1, e2, rect, sid)
        beam_frame = torch.stack([s / sid - 1.0, self._soft_clamp(l1), self._soft_clamp(l2)], dim=-1)

        # free-air collimated fluence: coverage × inverse square, referenced to the near-source
        # clamp (unit at depth == source_clamp) and to reference_area/(w·h) for the collimation
        depth = s.clamp_min(self._source_clamp)
        area = (rect[:, 0] * rect[:, 1]).clamp_min(1e-6)
        prior = self._coverage(l1, hw, s, sid) * self._coverage(l2, hh, s, sid) \
            * (self._source_clamp / depth) ** 2 * (self._reference_area / area)
        prior = torch.where(s > 0, prior, torch.zeros_like(prior))

        patient_xyz = None
        if translation is not None:
            assert self.has_translation, "BeamPatientGeometry was built without translation ranges"
            patient_xyz = x - self.translation_to_field(_rows(translation, n).float())

        return RayGeometry(beam_frame=beam_frame, direct_prior=prior, patient_xyz=patient_xyz)

    def ray_vs_patient(self, direction: Tensor, origin_distance: Tensor, rect: Tensor,
                       translation: Tensor) -> Tensor:
        """Per-field (B, 3) relation of the central ray to the patient origin: the closest-approach
        depth (in SIDs, isocentre at 0) and the impact vector in the collimator axes, in units of
        the collimated half-widths at that depth (|b| < 1: the patient origin lies inside the
        beam). Same normalised inputs as :meth:`forward`."""
        assert self.has_translation, "BeamPatientGeometry was built without translation ranges"
        n = translation.shape[0] if translation.ndim == 2 else 1
        d, source, sid, rect, e1, e2 = self._beam(direction, origin_distance, rect, n)
        t = self.translation_to_field(_rows(translation, n).float())
        s_c, b1, b2, _, _ = self._lateral(t - source, d, e1, e2, rect, sid)
        return torch.stack([s_c / sid - 1.0, self._soft_clamp(b1), self._soft_clamp(b2)], dim=-1)
