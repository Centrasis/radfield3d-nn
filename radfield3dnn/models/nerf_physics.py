"""PhysPBRFNet — TPBRFNet with a ray-marched physics probe over a canonical patient occupancy.

Same inputs as TPBRFNet (position, direction, source distance, spectrum, rectangle collimation,
patient translation). The patient is a fixed occupancy grid in the PATIENT frame, held inside the
model; the translation places it in the world. From it the model computes, per query point,
    * the direct-beam transmission  T_src = exp(−μ · chord(source → point)),
    * K scatter emitters where the beam meets the patient (per field: top-K attenuated primary ×
      occupancy on a coarse grid) and, per emitter, the distance, the scattering angle and the
      transmission exp(−μ · chord(emitter → point)) on the way out,
    * a first-order scatter estimate  S = Σ_k w_k · T_k / (4π r_k²)  (and its no-absorption twin),
and hands them to the trunk: the scalars as location features, the K emitters through one small
attention block (query = the point's location features). The flux is
    trunk_flux + prior · e^{amp_d} · T_src · σ(gate_d)  +  S · e^{amp_s} · e^{δ_s},
i.e. the network corrects computed physics instead of inferring line integrals from a point.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from radfield3dnn.models.nerf import PBRFNet, SPERFNet
from radfield3dnn.models.nerf_translation import TPBRFNet
from radfield3dnn.models.layers.beam_geometry import BeamPatientGeometry, RayGeometry
from radfield3dnn.rftypes import PositionalInput, RadiationField, AirKermaField


class PhysicsGeometry(NamedTuple):
    """RayGeometry plus the physics-probe outputs of one forward (row-aligned with the batch)."""
    beam_frame: Tensor
    direct_prior: Tensor
    patient_xyz: Tensor | None
    physics: dict


class PhysicsProbe(nn.Module):
    """Canonical patient occupancy (patient frame, field units: box = 1, isocentre at 0) and the
    ray marches over it. ``occupancy`` is (M, M, M) over the cube [−0.5, 0.5]³, indexed [x, y, z]
    like the dataset's arrays. With ``learn_density`` the grid becomes a parameter (sigmoid of a
    logit initialised from the occupancy) trained through the physics."""

    def __init__(self, occupancy: Tensor, emitters: int, direct_samples: int, emitter_samples: int,
                 learn_density: bool):
        super().__init__()
        assert occupancy.ndim == 3 and occupancy.shape[0] == occupancy.shape[1] == occupancy.shape[2]
        occupancy = occupancy.float().clamp(0.0, 1.0)
        self.grid = int(occupancy.shape[0])
        self.emitters = int(emitters)
        self.direct_samples = int(direct_samples)
        self.emitter_samples = int(emitter_samples)
        self.learn_density = bool(learn_density)
        if self.learn_density:
            self.density_logit = nn.Parameter(torch.logit(occupancy.clamp(0.02, 0.98)))
        else:
            self.register_buffer("occupancy", occupancy)

    def density(self) -> Tensor:
        return torch.sigmoid(self.density_logit) if self.learn_density else self.occupancy

    def sample(self, q: Tensor) -> Tensor:
        """Occupancy at patient-frame points ``q`` (..., 3), trilinear, zero outside the cube. One
        fused grid_sample call: the grid is stored [x, y, z], so grid_sample's (W, H, D) coordinate
        order is (z, y, x)."""
        shape = q.shape[:-1]
        coords = (2.0 * q.reshape(1, -1, 1, 1, 3)).flip(-1)
        vol = self.density()[None, None]
        v = torch.nn.functional.grid_sample(vol, coords.to(vol.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)
        return v.reshape(shape)

    def chord(self, a: Tensor, b: Tensor, samples: int) -> Tensor:
        """Occupancy-weighted length of the segments a→b (..., 3), field units. With a fixed
        occupancy nothing upstream of the samples needs a gradient, so the (…, S, 3) sample
        tensors are kept out of the autograd graph — they are the probe's largest activations."""
        if not self.learn_density:
            with torch.no_grad():
                return self._chord(a, b, samples)
        return self._chord(a, b, samples)

    def _chord(self, a: Tensor, b: Tensor, samples: int) -> Tensor:
        seg = b - a
        length = seg.norm(dim=-1)
        t = (torch.arange(samples, device=a.device, dtype=a.dtype) + 0.5) / samples
        pts = a.unsqueeze(-2) + seg.unsqueeze(-2) * t.view(*([1] * (a.ndim - 1)), samples, 1)
        return self.sample(pts).mean(-1) * length

    def _grid_centres(self, device, dtype) -> Tensor:
        ax = (torch.arange(self.grid, device=device, dtype=dtype) + 0.5) / self.grid - 0.5
        x, y, z = torch.meshgrid(ax, ax, ax, indexing="ij")
        return torch.stack([x, y, z], -1).reshape(-1, 3)

    def field_state(self, geo: BeamPatientGeometry, direction: Tensor, origin: Tensor, rect: Tensor,
                    translation: Tensor, mu: Tensor) -> dict:
        """Per-field emitters. All rows are fields (B). Returns world-frame emitter positions
        (B, K, 3), weights (B, K), a validity mask (B, K), the source (B, 3), the translation
        (B, 3) in field units and ``mu`` (B,)."""
        B = direction.shape[0]
        dev = direction.device
        centres = self._grid_centres(dev, torch.float32)                       # (M³, 3) world
        t_field = geo.translation_to_field(translation.float())                # (B, 3)
        source = geo.source_position(direction, origin)                        # (B, 3)
        pos_enc = geo.field_to_position(centres)
        emit_pos = torch.zeros(B, self.emitters, 3, device=dev)
        emit_w = torch.zeros(B, self.emitters, device=dev)
        for b in range(B):
            ray = geo(pos_enc, direction[b:b + 1], origin[b:b + 1], rect[b:b + 1], translation[b:b + 1])
            occ = self.sample(centres - t_field[b])
            chord_src = self.chord((source[b] - t_field[b]).expand_as(centres), centres - t_field[b], self.direct_samples)
            weight = ray.direct_prior * torch.exp(-mu[b] * chord_src) * occ
            w, idx = torch.topk(weight, self.emitters)
            emit_pos[b] = centres[idx]
            emit_w[b] = w
        # every chunk row carries its field's beam inputs verbatim; this key maps a row back to its field
        key = torch.cat([direction.float(), origin.float().reshape(B, -1), translation.float().reshape(B, -1)], dim=-1)
        # μ keeps its graph (one scalar per field: its per-row index backward is cheap); everything
        # else is a constant for this volume
        return dict(emit_pos=emit_pos.detach(), emit_w=emit_w.detach(), emit_valid=emit_w.detach() > 0, source=source.detach(),
                    translation=t_field.detach(), mu=mu.float(), key=key.detach())

    @staticmethod
    def rows_to_fields(state: dict, direction: Tensor, origin: Tensor, translation: Tensor) -> Tensor:
        """Field index of every row: the field whose beam inputs the row carries."""
        n = direction.shape[0]
        key = torch.cat([direction.float(), origin.float().reshape(n, -1), translation.float().reshape(n, -1)], dim=-1)
        return (key[:, None, :] - state["key"][None, :, :]).abs().sum(-1).argmin(dim=1)

    def point_features(self, state: dict, field_idx: Tensor, p_world: Tensor, direction: Tensor, mu: Tensor) -> dict:
        """Per-point physics for rows ``p_world`` (n, 3) of fields ``field_idx`` (n,) with the
        per-row attenuation ``mu`` (n,). The emitters, source and translation enter as constants:
        per-row copies of wide per-field tensors with gradient made index-backward (a million rows
        accumulated into B fields) dominate the training step. μ is one scalar per field and keeps
        its gradient; a learned density receives its gradient through the chords."""
        t = state["translation"][field_idx]                                     # (n, 3)
        q = p_world - t                                                         # patient frame
        src_q = state["source"][field_idx] - t
        chord_src = self.chord(src_q, q, self.direct_samples)                   # (n,)
        e = state["emit_pos"][field_idx] - t[:, None, :]                        # (n, K, 3)
        w = state["emit_w"][field_idx]                                          # (n, K)
        valid = state["emit_valid"][field_idx]
        v = q[:, None, :] - e
        r = v.norm(dim=-1).clamp_min(1.0 / self.grid)                           # ≥ one coarse voxel
        d = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cos = (v * d[:, None, :]).sum(-1) / r
        chord_k = self.chord(e, q[:, None, :].expand_as(e), self.emitter_samples)   # (n, K)
        att = torch.exp(-mu[:, None] * chord_k)
        geom = 1.0 / (4.0 * math.pi * r * r)
        contrib = w * att * geom * valid
        s_abs = contrib.sum(-1)
        s_iso = (w * geom * valid).sum(-1)
        w_max = w.max(dim=-1, keepdim=True).values.clamp_min(1e-30)
        tokens = torch.stack([
            torch.log(w / w_max + 1e-6),
            torch.log(r),
            cos,
            mu[:, None] * chord_k,
            torch.log(contrib + 1e-30) - torch.log(w_max),
        ], dim=-1) * valid[..., None]
        return dict(transmission_log=-mu * chord_src, s_abs=s_abs, s_iso=s_iso, tokens=tokens, valid=valid)


class EmitterAttention(nn.Module):
    """One single-head attention of a point over its K emitter tokens."""
    TOKEN_FEATURES = 5

    def __init__(self, query_dims: int, token_dim: int):
        super().__init__()
        self.token_dim = int(token_dim)
        self.token_mlp = nn.Sequential(nn.Linear(self.TOKEN_FEATURES, token_dim), nn.SiLU(True), nn.Linear(token_dim, token_dim), nn.SiLU(True))
        self.query = nn.Linear(query_dims, token_dim)
        self.key = nn.Linear(token_dim, token_dim)
        self.value = nn.Linear(token_dim, token_dim)

    def forward(self, query_features: Tensor, tokens: Tensor, valid: Tensor) -> Tensor:
        h = self.token_mlp(tokens)                                              # (n, K, d)
        q = self.query(query_features)                                          # (n, d)
        scores = (self.key(h) * q[:, None, :]).sum(-1) / math.sqrt(self.token_dim)
        scores = scores.masked_fill(~valid, -1e4)
        attn = torch.softmax(scores, dim=-1) * valid.to(scores.dtype)          # all-invalid rows → 0
        return (attn[..., None] * self.value(h)).sum(-2)


class PhysPBRFNet(TPBRFNet):
    """TPBRFNet plus the physics probe (see module docstring). ``conditioning_params["physics"]``:
        {"patient_occupancy_file": <npz with "occupancy" (M,M,M) in the patient frame over the
                                    field box, produced by scripts/make_patient_occupancy.py>,
         "emitters": 16, "direct_samples": 24, "emitter_samples": 12, "token_dim": 16,
         "learn_density": false, "mu_per_m": 20.0,
         "additive_terms": false}
    ``additive_terms`` false (default): the probe's outputs are trunk FEATURES only and the flux is
    PBRFNet's (trunk head + gated direct prior). true: the flux is additionally built from the
    computed terms, ``prior·e^{amp_d}·T_src·σ(gate) + S·e^{amp_s}·e^{δ_s}`` on top of the trunk
    head — measured to collapse on DS04 (the raw terms start 10–250× above the normalised truth,
    the bounded corrections saturate, the trunk head dies; claude-notes/geometry-shift-representation.md).
    Requires the ``beam_geometry`` block with a translation and ``direct_prior``."""
    __model_name__ = "PhysPBRFNet"

    class BackboneModel(TPBRFNet.BackboneModel):
        _PHYSICS_KEYS = frozenset({"patient_occupancy_file", "emitters", "direct_samples", "emitter_samples",
                                   "token_dim", "learn_density", "mu_per_m", "additive_terms"})
        SCATTER_CORRECTION_RANGE = 3.0   # δ_s is soft-clamped to ±3 (×e^±3 ≈ ×20)

        def __init__(self, *args, **kwargs):
            self.probe = None
            self._field_state = None
            super().__init__(*args, **kwargs)
            spec = self._conditioning_params.get("physics")
            if spec is None:
                raise ValueError("PhysPBRFNet needs conditioning_params.physics (see the class docstring)")
            unknown = set(spec) - self._PHYSICS_KEYS
            if unknown:
                raise ValueError(f"physics: unknown keys {sorted(unknown)}; valid: {sorted(self._PHYSICS_KEYS)}")
            if self.beam_geometry is None or not self.beam_geometry.has_translation or not self._direct_prior:
                raise ValueError("PhysPBRFNet needs conditioning_params.beam_geometry with a patient translation and direct_prior true")
            occ_file = spec.get("patient_occupancy_file")
            if not occ_file:
                raise ValueError("physics.patient_occupancy_file is required (scripts/make_patient_occupancy.py builds it)")
            with np.load(occ_file) as f:
                occupancy = torch.from_numpy(np.asarray(f["occupancy"], dtype=np.float32))
                if abs(float(f["field_size_m"]) - self.beam_geometry.field_size_m) > 1e-6:
                    raise ValueError(f"patient occupancy was built for a {float(f['field_size_m'])} m box, the model's is {self.beam_geometry.field_size_m} m")
            self._physics_params = {
                "patient_occupancy_file": occ_file,
                "emitters": int(spec.get("emitters", 16)),
                "direct_samples": int(spec.get("direct_samples", 24)),
                "emitter_samples": int(spec.get("emitter_samples", 12)),
                "token_dim": int(spec.get("token_dim", 16)),
                "learn_density": bool(spec.get("learn_density", False)),
                "mu_per_m": float(spec.get("mu_per_m", 20.0)),
                "additive_terms": bool(spec.get("additive_terms", False)),
            }
            pp = self._physics_params
            # base per-point features (position + beam/patient frames) are the attention query
            self._query_dims = super().location_feature_dims
            self.probe = PhysicsProbe(occupancy, pp["emitters"], pp["direct_samples"], pp["emitter_samples"], pp["learn_density"])
            self.emitter_attention = EmitterAttention(self._query_dims, pp["token_dim"])
            # attenuation per field unit: μ (1/m) · box (m), scaled by a bounded per-field factor
            # from the beam latent (spectrum hardening); starts at the nominal value
            self.mu_field = pp["mu_per_m"] * self.beam_geometry.field_size_m
            self.mu_head = nn.Sequential(nn.Linear(self.beam_latent_dims, self.scalar_encoding_dims), nn.SiLU(True), nn.Linear(self.scalar_encoding_dims, 1))
            self.scatter_amplitude = self.scatter_correction = None
            if pp["additive_terms"]:
                self.scatter_amplitude = nn.Sequential(nn.Linear(self.beam_latent_dims, self.scalar_encoding_dims), nn.SiLU(True), nn.Linear(self.scalar_encoding_dims, 1))
                self.scatter_correction = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.SiLU(True), nn.Linear(self.d_model, 1))
            self._build_trunk_blocks(self.location_feature_dims)

        PHYSICS_SCALARS = 4

        @property
        def location_feature_dims(self) -> int:
            dims = super().location_feature_dims
            if getattr(self, "probe", None) is not None:
                dims += self.PHYSICS_SCALARS + self.emitter_attention.token_dim
            return dims

        # ── field state: emitters per field, computed once per forward2volume ──────────────────
        def _mu(self, beam_latent: Tensor) -> Tensor:
            return self.mu_field * torch.exp(torch.tanh(self.mu_head(beam_latent).squeeze(-1).float()))

        def prepare_field_state(self, x) -> dict:
            """Emitters for the B fields of a field-level input (rows = fields)."""
            translation = self._translation_rows(x)
            latent = self.encode_additional_parameters(x)
            return self.probe.field_state(self.beam_geometry, x.direction, x.origin, x.beam_shape_parameters, translation, self._mu(latent))

        def _physics(self, batch: PositionalInput) -> dict:
            if self._field_state is not None:
                state = self._field_state
                field_idx = self.probe.rows_to_fields(state, batch.direction, batch.origin, self._translation_rows(batch))
            else:
                # rows are their own fields (direct forward calls: export tracing, probes)
                n = batch.position.shape[0]
                if n > 512:
                    raise RuntimeError("PhysPBRFNet.forward on more than 512 rows needs the per-field state; call forward2volume")
                state, field_idx = self.prepare_field_state(batch), torch.arange(n, device=batch.position.device)
            p_world = self.beam_geometry.position_to_field(batch.position.float())
            # per-row μ as a one-hot reduction: indexing B scalars from a million rows makes the
            # backward an atomic pile-up on B addresses
            onehot = torch.nn.functional.one_hot(field_idx, state["mu"].shape[0]).to(state["mu"].dtype)
            mu_rows = (onehot * state["mu"][None, :]).sum(-1)
            return self.probe.point_features(state, field_idx, p_world, batch.direction.float(), mu_rows)

        def point_geometry(self, batch: PositionalInput):
            ray: RayGeometry = super().point_geometry(batch)
            return PhysicsGeometry(ray.beam_frame, ray.direct_prior, ray.patient_xyz, self._physics(batch))

        def encode_location(self, batch: PositionalInput, region_state: Tensor | None = None, geometry=None) -> Tensor:
            base = super().encode_location(batch, region_state, geometry)
            ph = geometry.physics
            dtype = self._compute_dtype
            # scaled to the O(1) range of the sinusoidal features they sit next to: log10 of the
            # estimates over their ~12 decades, the direct optical depth over ~10 e-foldings
            log_abs = torch.log10(ph["s_abs"] + 1e-12)
            log_iso = torch.log10(ph["s_iso"] + 1e-12)
            scalars = torch.stack([ph["transmission_log"] / 10.0, (log_abs + 6.0) / 6.0, (log_iso + 6.0) / 6.0, log_abs - log_iso], dim=-1).to(dtype)
            # bf16 for the token block: its layers are tiny (5→16→16) over rows × K tokens, and the
            # fp32 split-K weight-gradient GEMMs of exactly that shape dominated the training step
            with torch.autocast(device_type=base.device.type, dtype=torch.bfloat16, enabled=base.is_cuda):
                context = self.emitter_attention(base, ph["tokens"], ph["valid"])
            context = context.to(dtype)
            return torch.cat([base, scalars, context], dim=-1)

        def decode_results(self, model_output: Tensor, beam_latent: Tensor | None = None, geometry=None) -> RadiationField | AirKermaField:
            if not self._physics_params["additive_terms"]:
                return super().decode_results(model_output, beam_latent, geometry)   # PBRFNet: trunk flux + gated direct prior
            field = SPERFNet.BackboneModel.decode_results(self, model_output)     # trunk flux + spectrum only
            ph = geometry.physics
            amp_d = self.direct_amplitude(beam_latent).squeeze(-1).float()
            amp_d = self.AMPLITUDE_RANGE * torch.tanh(amp_d / self.AMPLITUDE_RANGE)
            gate = torch.sigmoid(self.direct_transmission(model_output).squeeze(-1).float())
            direct = geometry.direct_prior * torch.exp(amp_d + ph["transmission_log"]) * gate
            amp_s = self.scatter_amplitude(beam_latent).squeeze(-1).float()
            amp_s = self.AMPLITUDE_RANGE * torch.tanh(amp_s / self.AMPLITUDE_RANGE)
            corr = self.SCATTER_CORRECTION_RANGE * torch.tanh(self.scatter_correction(model_output).squeeze(-1).float() / self.SCATTER_CORRECTION_RANGE)
            scatter_term = ph["s_abs"] * torch.exp(amp_s + corr)
            sc = field.scatter_field
            return field._replace(scatter_field=sc._replace(flux=sc.flux + direct + scatter_term))

        def initialize_heads(self):
            super().initialize_heads()
            nn.init.zeros_(self.mu_head[-1].weight); nn.init.zeros_(self.mu_head[-1].bias)
            if self.scatter_correction is not None:
                nn.init.zeros_(self.scatter_correction[-1].weight); nn.init.zeros_(self.scatter_correction[-1].bias)
                # the scatter estimate enters at 1/e² of its raw scale; the amplitude head grows it
                nn.init.zeros_(self.scatter_amplitude[-1].weight); nn.init.constant_(self.scatter_amplitude[-1].bias, -2.0)

    def forward2volume(self, x, voxel_counts, *args, **kwargs):
        """Computes the per-field emitter state once; every chunk row finds its field by the beam
        inputs it carries (PhysicsProbe.rows_to_fields)."""
        core = self.get_core_model()
        core._field_state = core.prepare_field_state(x)   # emitters are constants for this volume, μ keeps its graph
        try:
            return super().forward2volume(x, voxel_counts, *args, **kwargs)
        finally:
            core._field_state = None

    def get_custom_parameters(self):
        params = super().get_custom_parameters()
        cp = dict(params["conditioning_params"] or {})
        cp["physics"] = dict(self.get_core_model()._physics_params)
        params["conditioning_params"] = cp
        return params
