"""Does the predicted field follow the PATIENT, not just the beam?

The beam geometry is handed to the model analytically (`BeamPatientGeometry` gives it the beam
frame and a free-air direct prior), so a model can place a sharp, correctly-aimed beam while
knowing nothing about the body in the way. What it cannot get for free is the shadow: the drop
behind the phantom is a line integral of attenuation along the ray, and the only signal the model
has for it is the patient's translation. Every other metric in the set is dominated by the bright
field and scores such a model almost as well as one that models the body — which is why this one
exists.

Two things are measured, because two different failures are possible and they need separating:

* ``attenuation`` — how DEEP the shadow is. A model can put a shadow in the right place and make
  it far too shallow (measured on a real DS04 model: a true 17x drop reproduced as 2.9x).
* ``boundary`` — WHERE the shadow's edge falls, as the overlap of the attenuated region with the
  one the reference shows. A model can attenuate by the right factor in the wrong voxels, which is
  what a body-shaped silhouette learned from the endpoint coordinate rather than from the path
  through it looks like.
* ``onset`` — whether the attenuation STARTS at the phantom's surface, measured against the
  analytic free-air beam. The other two compare a region against its lateral neighbours, which
  still leaves the inverse-square falloff partly in the answer; this one divides the field by
  `BeamPatientGeometry`'s own collimated free-air prior first, so what remains along a ray is the
  transmission and nothing else. A field that obeys 1/r² right up to the boundary and drops the
  moment it enters the geometry scores 1; one whose drop is smeared across the surface, or begins
  before or after it, does not.

Both are computed per field, so a validation set that varies the patient translation scores the
model across those offsets by construction — that is what makes this a test of FOLLOWING the
geometry rather than of memorising one pose.
"""
from typing import Literal, Union

import torch
from torch import Tensor

from radfield3dnn.metrics.base import MetricBase
from radfield3dnn.preprocessing.airkerma import Airkerma
from radfield3dnn.rftypes import AirKermaField, RadiationFieldChannel, TrainingInputData


def _to_airkerma(airkerma, x):
    if isinstance(x, RadiationFieldChannel):
        return airkerma.forward(x.spectrum, x.flux)
    if isinstance(x, AirKermaField):
        return x.air_kerma
    return x


class PatientShadowAccuracy(MetricBase):
    """Agreement between predicted and reference attenuation behind the patient.

    :param mode: ``attenuation`` scores the depth of the shadow, ``boundary`` its extent, ``onset``
        whether the drop begins at the geometry's surface rather than before or after it.
    :param definition_file: the dataset definition the data was generated with. Required by
        ``onset`` only, which needs the analytic beam; the world size, the source-distance range
        and the translation ranges all come from it, so the analytic field is the one the data was
        made with rather than a second guess at it.
    :param onset_voxels: how far past the surface the "just inside" shell reaches.
    :param march_steps: samples along the ray from a voxel back towards the source. The ray is
        walked in the field's own normalised box, so this is a resolution in box units, not metres.
    :param band_voxels: how far outside the shadow the LIT reference band reaches. The comparison
        is against the shadow's own surroundings rather than the whole field, so the inverse-square
        falloff and the beam profile largely cancel and what is left is the attenuation.
    :param attenuated_fraction: a voxel counts as attenuated when it falls below this fraction of
        the lit band's level. 0.5 = "at least halved", which is well clear of Monte-Carlo noise
        while still catching a shadow that is present but shallow.
    """

    def __init__(self, mu_tr_file: str, spectra_bins: int, max_energy_eV: float,
                 mode: Union[Literal['attenuation'], Literal['boundary'], Literal['onset']] = 'attenuation',
                 march_steps: int = 48, band_voxels: int = 3, attenuated_fraction: float = 0.5,
                 min_voxels: int = 64, eps: float = 1e-12,
                 definition_file: str = None, onset_voxels: int = 3, cone_fraction: float = 0.05):
        super().__init__(layer_name=None, reduction='mean')
        self.airkerma = Airkerma(Airkerma.load_mu_tr_table(mu_tr_file), spectra_bins, max_energy_eV)
        self.mode = mode
        self.march_steps = int(march_steps)
        self.band_voxels = int(band_voxels)
        self.attenuated_fraction = float(attenuated_fraction)
        self.min_voxels = int(min_voxels)
        self.shadow_eps = float(eps)
        self.onset_voxels = int(onset_voxels)
        self.cone_fraction = float(cone_fraction)
        self._definition_file = definition_file
        self._geometry = None
        if mode == 'onset' and definition_file is None:
            raise ValueError("mode='onset' needs definition_file: the analytic beam is built from "
                             "the dataset definition the fields were generated with")

    # ── geometry ──────────────────────────────────────────────────────────────
    @staticmethod
    def _as_volume(x: Tensor, n: int) -> Tensor:
        return x.reshape(x.shape[0], n, n, n)

    def _chord(self, occupied: Tensor, direction: Tensor) -> Tensor:
        """How much phantom each voxel sits behind, in marched samples.

        Zero means the ray reached the voxel through free air; the count grows with the path
        length through the body, which is what makes "just past the surface" expressible.

        Walks each voxel's ray BACK towards the source and asks whether it crosses the phantom.
        Marching from the voxel rather than tracing forwards from the source is what makes this one
        gather per voxel instead of a scatter, so it is a handful of grid lookups and no loop over
        rays.
        """
        b, n = occupied.shape[0], occupied.shape[1]
        axis = (torch.arange(n, device=occupied.device, dtype=torch.float32) + 0.5) / n
        grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), dim=-1)   # (n,n,n,3)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1, -1)
        # `direction` points from the source into the field, so stepping against it walks back up
        # the ray towards the focal spot.
        step = direction[:, :3].reshape(b, 1, 1, 1, 3)
        crossed = torch.zeros(occupied.shape, device=occupied.device, dtype=torch.float32)
        for i in range(1, self.march_steps + 1):
            probe = grid - step * (float(i) / n)
            index = torch.clamp((probe * n - 0.5).round().long(), 0, n - 1)
            flat = (index[..., 0] * n + index[..., 1]) * n + index[..., 2]
            crossed += torch.gather(occupied.reshape(b, -1).float(), 1,
                                    flat.reshape(b, -1)).reshape_as(crossed)
        return crossed

    def _shadow_from_chord(self, chord: Tensor, occupied: Tensor) -> Tensor:
        return (chord > 0) & ~occupied

    def _regions(self, occupied: Tensor, direction: Tensor):
        """(shadow, lit band) — the shadow, and the ring of unshadowed voxels around it."""
        shadow = self._shadow_from_chord(self._chord(occupied, direction), occupied)
        near = shadow.unsqueeze(1).float()
        for _ in range(self.band_voxels):
            near = torch.nn.functional.max_pool3d(near, kernel_size=3, stride=1, padding=1)
        band = (near.squeeze(1) > 0.5) & ~shadow & ~occupied
        return shadow, band

    def _analytic_beam(self, n: int, beam, device) -> Tensor:
        """The free-air collimated fluence this beam would produce with no patient in it.

        `BeamPatientGeometry` is the model's own prior, built from the dataset definition the fields
        were generated with — so the reference is the beam the data was made with, not a second
        implementation of it (rule 5). Everything the drop is measured against therefore already
        contains the inverse-square falloff and the collimation edge, and what is left over is
        transmission.
        """
        from radfield3dnn.models.layers.beam_geometry import BeamPatientGeometry
        if self._geometry is None:
            self._geometry = BeamPatientGeometry.from_dataset_definition(
                self._definition_file, position_frame_extent=1.0,
                with_translation=getattr(beam, 'translation', None) is not None)
        geometry = self._geometry.to(device)
        axis = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n
        position = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
        translation = getattr(beam, 'translation', None)
        # One field at a time: the geometry pairs each position with a beam row, so a batch of
        # fields over one shared grid is B separate evaluations rather than one wide call.
        priors = []
        for i in range(beam.direction.shape[0]):
            ray = geometry(position, beam.direction[i:i + 1].float(), beam.origin[i:i + 1].float(),
                           beam.beam_shape_parameters[i:i + 1].float(),
                           translation[i:i + 1].float() if translation is not None else None)
            priors.append(ray.direct_prior.reshape(n, n, n))
        return torch.stack(priors)

    # ── the metric ────────────────────────────────────────────────────────────
    @staticmethod
    def _median(field: Tensor, mask: Tensor, eps: float) -> Tensor:
        """Per-field median over a mask, as a (B,) tensor. Median, not mean: the shadow holds a
        few bright leakage voxels and a mean would report the leak rather than the shadow."""
        out = []
        for i in range(field.shape[0]):
            values = field[i][mask[i]]
            out.append(values.median() if values.numel() > 0 else field.new_tensor(eps))
        return torch.stack(out)

    def forward(self, target, prediction, input: TrainingInputData = None) -> Tensor:
        if isinstance(prediction, RadiationFieldChannel) and (prediction.spectrum is None or prediction.flux is None):
            return None
        # No phantom description, no question to ask. Reported as absent rather than as a zero
        # score, which would read as "the model failed" when nothing was measured.
        beam = getattr(input, 'input', None) if input is not None else None
        geometry = getattr(beam, 'geometry', None) if beam is not None else None
        direction = getattr(beam, 'direction', None) if beam is not None else None
        if geometry is None or direction is None:
            return None

        tgt = _to_airkerma(self.airkerma, target).float()
        prd = _to_airkerma(self.airkerma, prediction).float()
        if tgt.dim() == 5:
            tgt, prd = tgt.squeeze(1), prd.squeeze(1)
        b = tgt.shape[0]
        n = int(round(float(tgt[0].numel()) ** (1.0 / 3.0)))
        tgt, prd = self._as_volume(tgt, n), self._as_volume(prd, n)
        occupied = self._as_volume(geometry.reshape(b, -1).float(), n) > 0.5
        chord = self._chord(occupied, direction.float())
        shadow = self._shadow_from_chord(chord, occupied)
        if self.mode == 'onset':
            return self._onset(tgt, prd, occupied, chord, beam, n)
        shadow, band = self._regions(occupied, direction.float())

        enough = (shadow.reshape(b, -1).sum(1) >= self.min_voxels) & \
                 (band.reshape(b, -1).sum(1) >= self.min_voxels)
        if not bool(enough.any()):
            # Every field in the batch has the phantom out of the beam: nothing casts a shadow, so
            # there is nothing to score. Not a failure — the geometry simply is not being tested.
            return None

        lit_t, shd_t = self._median(tgt, band, self.shadow_eps), self._median(tgt, shadow, self.shadow_eps)
        lit_p, shd_p = self._median(prd, band, self.shadow_eps), self._median(prd, shadow, self.shadow_eps)

        if self.mode == 'attenuation':
            ratio_t = lit_t / shd_t.clamp_min(self.shadow_eps)
            ratio_p = lit_p / shd_p.clamp_min(self.shadow_eps)
            # Scored in the LOG of the ratio, because attenuation is exponential in path length:
            # 17x against 2.9x and 170x against 29x are the same physical error and must score the
            # same. exp(-|ln|) gives 1.0 on an exact match and 0.37 at a factor of e.
            score = torch.exp(-(torch.log(ratio_p.clamp_min(self.shadow_eps)) -
                                torch.log(ratio_t.clamp_min(self.shadow_eps))).abs())
        elif self.mode == 'boundary':
            # Where each field says the shadow IS, judged against its own lit level so the two are
            # compared on shape rather than on scale; Dice over the shadow-plus-band region.
            region = shadow | band
            cut_t = (self.attenuated_fraction * lit_t).reshape(b, 1, 1, 1)
            cut_p = (self.attenuated_fraction * lit_p).reshape(b, 1, 1, 1)
            dark_t, dark_p = (tgt < cut_t) & region, (prd < cut_p) & region
            inter = (dark_t & dark_p).reshape(b, -1).sum(1).float()
            total = dark_t.reshape(b, -1).sum(1).float() + dark_p.reshape(b, -1).sum(1).float()
            # Both empty means both agree there is no shadow here, which is a pass, not a 0/0.
            score = torch.where(total > 0, 2.0 * inter / total.clamp_min(1.0), torch.ones_like(total))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return score[enough].mean()

    def _onset(self, tgt: Tensor, prd: Tensor, occupied: Tensor, chord: Tensor, beam, n: int):
        """Does the drop begin AT the surface, against the analytic free-air beam?

        Dividing by the prior removes the falloff and the collimation, so along one ray the
        remainder is flat until the geometry starts. The shell just past the surface is then
        compared with the free-air shell just before it — in the prediction and in the reference —
        and the two transmissions are held against each other.
        """
        prior = self._analytic_beam(n, beam, tgt.device)
        b = tgt.shape[0]
        # Only inside the collimated cone: outside it the prior is ~0 and the quotient is noise
        # divided by nothing. Relative to the TYPICAL in-cone value, not the peak: the prior falls
        # as 1/r² across the box, so a fraction of the peak keeps only the end nearest the source
        # (measured: 28 voxels of a ~7800-voxel cone) and the shells never fill.
        lit_prior = torch.where(prior > 0, prior, torch.full_like(prior, float('nan')))
        typical = lit_prior.reshape(b, -1).nanmedian(dim=1).values.reshape(b, 1, 1, 1)
        inside = prior > (self.cone_fraction * typical).clamp_min(self.shadow_eps)
        transmission_t = tgt / prior.clamp_min(self.shadow_eps)
        transmission_p = prd / prior.clamp_min(self.shadow_eps)

        # The first material along the ray — INCLUDING the occupied voxels themselves, because the
        # boundary is where the geometry starts and that is exactly where the flux must leave the
        # free-air curve. Excluding them would leave only rays that graze the silhouette, since
        # `chord` counts the whole path and a body thicker than this shell never has a voxel
        # behind it with a small count.
        entered = (chord >= 1.0) & (chord <= float(self.onset_voxels)) & inside
        # Free air, but within reach of the surface: the same rays, just short of entering.
        near = (chord == 0).unsqueeze(1).float()
        for _ in range(self.band_voxels):
            near = torch.nn.functional.max_pool3d(near, kernel_size=3, stride=1, padding=1)
        before = (chord == 0) & inside & ~occupied & (near.squeeze(1) > 0.5)

        enough = (entered.reshape(b, -1).sum(1) >= self.min_voxels) & \
                 (before.reshape(b, -1).sum(1) >= self.min_voxels)
        if not bool(enough.any()):
            return None
        ratio = lambda f: (self._median(f, before, self.shadow_eps).clamp_min(self.shadow_eps) /
                           self._median(f, entered, self.shadow_eps).clamp_min(self.shadow_eps))
        drop_t, drop_p = ratio(transmission_t), ratio(transmission_p)
        score = torch.exp(-(torch.log(drop_p.clamp_min(self.shadow_eps)) -
                            torch.log(drop_t.clamp_min(self.shadow_eps))).abs())
        return score[enough].mean()
