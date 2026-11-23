"""Loading the directional ground truth: von Mises-Fisher lobes per voxel.

A `vmf_mixture` layer stores, per voxel, L slots of ``[weight, mx, my, mz, kappa]`` — the weights
summing to one over the slots and the means unit-norm. It describes WHERE the flux in that voxel is
going, which no scalar field can: two voxels of identical flux and identical spectrum can be lit
from opposite sides, and only this tells them apart.

RadFiled3D's own dataset reads ``["spectrum", "flux", "error"]`` and builds a NamedTuple with no
slot for a fourth layer, so the lobes are loaded here and carried on the repo's `TrainingInputData`
beside the channel (see `rftypes.TrainingInputData.lobes`).

**Unused slots are zero, not absent.** A generator configured for four lobes writes five slots and
fills as many as the voxel needed; a slot that was never filled is all-zero, including its mean,
which is not a unit vector. Anything consuming these must read the WEIGHT to decide a slot is real
— normalising a zero mean yields a NaN, and a loss that averages over slots would then be NaN for
the whole field.

**A voxel whose weights sum to zero is loaded as the uniform distribution.** No particle crossed it,
so nothing says where its flux goes; that is the uniform density on the sphere, not an absent one.
It is written as slot 0 with weight 1 and kappa 0 — the vMF limit that IS uniform — so the target
is a valid mixture everywhere and every voxel can be scored. Its mean carries no information and is
set to a unit vector only so the slot stays well formed.
"""
import torch

from radfiled3d.pytorch.datasets.radfield3d import RadField3DDataset
from radfield3dnn.rftypes import TrainingInputData

LOBE_LAYER = "vmf_lobes"
LOBE_PARAMS = 5  # weight, mx, my, mz, kappa


def fill_uniform(lobes: torch.Tensor) -> torch.Tensor:
    """``(L*5, D, W, H)`` lobes with every zero-weight voxel replaced by the uniform distribution."""
    lobes = lobes.clone()
    empty = lobes[0::LOBE_PARAMS].sum(0) <= 0.0                            # (D, W, H)
    slot0 = lobes[:LOBE_PARAMS]
    slot0[0][empty] = 1.0                                                   # weight
    slot0[1][empty] = 0.0
    slot0[2][empty] = 0.0
    slot0[3][empty] = 1.0                                                   # mean: any unit vector
    slot0[4][empty] = 0.0                                                   # kappa 0: uniform
    return lobes


def uniform_voxels(weight: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """True where a mixture is the uniform distribution: its heaviest lobe has kappa 0. Such a
    voxel has no direction, so a heaviest-lobe angle against it is undefined."""
    return torch.gather(kappa, -1, weight.argmax(-1, keepdim=True)).squeeze(-1) <= 0.0


class PerFileAccessorDataset(RadField3DDataset):
    """A RadField3DDataset that reads each field through an accessor built for THAT file.

    The base class constructs one accessor from ``file_paths[0]`` and reuses it for the whole
    dataset, which assumes every field shares one byte layout. A generator that writes a dynamic
    metadata entry only for SOME fields breaks that: DS04-Angio-ArtisQ writes ``patient_overlap``
    only where the patient is in the beam, so 35 of 274 headers are 149 bytes longer than the rest.
    Reading one layout with the other's offsets does not fail cleanly — it returns garbage, or
    raises `bad_alloc` out of a length read from the middle of a voxel.

    Building the accessor per file costs about a tenth of a percent of the read, so nothing is
    gained by the uniform-layout assumption. It is a workaround: the shared accessor is
    RadFiled3D's, and the fix there is to key the accessor on the file it reads.
    """

    def _access_field_arrays(self, idx: int, channels: list, layers: list) -> dict:
        from radfiled3d.store import FieldStore
        if self.is_dataset_zipped:
            buffer = self.load_file_buffer(idx)
            return FieldStore.construct_field_accessor_from_buffer(buffer).access_field_arrays_from_buffer(
                buffer, channels, layers, True)
        path = self.file_paths[idx]
        return FieldStore.construct_field_accessor(path).access_field_arrays(path, channels, layers, True)


class LobedFieldDataset(PerFileAccessorDataset):
    """A field dataset that also loads the directional lobes of the radiation channel.

    `lobes` arrives as ``(L*5, D, W, H)`` and is attached to the item, leaving the ground truth
    exactly the type every downstream dispatch expects.
    """

    def __init__(self, *args, lobe_channel: str = "radiation", **kwargs):
        super().__init__(*args, **kwargs)
        self.lobe_channel = lobe_channel

    def __getitem__(self, idx: int) -> TrainingInputData:
        item = super().__getitem__(idx)
        file_idx = idx % len(self.file_paths)
        arrays = self._access_field_arrays(file_idx, [self.lobe_channel], [LOBE_LAYER])
        lobes = fill_uniform(torch.from_numpy(arrays[self.lobe_channel][LOBE_LAYER]).to(torch.float32))
        return TrainingInputData(input=item.input, ground_truth=item.ground_truth,
                                 original_ground_truth=getattr(item, "original_ground_truth", None),
                                 lobes=lobes)


def split_lobes(lobes: torch.Tensor):
    """``(..., L*5, D, W, H)`` -> ``(weight, mean, kappa)`` over the voxel grid.

    The layer arrives CHANNEL-FIRST and flattened, like flux and spectrum, with channel
    ``lobe * 5 + param`` — lobe-major, verified against the field's own ``(D, W, H, L, 5)`` array.
    What comes back is voxel-major instead, because every directional quantity below is computed
    per voxel over the lobe axis:

        weight ``(..., D, W, H, L)``   mean ``(..., D, W, H, L, 3)``   kappa ``(..., D, W, H, L)``

    No normalisation and no masking: that is the caller's, because what counts as a real slot
    depends on what the caller is doing with it.
    """
    channels = lobes.shape[-4]
    assert channels % LOBE_PARAMS == 0, (
        f"expected a multiple of {LOBE_PARAMS} channels (lobe-major [weight, mx, my, mz, kappa]), "
        f"got {channels}")
    lobe_count = channels // LOBE_PARAMS
    # (..., C, D, W, H) -> (..., D, W, H, C) -> (..., D, W, H, L, 5)
    voxelwise = lobes.movedim(-4, -1).unflatten(-1, (lobe_count, LOBE_PARAMS))
    return voxelwise[..., 0], voxelwise[..., 1:4], voxelwise[..., 4]
