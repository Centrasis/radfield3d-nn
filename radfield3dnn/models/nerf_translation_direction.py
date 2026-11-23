import torch
from torch import Tensor

from .nerf_translation import TPBRFNet
from radfield3dnn.rftypes import positional_like
from radfield3dnn.models.layers.vmf_head import VMFLobeHead


class TDPBRFNet(TPBRFNet):
    """TPBRFNet plus a von Mises-Fisher head: per point, an L-lobe mixture over the directions the
    flux there is travelling, learned beside flux and spectrum.

    The head reads the same per-point trunk features the flux decoder does, so it shares everything
    TPBRFNet learns about the beam and the patient. It is trained against the dataset's ``vmf_lobes``
    layer (``dataset: use_lobes: true``) by the Cauchy-Schwarz divergence, and joins the DB-MTL
    combine as its own task. Only a dataset that carries those lobes can train it; TPBRFNet is the
    model for the ones that do not, and the two score identically on flux and spectrum.
    """
    __model_name__ = "TDPBRFNet"
    learns_lobes = True
    warm_start_new_parameters = ("lobe_head.",)

    def __init__(self, *args, direction_lobes: int, lobe_samples: int = 16384, lobe_hidden: int = 128,
                 **kwargs):
        """direction_lobes: L >= 1 lobes per point, the dataset's own lobe count.

        lobe_samples: voxels per field the directional loss is evaluated on each step. The head is a
        per-point quantity, so a random subset is an unbiased estimate of the whole-field loss at a
        fraction of the trunk cost — and it keeps the lobes OUT of forward2volume, whose voxel-mean
        supersampling would average vMF parameters, which does not yield a vMF.
        """
        super().__init__(*args, **kwargs)
        if int(direction_lobes) < 1:
            raise ValueError(f"TDPBRFNet needs direction_lobes >= 1, got {direction_lobes}; "
                             f"a model without a directional head is TPBRFNet.")
        self._direction_lobes = int(direction_lobes)
        self._lobe_samples = int(lobe_samples)
        self._lobe_hidden = int(lobe_hidden)
        # The head reads the same per-point features the flux decoder does.
        trunk_width = next(m for m in self.backbone_model.flux_decoder.modules()
                           if isinstance(m, torch.nn.Linear)).in_features
        self.lobe_head = VMFLobeHead(trunk_width, lobes=self._direction_lobes, hidden=self._lobe_hidden)
        self.lobe_head.initialize()

    def _positions(self, linear_index: Tensor) -> Tensor:
        """Query positions of flat voxel indices, built exactly as forward2volume builds them, so the
        lobe pass and the flux pass see one voxel at one coordinate."""
        grid = self.base_voxel_map.reshape(-1, self.base_voxel_map.shape[-1])[:, :3]
        position = grid.index_select(0, linear_index.reshape(-1))
        if self.voxels_centered_around_origin:
            position = position * 2.0 - 1.0
        return position

    def predict_lobes(self, x, linear_index: Tensor, voxel_counts):
        """Lobes at the flat voxel indices ``(B, n)`` of each field: weight ``(B, n, L)``, mean
        ``(B, n, L, 3)``, kappa ``(B, n, L)``.

        `voxel_counts` is the grid the indices refer to. Taken explicitly rather than read from
        whatever the last `forward2volume` left behind: an index means a different position on a
        different grid, and depending on state another call set would score the wrong voxels the
        first time this ran on its own.
        """
        self.voxel_counts = torch.as_tensor(tuple(int(v) for v in voxel_counts))
        batch, n = linear_index.shape
        rows = torch.arange(batch, device=linear_index.device).repeat_interleave(n)
        take = lambda t: t.index_select(0, rows) if t is not None else None
        query = positional_like(
            x, direction=take(x.direction), position=self._positions(linear_index),
            spectrum=take(x.spectrum), geometry=take(getattr(x, "geometry", None)),
            origin=take(x.origin), beam_shape_parameters=take(x.beam_shape_parameters),
            beam_shape_type=take(x.beam_shape_type), translation=take(getattr(x, "translation", None)))
        features, _, _ = self.backbone_model.trunk_features(query)
        dtype = next(self.lobe_head.parameters()).dtype
        weight, mean, kappa = self.lobe_head(features.to(dtype))
        return (weight.unflatten(0, (batch, n)), mean.unflatten(0, (batch, n)),
                kappa.unflatten(0, (batch, n)))

    @staticmethod
    def flat_lobes(lobes: Tensor):
        """Target lobes ``(B, L*5, D, W, H)`` -> voxel-major ``(B, D*W*H, L[, 3])``."""
        from radfield3dnn.datasets.lobes import split_lobes
        weight, mean, kappa = split_lobes(lobes)
        b = lobes.shape[0]
        return weight.reshape(b, -1, weight.shape[-1]), mean.reshape(b, -1, *mean.shape[-2:]), \
            kappa.reshape(b, -1, kappa.shape[-1])

    def direction_loss(self, x, lobes: Tensor, trained: Tensor) -> Tensor:
        """Mean Cauchy-Schwarz divergence over `lobe_samples` voxels per field, drawn uniformly
        among the ``trained`` ones (B, V) — those whose flux target is not masked to -inf, so the
        floor cut applies to the direction exactly as to flux and spectrum.

        Every voxel has a target: one no particle crossed is the uniform distribution (see
        ``datasets.lobes.fill_uniform``). Uniform over voxels rather than flux-weighted, so the
        scatter's multi-lobe structure counts as much as the bright, single-lobed beam core.
        """
        from radfield3dnn.losses.vmf import cauchy_schwarz_divergence
        t_weight, t_mean, t_kappa = self.flat_lobes(lobes)
        b, v = t_weight.shape[:2]
        n = min(self._lobe_samples, v)
        # masked voxels lose every draw: their keys sit below the [0, 1) of the trained ones
        keys = torch.rand(b, v, device=t_weight.device) - 2.0 * (~trained).to(t_weight.dtype)
        index = keys.topk(n, dim=1).indices
        gather = lambda t: torch.gather(t, 1, index.view(*index.shape, *([1] * (t.dim() - 2))).expand(
            *index.shape, *t.shape[2:]))
        target = (gather(t_weight), gather(t_mean), gather(t_kappa))
        prediction = self.predict_lobes(x, index, lobes.shape[-3:])
        return cauchy_schwarz_divergence(prediction, target).mean()

    def evaluate_forward(self, batch):
        lobes = getattr(batch, "lobes", None)
        assert lobes is not None, (
            "TDPBRFNet learns the lobes but the batch carries none — set `dataset: use_lobes: true`")
        metrics = super().evaluate_forward(batch)
        # Folded into the DB-MTL combine as its own task, and logged as `<stage>_direction_loss`.
        flux = batch.ground_truth.flux.to(self.device)
        trained = torch.isfinite(flux).reshape(flux.shape[0], -1)
        self._extra_task_losses["direction"] = self.direction_loss(batch.input, lobes.to(self.device), trained)
        return metrics

    def get_custom_parameters(self):
        params = super().get_custom_parameters()
        params["direction_lobes"] = self._direction_lobes
        params["lobe_samples"] = self._lobe_samples
        params["lobe_hidden"] = self._lobe_hidden
        return params
