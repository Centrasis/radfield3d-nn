"""End-to-end CPU smoke test of the TPBRFNet training path.

Every TPBRFNet failure so far surfaced only when the REAL Lightning machinery ran (batch-size
search at fit start, the LR finder, a training step, validation) — never in a unit test of the
model alone. This test drives that machinery on a synthetic TranslationalInput dataset: a
LightningDataModule (wrapped in a prefetcher-shaped iterable, like the CUDA path), the LR finder,
and a short fit with train + validation.

Deliberately tiny (8 fields, 8^3 voxels, d_model=32) so it runs in seconds on CPU.
"""
import pytest
import torch
import torch.nn.functional as F

try:
    import RadFiled3D  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip("RadFiled3D not available", allow_module_level=True)

import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from torch.utils.data import DataLoader, Dataset

from RadFiled3D.pytorch.types import RadiationFieldChannel, TranslationalInput
from radfield3dnn.models import ModelConstructor
from radfield3dnn.rftypes import RadiationField, TrainingInputData

VC = (8, 8, 8)
OUT_BINS = 32
IN_BINS = 150          # the dataset's tube-spectrum width (rebinned internally to in_spectra_dim)


def _model_config():
    return {
        "model_name": "TPBRFNet",
        "parameters": {
            "normalizer": "linear0_1", "d_model": 32, "out_spectra_dim": OUT_BINS,
            "trunk_depth": 2, "flux_activation": "sigmoid", "flux_loss": "SMAPEBalanced",
            "spectrum_loss": "HistogramLoss",
            "location_encoding_params": {"type": "sinusoidal", "pos_enc_dim": 6, "append_input": True},
            "direction_encoding_params": {"type": "spherical_harmonics", "degree": 4, "append_input": True},
            "spectra_encoding_params": {"type": "simple", "in_spectra_dim": OUT_BINS,
                                        "encoded_spectra_dims": OUT_BINS, "input_spectra_dim": IN_BINS},
            # rectangle collimation + Fourier translation encoding: the deployed ds04 recipe
            "conditioning_params": {"type": "Concat", "use_beam_shape": True,
                                    "beam_shape_param_dims": 2,
                                    "translation_encoding": {"type": "fourier", "n_frequencies": 6,
                                                             "append_input": True}},
            "training_params": {"learning_rate": 1.0e-3, "max_lr": 5.0e-4,
                                "voxel_supersampling": 1, "voxels_centered_around_origin": False},
        },
    }


class _FakeFields(Dataset):
    """Mimics RadField3DTranslationDataset: TranslationalInput + a whole-volume ground truth."""

    def __len__(self):
        return 8

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        inp = TranslationalInput(
            direction=F.normalize(torch.randn(3, generator=g), dim=-1),
            origin=torch.rand(1, generator=g),                       # normalized source distance
            spectrum=torch.rand(IN_BINS, generator=g).softmax(-1),
            translation=torch.rand(3, generator=g),                  # normalized [0, 1]
            beam_shape_type=torch.full((1,), 3.0),                   # FieldShape.RECTANGLE
            beam_shape_parameters=torch.rand(2, generator=g) * 0.15,
        )
        gt = RadiationField(
            scatter_field=RadiationFieldChannel(
                flux=torch.rand(*VC, generator=g),
                spectrum=torch.rand(OUT_BINS, *VC, generator=g).softmax(0),
                error=None),
            direct_beam=None, geometry=None)
        return TrainingInputData(input=inp, ground_truth=gt, original_ground_truth=gt)


def _collate(items):
    def stack(vals):
        if vals[0] is None:
            return None
        if isinstance(vals[0], torch.Tensor):
            return torch.stack(vals)
        return type(vals[0])(*[stack([getattr(v, f) for v in vals]) for f in vals[0]._fields])
    return TrainingInputData(*[stack([getattr(it, f) for it in items]) for f in TrainingInputData._fields])


class _PrefetcherLike:
    """Same object protocol as CudaStreamPrefetcher (an iterable wrapper, NOT a DataLoader) so the
    CUDA prefetch path's shape is exercised on CPU."""

    def __init__(self, loader):
        self.loader = loader
        self._it = None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self._it = iter(self.loader)
        return self

    def __next__(self):
        return next(self._it)


class _FakeDM(pl.LightningDataModule):
    def __init__(self, batch_size=2):
        super().__init__()
        self.batch_size = batch_size

    def _dl(self):
        return _PrefetcherLike(DataLoader(_FakeFields(), batch_size=self.batch_size, collate_fn=_collate))

    def train_dataloader(self):
        return self._dl()

    def val_dataloader(self):
        return self._dl()


def _build():
    torch.manual_seed(0)
    model = ModelConstructor.create_model_from_dict(_model_config())()
    model.max_inner_batch_size = 4096
    return model


def _trainer(**kw):
    return pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
                      enable_model_summary=False, **kw)


def test_training_step_runs_on_translational_batches():
    model = _build()
    trainer = _trainer(max_steps=2, enable_checkpointing=False, num_sanity_val_steps=0,
                       limit_train_batches=2, limit_val_batches=1, max_epochs=1)
    trainer.fit(model, datamodule=_FakeDM())
    assert trainer.state.finished


def test_validation_runs():
    model = _build()
    trainer = _trainer(max_epochs=1, enable_checkpointing=False, num_sanity_val_steps=2,
                       limit_train_batches=1, limit_val_batches=2)
    trainer.fit(model, datamodule=_FakeDM())
    assert trainer.state.finished


def test_lr_finder_runs():
    # tasks/train.py runs the LR finder on its own Trainer (max_steps set, NO max_epochs) before
    # the real fit — the configuration the scheduler's estimated_stepping_batches sees there.
    model = _build()
    lr_trainer = _trainer(max_steps=8, enable_checkpointing=True, num_sanity_val_steps=0)
    result = Tuner(lr_trainer).lr_find(model, datamodule=_FakeDM(), min_lr=1e-4, max_lr=1e-2,
                                       num_training=8)
    assert result is not None
    # documents the finder trainer's shape: max_steps-driven, so max_epochs is unset/infinite
    # (Lightning normalizes the unset value to -1) — which is what makes
    # trainer.estimated_stepping_batches fall back to max_steps in build_warmup_cosine_schedule.
    assert lr_trainer.max_epochs in (None, -1)
    suggestion = result.suggestion()
    assert suggestion is None or suggestion > 0


def test_activation_estimate_scales_with_points_not_chunks():
    # The whole-field step retains every voxel's activations for backward, so the estimate must
    # scale linearly in batch_size and in voxel count — and must NOT depend on max_inner_batch_size
    # (chunking bounds transient peak memory only). This is the check that would have flagged
    # batch_size=32 at 64^3 (~67 GiB) before it OOM'd on an 80 GiB card.
    model = _build()
    base = model.estimate_step_activation_bytes(1, (16, 16, 16))
    assert model.estimate_step_activation_bytes(4, (16, 16, 16)) == 4 * base
    assert model.estimate_step_activation_bytes(1, (32, 16, 16)) == 2 * base
    model.max_inner_batch_size = 128
    assert model.estimate_step_activation_bytes(1, (16, 16, 16)) == base
    # sanity on magnitude: a d_model=32 / trunk=2 model keeps a few KB per point
    per_point = base / (16 ** 3)
    assert 100 < per_point < 100_000


def test_batch_size_search_at_fit_start():
    # max_inner_batch_size unset -> on_fit_start runs _search_optimal_batch_size, which feeds
    # _generate_random_input through the model's own encoders (the original crash site: the random
    # input had no patient translation).
    #
    # The search doubles the inner batch until CUDA reports OOM; on CPU there is no such signal,
    # so the OOM is injected after a few rounds to make the loop terminate deterministically.
    torch.manual_seed(0)
    model = ModelConstructor.create_model_from_dict(_model_config())()
    assert model.max_inner_batch_size is None

    real_eval = model._search_optimal_batch_size_evaluate_forward
    rounds = {"n": 0}

    def bounded(batch):
        rounds["n"] += 1
        if rounds["n"] > 3:
            raise torch.cuda.OutOfMemoryError("injected: CPU has no OOM signal to end the search")
        return real_eval(batch)

    model._search_optimal_batch_size_evaluate_forward = bounded
    model._search_optimal_batch_size()
    assert rounds["n"] > 1                    # the real encoders ran on the random input
    assert model.max_inner_batch_size is not None and model.max_inner_batch_size >= 2
