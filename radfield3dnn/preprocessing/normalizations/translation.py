"""Patient-translation normalization for TranslationalInput batches.

The patient translation is stored in field metadata in metres (world space). This normalizer maps
each translated axis into [0, 1] using the translation ranges the dataset was GENERATED with, so
the network always sees the same [0, 1] scale as the other normalized beam parameters (distance,
collimation) regardless of the dataset's physical extent.

The ranges come from the dataset definition file (the JSON handed to the dataset generator), whose
``GeometryTransformations.<object>.Translation`` block lists per-axis [min, max] ranges, e.g.::

    "GeometryTransformations": {
        "patient": {
            "Translation": {"X": [-0.25, 0.25], "Y": [-0.5, 0.5]}
        }
    }

Axes absent from the definition (e.g. Z above) were never varied during generation: they carry no
information, so they are mapped to the constant 0.0. Ranges may also be given explicitly.
"""
import json
from pathlib import Path

import torch

from radfiled3d.pytorch.datasets.processing import DataProcessing
from radfield3dnn.rftypes import TrainingInputData

_AXES = ("X", "Y", "Z")


class TranslationNormalization(DataProcessing):
    def __init__(self, translation_ranges_m: dict[str, tuple[float, float]],
                 output_range: tuple[float, float] = (0.0, 1.0)):
        """translation_ranges_m: per-axis {"X": (min, max), ...} in metres. Axes not listed are
        treated as constant and mapped to the LOW end of `output_range` — 0.0 for the default
        (0, 1), which is what every existing checkpoint was trained with. A degenerate range
        (min == max) likewise maps there instead of dividing by zero.

        PER AXIS, from the generation ranges, because the axes are different physical quantities
        with different spans — on DS04-Angio-ArtisQ the table axis runs 1.4 m against 0.245 m for
        table height, so one shared scale would leave height occupying a fourteenth of the range an
        encoder resolves and make the model that much less sensitive to it. The cost is isotropy:
        equal normalized distances stop meaning equal physical ones. That is the right way round
        only because the translation is CONDITIONING, never added to a position in metres.

        output_range: (-1, 1) centres it for a Fourier-feature encoder, which resolves a full
        period at its lowest band and wants its input zero-centred; (0, 1) matches the other
        normalized beam parameters.
        """
        super().__init__()
        assert output_range[1] > output_range[0], f"output_range must increase, got {output_range}"
        self.output_range = (float(output_range[0]), float(output_range[1]))
        assert translation_ranges_m, "translation_ranges_m must define at least one axis"
        unknown = set(translation_ranges_m) - set(_AXES)
        assert not unknown, f"Unknown translation axes {unknown}; expected subset of {_AXES}"
        self.translation_ranges = {
            axis: (float(lo), float(hi)) for axis, (lo, hi) in translation_ranges_m.items()
        }
        for axis, (lo, hi) in self.translation_ranges.items():
            assert hi >= lo, f"Axis {axis}: expected min <= max, got ({lo}, {hi})"
        # Precomputed per-axis (lo, span) with span=0 marking constant axes.
        lows, spans = [], []
        for axis in _AXES:
            lo, hi = self.translation_ranges.get(axis, (0.0, 0.0))
            lows.append(lo)
            spans.append(hi - lo)
        self._lows = torch.tensor(lows, dtype=torch.float32)
        self._spans = torch.tensor(spans, dtype=torch.float32)

    def forward(self, x: TrainingInputData) -> TrainingInputData:
        translation = getattr(x.input, "translation", None)
        assert translation is not None, (
            "TranslationNormalization requires a TranslationalInput batch with a `translation` "
            "tensor (use_translation datasets); got input without one."
        )
        with torch.no_grad():
            assert translation.shape[-1] == 3, f"Expected (..., 3) translation, got {translation.shape}"
            lows = self._lows.to(device=translation.device)
            spans = self._spans.to(device=translation.device)
            unit = torch.where(
                spans > 0,
                (translation - lows) / spans.clamp_min(1e-12),
                torch.zeros_like(translation),
            )
            # Forgive float noise at the borders, but fail loudly on values truly outside the
            # generation range — that means the wrong definition file was configured.
            unit = torch.where(unit.isclose(torch.ones_like(unit)), torch.ones_like(unit), unit)
            unit = torch.where(unit.isclose(torch.zeros_like(unit)), torch.zeros_like(unit), unit)
            assert (unit >= 0.0).all() and (unit <= 1.0).all(), (
                f"Translation outside the configured generation ranges {self.translation_ranges}: "
                f"min {translation.min().item():.4f} m, max {translation.max().item():.4f} m. "
                f"Check the dataset definition file passed in the training config."
            )
            lo, hi = self.output_range
            normed = unit * (hi - lo) + lo
            return TrainingInputData(
                input=x.input._replace(translation=normed),
                ground_truth=x.ground_truth,
                original_ground_truth=x.original_ground_truth if hasattr(x, "original_ground_truth") else None,
                lobes=getattr(x, "lobes", None),
            )

    @classmethod
    def from_dataset_definition(cls, definition: str | Path | dict, object_name: str | None = None,
                                output_range: tuple[float, float] = (0.0, 1.0)) -> "TranslationNormalization":
        """Build from the dataset definition JSON (path or already-parsed dict) used to generate
        the dataset, reading ``GeometryTransformations.<object_name>.Translation``.

        object_name None takes the ONE object the definition translates — generators name it
        differently (``patient`` in DS04, ``patient_on_table`` in DS04-Angio-ArtisQ, which moves the
        mattress and desk with it). Several translated objects make that ambiguous, and it raises.
        """
        if not isinstance(definition, dict):
            with open(definition, "r") as f:
                definition = json.load(f)
        transforms = definition.get("GeometryTransformations", {})
        if object_name is None:
            translated = sorted(k for k, v in transforms.items() if isinstance(v, dict) and v.get("Translation"))
            assert len(translated) == 1, (
                f"expected exactly one translated object in GeometryTransformations, found {translated}; "
                f"name the one the model conditions on")
            object_name = translated[0]
        assert object_name in transforms, (
            f"Dataset definition has no GeometryTransformations entry for {object_name!r}; "
            f"found {sorted(transforms)}"
        )
        translation = transforms[object_name].get("Translation")
        assert translation, f"GeometryTransformations.{object_name} defines no Translation ranges"
        ranges = {axis: (float(r[0]), float(r[1])) for axis, r in translation.items()}
        return cls(ranges, output_range=output_range)

    @staticmethod
    def create_from_config(config: dict) -> "TranslationNormalization":
        return TranslationNormalization(
            translation_ranges_m=config["translation_ranges_m"],
            output_range=tuple(config.get("output_range", (0.0, 1.0))))

    def get_parameters(self):
        return {"translation_ranges_m": self.translation_ranges, "output_range": self.output_range}
