"""Render a self-contained TailwindCSS page that maps every PUBLISHED deep-learning method used in
this repo to its canonical citation AND to the exact code that implements it. The catalog is curated
data (below), not scraped — keep it in sync when the architecture changes. Same visual language as the
fp16 / seed-stability study pages.

    python scripts/build_citations_html.py    ->  results/citations.html

Paths are absolute under the repo root so the file:// code links resolve when the page is opened
locally. Line numbers are point-in-time (2026-06-22, post module-tree de-duplication).
"""
from __future__ import annotations

import html
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_HTML = os.path.join(REPO, "results", "citations.html")

# Each section: (id, title, accent colour, [entries]). Each entry is a dict:
#   name, cite, what, usage, code=[(path, "symbol / note"), ...], status (optional: "missing").
SECTIONS = [
    ("encodings", "Input encodings", "#22c55e", [
        dict(name="NeRF positional / sinusoidal Fourier encoding",
             cite="Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi, Ng — “NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.” ECCV 2020.",
             what="Lifts low-dimensional coordinates into octave-frequency (sin, cos) features so an MLP can fit high-frequency detail.",
             usage="Default location encoder of the NeRF models ({\"type\":\"sinusoidal\"}); the pure-PyTorch path is numerically matched to tcnn’s “Frequency” encoding for ONNX export.",
             code=[("radfield3dnn/models/encoders/sinusoidal_encoding.py:10", "SinusoidalFrequencyEncoding (sin/cos :47, freq buffer 2^i·π :30); AngularSinusoidalFrequencyEncoding :54")]),
        dict(name="Random Fourier Features (RFF)",
             cite="Tancik, Srinivasan, Mildenhall, Fridovich-Keil, Raghavan, Singhal, Ramamoorthi, Barron, Ng — “Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains.” NeurIPS 2020.",
             what="Encodes coordinates as [sin(2πBx), cos(2πBx)] with a fixed Gaussian random matrix B; the bandwidth σ sets the representable frequency.",
             usage="Selectable location/coordinate encoder — the low-frequency counterpart to the hash grid.",
             code=[("radfield3dnn/models/encoders/rff_encoding.py:8", "RandomFourierFeatures (projection / sin / cos :40-41)")]),
        dict(name="Multiresolution hash-grid encoding (Instant-NGP)",
             cite="Müller, Evans, Schied, Keller — “Instant Neural Graphics Primitives with a Multiresolution Hash Encoding.” ACM SIGGRAPH 2022.",
             what="Multi-level learnable hash-table feature grids, trilinearly interpolated per level, for fast neural fields.",
             usage="Selectable location encoder for the tcnn-backed *CPP NeRF models; requires a tinycudann build.",
             code=[("radfield3dnn/models/encoders/hash_encoding.py:12", "HashGridEncoding (tcnn otype:\"HashGrid\" :37); VoxelHashGridEncoding :76"),
                   ("radfield3dnn/models/nerf_cpp.py:216", "wiring into PBRFNetCPP / SPERFNetCPP")]),
        dict(name="Spherical-harmonics directional encoding",
             cite="Real-SH directional basis as popularized for neural fields by Müller et al., Instant-NGP (SIGGRAPH 2022); standard real spherical-harmonics basis.",
             what="Encodes a unit direction vector in the real spherical-harmonics basis (degree ≤4 → up to 16 coefficients).",
             usage="Default beam-direction encoder for the NeRF family (degree 4); the pure-PyTorch class is ONNX-exportable and matched to tcnn’s SphericalHarmonics.",
             code=[("radfield3dnn/models/encoders/spherical_hamonics.py:42", "TorchSphericalHarmonics (basis math _sh_components :13); TcnnSphericalHarmonics :72")]),
    ]),
    ("fusion", "Conditioning / fusion", "#3b82f6", [
        dict(name="FiLM — Feature-wise Linear Modulation",
             cite="Perez, Strub, de Vries, Dumoulin, Courville — “FiLM: Visual Reasoning with a General Conditioning Layer.” AAAI 2018.",
             what="Conditions features by a learned per-channel affine transform γ·x + β derived from a conditioning vector.",
             usage="Default beam/spectrum conditioning of the NeRF trunk (\"FiLM\" / \"ResFiLM\"); also the spectrum conditioning in both U-Nets and nerf_cpp (beam_fusion=\"film\").",
             code=[("radfield3dnn/models/layers/fusions/film.py:10", "FiLM (identity-start γ=1+tanh(...) :61); ResidualFiLM :68"),
                   ("radfield3dnn/models/field_unet.py:59", "_FiLM3D CNN variant (γ=1+tanh :71)")]),
        dict(name="Concat conditioning / early fusion",
             cite="Standard concat-conditioning baseline (no single canonical paper; framed against attention by Rebain et al. 2023, below).",
             what="Concatenates the main and conditioning vectors and projects back, giving the condition a full-rank additive voice.",
             usage="Conditioning option \"Concat\" in the NeRF family.",
             code=[("radfield3dnn/models/layers/fusions/concat.py:8", "ConcatLinear (identity-init :35)")]),
        dict(name="Cross-attention conditioning of neural fields",
             cite="Rebain, Matthews, Yi, Sharma, Lagun, Tagliasacchi — “Attention Beats Concatenation for Conditioning Neural Fields.” TMLR 2023.",
             what="A voxel query attends over tokens projected from the conditioning vector; the attended value is residually added.",
             usage="Conditioning options \"Attention\" / \"TokenAttention\" in the NeRF family; output zero-init for an identity start.",
             code=[("radfield3dnn/models/layers/fusions/attention.py:8", "CrossAttentionFusion; TokenCrossAttentionFusion :62 (paper cited in docstring :70)")]),
        dict(name="Gated multimodal fusion (GMU / GRU-style)",
             cite="Arevalo, Solorio, Montes-y-Gómez, González — “Gated Multimodal Units for Information Fusion.” ICLR Workshop 2017.",
             what="A learned per-channel sigmoid-gated convex blend between the transformed main feature and the conditioning.",
             usage="Conditioning option \"Gated\" in the NeRF family.",
             code=[("radfield3dnn/models/layers/fusions/gates.py:72", "GatedFusion (also ModulativeSigmoidGate :8, ResidualAdditiveTanhGate :47)")]),
    ]),
    ("arch", "Architectures", "#a855f7", [
        dict(name="NeRF / implicit neural field (coordinate MLP)",
             cite="Mildenhall et al. — “NeRF.” ECCV 2020.",
             what="Maps continuous (xyz, beam-params) through an MLP to per-point field values (an implicit, coordinate-based field).",
             usage="Core per-voxel flux+spectrum predictor family; assembled into a volume by forward2volume.",
             code=[("radfield3dnn/models/nerf.py:24", "RFNetBase / RFBackboneModel; SRBFNet :298, SPERFNet :461, PBRFNet :531; trunk residual skip :294"),
                   ("radfield3dnn/models/feedforward.py", "forward2volume volume assembly")]),
        dict(name="Instant-NGP fused-MLP training recipe (fp32 master weights)",
             cite="Müller, Evans, Schied, Keller — Instant-NGP, SIGGRAPH 2022 (code cites NVlabs/instant-ngp configs/nerf/base.json).",
             what="A tcnn fused fp16 MLP trained with fp32 master weights; AdamW betas (0.9, 0.99), eps 1e-15, weight decay 1e-6.",
             usage="PBRFNetCPP / SPERFNetCPP, the tcnn-backed NeRF variants (require a tinycudann build).",
             code=[("radfield3dnn/models/nerf_cpp.py:27", "_TcnnModule (fp32 masters :69-92, on_after_backward :313, optimizer recipe :345-396, cite :351)")]),
        dict(name="U-Net (encoder–decoder with skip connections)",
             cite="Ronneberger, Fischer, Brox — “U-Net: Convolutional Networks for Biomedical Image Segmentation.” MICCAI 2015.",
             what="A symmetric down/up-sampling convnet with encoder→decoder skip concatenations.",
             usage="Field-wise CNNs predicting the whole scatter volume at once (3D convolution).",
             code=[("radfield3dnn/models/cnn.py:232", "Beam2ScatterUNet (skip concat :343, layer builder :375)"),
                   ("radfield3dnn/models/field_unet.py:91", "FieldScatterUNet (skip concat :181)")]),
        dict(name="Attention U-Net (attention gates)",
             cite="Oktay, Schlemper, Le Folgoc, Lee, Heinrich, Misawa, Mori, McDonagh, Hammerla, Kainz, Glocker, Rueckert — “Attention U-Net: Learning Where to Look for the Pancreas.” MIDL 2018.",
             what="Gates the encoder skip features by an attention coefficient computed from the coarser decoder signal.",
             usage="Applied to every skip in FieldScatterUNet to focus on the sparse high-flux region.",
             code=[("radfield3dnn/models/field_unet.py:76", "_AttentionGate (cites Oktay), applied :145, :180")]),
        dict(name="Residual learning (ResNet-style)",
             cite="He, Zhang, Ren, Sun — “Deep Residual Learning for Image Recognition.” CVPR 2016.",
             what="Adds a block’s input (or a 1×1 projection of it) to its output to ease gradient flow.",
             usage="Trunk residual in the NeRF backbone; residual conv blocks form the FieldScatterUNet body.",
             code=[("radfield3dnn/models/nerf.py:294", "trunk residual"),
                   ("radfield3dnn/models/field_unet.py:43", "_ResBlock3D (:56)"),
                   ("radfield3dnn/models/cnn.py:51", "ResidualDeConvBlock3D")]),
    ]),
    ("losses", "Losses", "#f59e0b", [
        dict(name="SSIM loss (3D, single-scale)",
             cite="Wang, Bovik, Sheikh, Simoncelli — “Image Quality Assessment: From Error Visibility to Structural Similarity.” IEEE TIP 2004.",
             what="Returns 1 − windowed structural similarity between the predicted and target volume.",
             usage="Optional curated loss; one Gaussian-window pass over the 3D field.",
             code=[("radfield3dnn/losses/std.py:148", "StructuralSimilarity3DLoss (_ssim_3d :197)")]),
        dict(name="Histogram / Wasserstein (EMD) spectrum loss",
             cite="Earth-Mover’s Distance via cumulative-sum — Rubner, Tomasi, Guibas — “The Earth Mover’s Distance as a Metric for Image Retrieval.” IJCV 2000.",
             what="Compares normalized spectral histograms via |cumsum(pred) − cumsum(target)| (1-D EMD) plus an L1 term.",
             usage="Default spectrum-channel loss; default split 0.7·EMD + 0.3·L1.",
             code=[("radfield3dnn/losses/combinations.py:10", "HistogramLoss"),
                   ("radfield3dnn/losses/std.py:132", "WassersteinLossWeighted (cumsum :139)")]),
        dict(name="RawNeRF HDR loss (self-normalizing relative L2)",
             cite="Mildenhall, Hedman, Martin-Brualla, Srinivasan, Barron — “NeRF in the Dark: High Dynamic Range View Synthesis from Noisy Raw Images.” CVPR 2022.",
             what="Linear-space L2 weighted by a stop-gradient relative factor, so every decade receives gradient without a log transform.",
             usage="Optional HDR flux loss paired with LinearNormalizer(0,1).",
             code=[("radfield3dnn/losses/std.py:96", "RawNeRFLoss (cited in docstring; weight :122)")]),
    ]),
    ("metrics", "Metrics", "#14b8a6", [
        dict(name="SSIM / MS-SSIM / gradient-SSIM",
             cite="Wang et al. 2004 (SSIM); Wang, Simoncelli, Bovik — “Multiscale Structural Similarity.” Asilomar 2003; Chen, Yang, Xie — “Gradient-based SSIM.” ICIP 2006.",
             what="Structural-similarity scoring of two volumes.",
             usage="Air-kerma volume comparison via AirkermaSSIM. MultiLevelSSIM is a simplified uniform-average multi-scale (not exact MS-SSIM).",
             code=[("radfield3dnn/metrics/ssim.py:10", "SSIM3D; MultiLevelSSIM :66; GradientSSIM3D :87; AirkermaSSIM :149")]),
        dict(name="PSNR",
             cite="Peak signal-to-noise ratio — standard image-quality metric.",
             what="10·log10(1/MSE) on [0,1]-normalized fields.",
             usage="Reconstruction-fidelity metric on air-kerma fields.",
             code=[("radfield3dnn/metrics/psnr.py:9", "PSNR (formula :17); AirkermaPSNR :21")]),
        dict(name="NCC (normalized cross-correlation / Pearson)",
             cite="Normalized cross-correlation — e.g. Lewis, “Fast Normalized Cross-Correlation,” Vision Interface 1995.",
             what="Pearson correlation between target and prediction over valid voxels.",
             usage="metric_type='ncc' option in the air-kerma accuracy wrappers.",
             code=[("radfield3dnn/metrics/ncc.py:8", "NCCAccuracy (_calc_metric :15)")]),
        dict(name="SMAPE — symmetric mean absolute percentage error",
             cite="Makridakis — “Accuracy measures: theoretical and practical concerns.” Int. J. Forecasting 1993.",
             what="2|p−t|/(|p|+|t|) folded to an accuracy (1 − ½·SMAPE), clamped.",
             usage="The primary reported accuracy metric — physical-space air-kerma SMAPE.",
             code=[("radfield3dnn/metrics/smape.py:8", "SMAPEAccuracy (formula :62); EnergyWeightedSMAPEAccuracy :126"),
                   ("radfield3dnn/metrics/airkerma_accuracy.py", "air-kerma masking + region split")]),
        dict(name="Gamma index / Gamma pass rate",
             cite="Low, Harms, Mutic, Purdy — “A technique for the quantitative evaluation of dose distributions.” Med. Phys. 1998 (criteria per AAPM TG-218).",
             what="Per-voxel minimum over a distance-to-agreement sphere combining dose-difference and distance terms; reports the fraction with γ ≤ 1.",
             usage="metric_type='gpr' in AirkermaAccuracy — medical-physics dose comparison of predicted vs GT air-kerma (default 3%/3 mm).",
             code=[("radfield3dnn/metrics/gpr.py:8", "GammaPassingRate (gamma_index :17)")]),
    ]),
    ("norm", "Normalization (HDR)", "#ef4444", [
        dict(name="Asinh tonemapping (bounded HDR)",
             cite="Inverse-hyperbolic-sine stretch — Lupton, Blanton, Fink, Hogg — “Preserving the Detail in Astronomical Images.” PASP 2004 (generic HDR tonemap).",
             what="y = asinh(x/σ) / asinh(1/σ): linear near 0, logarithmic for x ≫ σ, bounded to [0,1], with a closed-form sinh inverse.",
             usage="Flux normalizer options asinh / asinh_split / asinh_auto in the model JSON.",
             code=[("radfield3dnn/preprocessing/normalizations/asinh.py:8", "AsinhTonemapNormalizer (forward :68, inverse :92)")]),
        dict(name="Log-scale HDR normalization (log1p / expm1)",
             cite="Standard log / HDR compression (no single canonical paper).",
             what="log1p(x·input_scale) over a LinearNormalizer base; invertible via expm1. CLAUDE.md notes it under-fits the high-flux beam peak vs linear.",
             usage="Present as LogNormalizer in the deep tree (subclass of the canonical LinearNormalizer). Caveat: the string-name factory only wires linear0_1 / asinh, and tests still import a separate preprocessing.normalizations.logscale.LogScaleNormalizer that does not exist — see the code-audit page.",
             code=[("radfield3dnn/preprocessing/normalizations/lognormalizer.py:7", "LogNormalizer (log :64, inverse :69)"),
                   ("radfield3dnn/preprocessing/normalizations/__init__.py", "construct_by_name wires only linear0_1 / asinh (no log_scale branch)")]),
    ]),
    ("train", "Training / optimization", "#0ea5e9", [
        dict(name="DB-MTL — Dual-Balancing Multi-Task Learning",
             cite="Lin, Ye, Zhang, Tsang — “Dual-Balancing for Multi-Task Learning.” arXiv:2308.12029, 2023.",
             what="Two-step balancing: log-scale loss balancing plus gradient-magnitude balancing of each task’s gradient w.r.t. the shared trunk.",
             usage="Balances the flux vs spectrum tasks; gradient norms are EMA-smoothed/clamped and recomputed every update_every=16 steps. (No GradNorm / PCGrad / uncertainty weighting.)",
             code=[("radfield3dnn/losses/mtl/mtl.py:44", "MultiTaskLossBalancer (cited at top of file; gradient step combine :93)")]),
        dict(name="Linear-warmup → cosine-annealing LR schedule",
             cite="Warmup — Goyal et al., “Accurate, Large Minibatch SGD,” 2017; cosine annealing — Loshchilov & Hutter, “SGDR,” ICLR 2017.",
             what="Linear LR warmup followed by cosine decay (per step); AdamW with separate encoding / MLP / no-decay parameter groups.",
             usage="Default optimizer + schedule for the NeRF models.",
             code=[("radfield3dnn/optim/cosine_warmup.py:15", "CosineWithWarmup (LinearLR :94, CosineAnnealingLR :101, SequentialLR :104)")]),
    ]),
    ("physics", "Physics priors / sampling", "#84cc16", [
        dict(name="Analytic direct beam: inverse-square (1/r²) + Beer–Lambert",
             cite="Domain physics priors — inverse-square law; Beer–Lambert attenuation (no DL citation).",
             what="Reconstructs the primary beam analytically: point-source 1/r² falloff × spectrum-weighted air transmission × diverging-rectangle in-beam mask × phantom shadow, so the network only learns scatter.",
             usage="Supplies the physical direct beam for ONNX-deployable scatter-only learning; magnitude recombination via ScatterScaleHead.",
             code=[("radfield3dnn/models/layers/analytic_direct.py:27", "AnalyticDirectBeam (1/r² :131, Beer–Lambert :125)"),
                   ("radfield3dnn/models/layers/scatter_scale.py:22", "ScatterScaleHead"),
                   ("radfield3dnn/preprocessing/analytic_direct_beam.py:50", "NumPy reference")]),
        dict(name="Error-based importance sampling",
             cite="Importance sampling (generic; no single canonical DL paper).",
             what="Stochastically drops voxels inversely to their Monte-Carlo error (keeping high-error / high-flux voxels), with an annealed drop schedule.",
             usage="Training-time augmentation; dropped voxels masked to −inf so the loss ignores them.",
             code=[("radfield3dnn/preprocessing/augmentations/importance_sampling.py:7", "ErrorbasedImportanceSampler (forward :46)")]),
        dict(name="ROI / region-of-interest sampling",
             cite="ROI sampling (generic).",
             what="Samples voxels by physical region (beam / scatter / floor) rather than uniformly — full beam, random scatter subset, capped floor subset.",
             usage="Structural twin of the importance sampler with a scheduled scatter ratio for fine-tuning.",
             code=[("radfield3dnn/preprocessing/augmentations/roi_sampling.py:26", "ROIbasedSampler (_keep_masks :95)"),
                   ("radfield3dnn/roi.py", "compute_roi_masks")]),
    ]),
]

# Referenced in docs/comments but not actually implemented in the current source.
NOT_IMPLEMENTED = [
    dict(name="EMA of model weights",
         cite="Polyak & Juditsky, 1992 (weight averaging).",
         note="No weight EMA. The only EMA in the codebase is of gradient norms inside DB-MTL (callbacks/ema.py should be verified by the audit)."),
    dict(name="GradNorm / PCGrad / uncertainty weighting",
         cite="Chen et al. 2018 / Yu et al. 2020 / Kendall et al. 2018.",
         note="Not present; DB-MTL is the only multi-task balancer."),
]


def _e(s):
    return html.escape(str(s))


def _code_chip(path_line: str, note: str) -> str:
    abspath = os.path.join(REPO, path_line.split(":")[0])
    return (f'<div class="flex items-start gap-2 text-xs py-0.5">'
            f'<a class="font-mono text-emerald-400 hover:underline shrink-0" '
            f'href="file://{abspath}">{_e(path_line)}</a>'
            f'<span class="text-slate-400">{_e(note)}</span></div>')


def _card(e: dict, accent: str) -> str:
    missing = e.get("status") == "missing"
    ring = "ring-red-500/40" if missing else "ring-white/10"
    badge = ('<span class="ml-2 text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded '
             'bg-red-500/20 text-red-300 align-middle">not in live tree</span>') if missing else ""
    code = "".join(_code_chip(p, n) for p, n in e["code"])
    return f"""
      <div class="bg-slate-800/40 rounded-2xl ring-1 {ring} p-5">
        <div class="flex items-center gap-2 mb-1">
          <span class="w-2.5 h-2.5 rounded-sm shrink-0" style="background:{accent}"></span>
          <h3 class="text-base font-semibold text-slate-100">{_e(e['name'])}</h3>{badge}
        </div>
        <p class="text-sm text-slate-300 mb-2">{_e(e['what'])}</p>
        <p class="text-xs text-slate-400 mb-3"><span class="text-slate-500">Cite:</span> {_e(e['cite'])}</p>
        <div class="border-t border-white/10 pt-2 mb-2">{code}</div>
        <p class="text-xs text-slate-400"><span class="text-slate-500">In this repo:</span> {_e(e['usage'])}</p>
      </div>"""


def _section(sid, title, accent, entries) -> str:
    cards = "\n".join(_card(e, accent) for e in entries)
    return f"""
    <section id="{sid}" class="mt-10">
      <div class="flex items-center gap-3 mb-4">
        <span class="w-3 h-8 rounded" style="background:{accent}"></span>
        <h2 class="text-xl font-bold text-white">{title}</h2>
        <span class="text-xs text-slate-500">{len(entries)} method{'s' if len(entries)!=1 else ''}</span>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-5">{cards}</div>
    </section>"""


def _not_impl() -> str:
    rows = "".join(
        f"""
        <div class="bg-slate-900/60 rounded-xl ring-1 ring-amber-500/20 p-4">
          <h3 class="text-sm font-semibold text-amber-200">{_e(e['name'])}</h3>
          <p class="text-xs text-slate-400 mt-1"><span class="text-slate-500">Cite:</span> {_e(e['cite'])}</p>
          <p class="text-xs text-slate-300 mt-2">{_e(e['note'])}</p>
        </div>""" for e in NOT_IMPLEMENTED)
    return f"""
    <section id="missing" class="mt-12">
      <div class="flex items-center gap-3 mb-4">
        <span class="w-3 h-8 rounded bg-amber-500"></span>
        <h2 class="text-xl font-bold text-white">Referenced but NOT implemented</h2>
      </div>
      <p class="text-sm text-slate-400 mb-4 max-w-3xl">Named in CLAUDE.md / scratch notes / docstrings,
        but with no implementation in the current source — listed so the citations stay honest.</p>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">{rows}</div>
    </section>"""


def main():
    total = sum(len(e) for _, _, _, e in SECTIONS)
    nav = " ".join(
        f'<a href="#{sid}" class="text-xs text-slate-300 hover:text-white px-2 py-1 rounded '
        f'ring-1 ring-white/10 hover:ring-white/30">{title}</a>'
        for sid, title, _, _ in SECTIONS)
    body = "\n".join(_section(*s) for s in SECTIONS) + "\n" + _not_impl()
    html = _PAGE.format(nav=nav, body=body, total=total)
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w") as f:
        f.write(html)
    print(f"Wrote {OUT_HTML}  ({total} methods across {len(SECTIONS)} sections + {len(NOT_IMPLEMENTED)} not-implemented)")


_PAGE = """<!doctype html>
<html lang="en"><head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RadField3D-NN — methods &amp; citations</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-900 text-slate-200 antialiased">
  <div class="max-w-6xl mx-auto px-6 py-10">
    <header class="mb-6">
      <h1 class="text-3xl font-bold text-white">RadField3D-NN — published methods &amp; citations</h1>
      <p class="text-slate-400 mt-2 max-w-3xl">Every named, published deep-learning technique used in
        this repository, mapped to its canonical paper and to the exact code that implements it
        (<span class="font-mono text-emerald-400">file:line</span> links resolve when this page is
        opened locally). {total} methods. Line numbers are a point-in-time snapshot
        (2026-06-22, after the duplicate module-tree clean-up); the authoritative implementations live
        under <span class="font-mono text-slate-300">radfield3dnn/models/</span> and
        <span class="font-mono text-slate-300">radfield3dnn/preprocessing/</span>.</p>
      <div class="flex flex-wrap gap-2 mt-4">{nav}</div>
    </header>
    {body}
    <footer class="mt-12 pt-6 border-t border-white/10 text-xs text-slate-500">
      Citations are canonical where the implementing code names the paper; otherwise the closest
      standard reference is given. Treat author/year as best-effort for the generically-named techniques
      (concat fusion, log-scale, ROI/importance sampling, NCC).
    </footer>
  </div>
</body></html>"""


if __name__ == "__main__":
    main()
