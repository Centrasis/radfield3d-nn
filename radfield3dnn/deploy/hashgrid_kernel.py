"""A tiny-cuda-nn multiresolution hash grid, as a self-contained CUDA kernel for an RF3M stage.

A tcnn `HashGrid` encoding has no ONNX lowering, so a model that uses one cannot ship as a single
graph: the export either refuses or emits a graph missing the encoder entirely. RF3M's answer is a
**stage** — the encoder travels as compiled code (`cuda_ptx`) beside the ONNX graphs, with its table
in the stage's weights buffer and a named buffer carrying its output to the trunk.

The kernel is generated per model rather than written once, because everything that varies between
configurations — level count, per-level resolution and table offsets, the input domain — is baked in
as a constant. Those constants are computed HERE, in float32, exactly as tcnn computes them; see
`grid_layout`. Nothing at runtime re-derives them, so nothing can derive them differently.

Correctness is not argued, it is measured: `tests/test_hashgrid_kernel.py` runs this kernel against
the live `tcnn.Encoding` on random points with a trained table and requires agreement to f32 round-off.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from typing import NamedTuple

import numpy as np
import torch

# tcnn's `coherent_prime_hash` factors, in order; a grid of N dimensions uses the first N. The
# leading 1 is not a typo: multiplying the first dimension by 1 keeps neighbouring cells along that
# axis adjacent in the table.
_HASH_PRIMES = (1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737)

# `grid_index` indexes densely only while resolution^N cannot overflow a uint32. tcnn's table gives
# the largest safe base per dimensionality; the index is N_DIMS, entry 0 is unused.
_MAX_DENSE_RESOLUTION = (0x0, 0xFFFFFFFF, 0xFFFF, 0x659, 0xFF, 0x54, 0x28, 0x17, 0xF, 0xB, 0x9)


def _log2_per_level_scale(per_level_scale: float) -> np.float32:
    """`log2(per_level_scale)` the way tcnn computes it: float32 in, float32 log, float32 out.

    tcnn holds `m_per_level_scale` as a float and calls `std::log2` on it, which selects the FLOAT
    overload. Taking the double log2 and narrowing afterwards is a different number — 2 ulp apart
    at `per_level_scale = 1.3819` — and those 2 ulp move every level's sampling scale, so the
    interpolation weights and occasionally the chosen cell come out different. Only exact powers of
    two survive the sloppier route, which is why a `2.0` configuration hides this bug completely.
    """
    return np.log2(np.float32(per_level_scale), dtype=np.float32)


class GridLayout(NamedTuple):
    """Per-level geometry of a hash grid, derived the way tcnn derives it."""
    resolutions: list        # grid vertices per axis, per level
    offsets: list            # first table entry of each level, plus the total at [n_levels]
    scales: list             # the float32 `grid_scale` each level samples at
    n_entries: int           # table entries in total (parameters = n_entries * features_per_level)


def grid_layout(n_levels: int, base_resolution: int, per_level_scale: float,
                log2_hashmap_size: int, n_dims: int = 3) -> GridLayout:
    """The level resolutions and table offsets tcnn computes for this configuration.

    Float32 throughout, because tcnn's is: `grid_scale` is `exp2f`, and its `log2_per_level_scale`
    reaches the kernel as a float. In float64 a level whose scale lands exactly on an integer —
    1.5^3 * 8 - 1 = 26 — rounds up instead, giving resolution 28 where tcnn has 27 and shifting
    every later level's offset. The whole table would then be read at the wrong place.
    """
    log2_scale = _log2_per_level_scale(per_level_scale)
    resolutions, offsets, scales = [], [], []
    offset = 0
    for level in range(n_levels):
        scale = np.float32(np.exp2(np.float32(np.float32(level) * log2_scale))
                           * np.float32(base_resolution) - np.float32(1.0))
        resolution = int(np.ceil(scale)) + 1
        dense = resolution ** n_dims
        # Aligned to 8 BEFORE the hashmap cap, in that order: capping first would leave levels
        # whose padded size exceeds the cap.
        params_in_level = min(-(-dense // 8) * 8, 1 << log2_hashmap_size)
        offsets.append(offset)
        offset += params_in_level
        resolutions.append(resolution)
        scales.append(float(scale))
    offsets.append(offset)
    return GridLayout(resolutions, offsets, scales, offset)


_SOURCE = r'''
// Generated for one hash-grid configuration; every constant below is baked.
#include <cuda_fp16.h>

#define N_LEVELS %(n_levels)d
#define N_FEATURES %(features)d
#define N_DIMS %(n_dims)d

__device__ __forceinline__ unsigned int grid_index(
        unsigned int hashmap_size, unsigned int resolution, const unsigned int pos_grid[N_DIMS]) {
    unsigned int stride = 1u, index = 0u;
    // Dense while the strided index cannot overflow; past that tcnn forces the hash by making the
    // stride unreachable rather than by branching on the level.
    if (resolution <= %(max_dense)uu) {
#pragma unroll
        for (int d = 0; d < N_DIMS; ++d) { index += pos_grid[d] * stride; stride *= resolution; }
    } else {
        stride = 0xFFFFFFFFu;
    }
    if (hashmap_size < stride) {
        index = %(hash_expr)s;
    }
    return index %% hashmap_size;
}

// RF3M kernel calling convention (docs/custom-code.md §4): weights, then inputs, then outputs,
// then the row count.
extern "C" __global__ void %(entry)s(
        const __half* __restrict__ weights,
        const float* __restrict__ positions,
        float* __restrict__ encoded,
        unsigned int rows) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;

    const unsigned int OFFSETS[N_LEVELS + 1] = { %(offsets)s };

    // Read through a volatile so the compiler cannot constant-fold the scale expression below.
    // Every operand is a literal here, and nvcc would evaluate `exp2f(level * log2)` at COMPILE
    // time in double precision; tcnn receives the same value as a kernel parameter and evaluates
    // it on the device in float. At `per_level_scale = 2.5, base = 32` the two differ either side
    // of 79.0, which flips `ceilf` and hence the resolution — and at that level the table is
    // exactly `resolution^3` entries, so the flip decides dense indexing versus hashing and every
    // value of the level comes from the wrong place.
    volatile float log2_scale_v = %(log2_scale)af;
    const float log2_per_level_scale = log2_scale_v;

    float x[N_DIMS];
#pragma unroll
    for (int d = 0; d < N_DIMS; ++d) {
        // tcnn's grid indexes the UNIT cube; a coordinate outside it wraps onto unrelated cells.
        float v = (positions[i * N_DIMS + d] - %(lo)af) * %(inv_span)af;
        x[d] = fminf(fmaxf(v, 0.0f), 1.0f);
    }

    for (unsigned int level = 0; level < N_LEVELS; ++level) {
        const __half* grid = weights + (size_t)OFFSETS[level] * N_FEATURES;
        const unsigned int hashmap_size = OFFSETS[level + 1] - OFFSETS[level];
        // Evaluated HERE with exp2f rather than baked as a decimal literal: tcnn computes it on
        // the device the same way, and a host `exp2` that lands one ulp away shifts every
        // interpolation weight — and, where the shifted value crosses an integer, the cell itself.
        const float scale = exp2f((float)level * log2_per_level_scale) * %(base_resolution)af - 1.0f;
        // Derived from THIS scale, on the device, exactly as tcnn's kernel does — never from a
        // host-computed constant. Where a level's scale lands on an integer, a host `exp2` one ulp
        // away flips `ceilf` and the resolution with it; tcnn then hashes a level this would index
        // densely (its table offsets come from the host, its resolution from the device, and at
        // that boundary the two genuinely disagree). Recomputing here reproduces tcnn including
        // that quirk, which is what bit-exactness against a trained table means.
        const unsigned int resolution = (unsigned int)ceilf(scale) + 1u;

        float pos[N_DIMS];
        unsigned int pos_grid[N_DIMS];
#pragma unroll
        for (int d = 0; d < N_DIMS; ++d) {
            // The 0.5 offset staggers the levels against each other (Instant-NGP appendix A); it
            // is part of the encoding, not a rounding convenience.
            const float p = fmaf(scale, x[d], 0.5f);
            const float t = floorf(p);
            pos_grid[d] = (unsigned int)(int)t;
            pos[d] = p - t;
        }

        // HALF accumulation, because that is what tcnn does. Its torch wrapper keeps an f32
        // master copy of the table for the optimizer and casts it to the native precision on
        // every forward, so the numbers the trained trunk actually saw came out of an fp16
        // accumulator. Summing the same corners in f32 here is MORE accurate and therefore wrong:
        // it disagrees with tcnn by ~2e-3 relative, which is the encoder's own quantisation.
        __half result[N_FEATURES];
#pragma unroll
        for (int f = 0; f < N_FEATURES; ++f) result[f] = __float2half(0.0f);

        // Corners in ascending index order, matching tcnn's accumulation order: in half a
        // different order is a different answer well above the last bits.
#pragma unroll
        for (unsigned int c = 0; c < (1u << N_DIMS); ++c) {
            float weight = 1.0f;
            unsigned int corner[N_DIMS];
#pragma unroll
            for (int d = 0; d < N_DIMS; ++d) {
                if ((c & (1u << d)) == 0u) { weight *= 1.0f - pos[d]; corner[d] = pos_grid[d]; }
                else                       { weight *= pos[d];        corner[d] = pos_grid[d] + 1u; }
            }
            const unsigned int index = grid_index(hashmap_size, resolution, corner) * N_FEATURES;
            // The weight is built in f32 and narrowed ONCE, as tcnn's `fma((T)weight, ...)` does.
            const __half hw = __float2half(weight);
#pragma unroll
            for (int f = 0; f < N_FEATURES; ++f) result[f] = __hfma(hw, grid[index + f], result[f]);
        }

        // Widened on the way out: the buffer the trunk graph reads is f32.
#pragma unroll
        for (int f = 0; f < N_FEATURES; ++f)
            encoded[(size_t)i * (N_LEVELS * N_FEATURES) + level * N_FEATURES + f] = __half2float(result[f]);
    }
}
'''


def cuda_source(encoder, entry_point: str = "hashgrid_encode") -> str:
    """CUDA for this encoder's configuration, with every constant baked in."""
    layout = grid_layout(encoder.n_levels, encoder.base_resolution,
                         encoder.per_level_scale, encoder.log2_hashmap_size, encoder.in_dim)
    lo, hi = encoder.input_domain
    return _SOURCE % {
        "n_levels": encoder.n_levels,
        "features": encoder.features_per_level,
        "entry": entry_point,
        "n_dims": encoder.in_dim,
        "max_dense": _MAX_DENSE_RESOLUTION[encoder.in_dim],
        "hash_expr": " ^ ".join(f"(pos_grid[{d}] * {_HASH_PRIMES[d]}u)" for d in range(encoder.in_dim)),
        "offsets": ", ".join(f"{o}u" for o in layout.offsets),
        "log2_scale": float(_log2_per_level_scale(encoder.per_level_scale)),
        "base_resolution": float(encoder.base_resolution),
        "lo": lo, "inv_span": 1.0 / (hi - lo),
    }


def compile_ptx(source: str, arch: str = "compute_75") -> bytes:
    """Compile to PTX for a VIRTUAL architecture, so one block runs on every NVIDIA device.

    PTX is JIT-compiled by the driver for whatever card is present, which is why the package
    records `target_arch=''` for it. A cubin would be faster to load and would have to be shipped
    per architecture.
    """
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not os.path.exists(nvcc):
        raise RuntimeError("nvcc is required to build the hash-grid stage; none on PATH")
    with tempfile.TemporaryDirectory(prefix="rfnn-ptx-") as tmp:
        cu, ptx = os.path.join(tmp, "hashgrid.cu"), os.path.join(tmp, "hashgrid.ptx")
        with open(cu, "w") as f:
            f.write(source)
        subprocess.run([nvcc, "-ptx", f"-arch={arch}", "-o", ptx, cu],
                       check=True, capture_output=True, text=True)
        with open(ptx, "rb") as f:
            return f.read()


def table_weights(encoder) -> np.ndarray:
    """The encoder's hash table, in the precision tcnn actually computes with.

    `Module.params` is an f32 master copy kept for the optimizer; `Module.forward` casts it to
    `param_precision()` — fp16 on every build here — before each call. The package must carry the
    cast copy, because that is the table the trained trunk was fed from.
    """
    native = getattr(encoder.encoding, "dtype", torch.float16)
    params = encoder.encoding.params.detach().to(native).cpu().numpy().ravel()
    layout = grid_layout(encoder.n_levels, encoder.base_resolution,
                         encoder.per_level_scale, encoder.log2_hashmap_size, encoder.in_dim)
    expected = layout.n_entries * encoder.features_per_level
    if params.size != expected:
        raise ValueError(
            f"hash table has {params.size} parameters but this configuration derives {expected}. "
            f"The level offsets would be wrong, so the table would be read at the wrong places.")
    return params
