#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace rfnn::tcnn {
namespace utils {

// Resample a batch of histograms from `source_bins` to `target_bins` with
// align-corners bilinear interpolation, then L1-normalise every row (rows that
// sum to zero are left untouched). This is the CUDA port of the Python
// `radfield3dnn.utils.mean_sampling.resample_histogram_bilinear`, provided so
// the fused PBRFNetCPP / SPERFNetCPP encoders can resample the raw beam spectrum
// to their `in_spectra_dim` during **pure-C++ inference** (no Python / torch
// grid_sample needed).
//
// Raw-pointer entry point — usable in a Python-free deployment. `in` and `out`
// are device pointers; `in` is (N, source_bins) and `out` is (N, target_bins),
// both row-major contiguous. Asynchronous on `stream`.
void resample_histogram_bilinear(
    cudaStream_t stream,
    const float* in,
    float* out,
    uint32_t batch_size,
    uint32_t source_bins,
    uint32_t target_bins);

}  // namespace utils
}  // namespace rfnn::tcnn
