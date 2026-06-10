#include "radfield3d-nn/tcnn/utils/histogram_resample.h"

namespace rfnn::tcnn {
namespace utils {

namespace {

// One thread per output element. Align-corners bilinear sampling: output bin j
// maps to source position src = j * (source_bins-1) / (target_bins-1), then a
// linear blend of the two neighbouring source bins. Mirrors torch.grid_sample(
// mode='bilinear', align_corners=True) over x in [-1, 1] with zero rows never
// reaching the borders, so no padding term is required.
__global__ void resample_interp_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    uint32_t batch_size,
    uint32_t source_bins,
    uint32_t target_bins) {
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t total = static_cast<uint64_t>(batch_size) * target_bins;
    if (idx >= total) return;

    const uint32_t j = static_cast<uint32_t>(idx % target_bins);   // output bin
    const uint32_t n = static_cast<uint32_t>(idx / target_bins);   // row

    const float denom = (target_bins > 1) ? static_cast<float>(target_bins - 1) : 1.0f;
    const float src = static_cast<float>(j) * static_cast<float>(source_bins - 1) / denom;

    uint32_t s0 = static_cast<uint32_t>(floorf(src));
    const float frac = src - static_cast<float>(s0);
    uint32_t s1 = s0 + 1u;
    if (s0 >= source_bins) s0 = source_bins - 1u;
    if (s1 >= source_bins) s1 = source_bins - 1u;

    const float* row = in + static_cast<uint64_t>(n) * source_bins;
    out[idx] = (1.0f - frac) * row[s0] + frac * row[s1];
}

// One block per row: sum the resampled row and divide by it (L1 normalise).
// A zero-sum row is left as-is (matches the Python `where(sum==0, 1, sum)`).
__global__ void normalize_rows_kernel(
    float* __restrict__ out,
    uint32_t batch_size,
    uint32_t target_bins) {
    const uint32_t n = blockIdx.x;
    if (n >= batch_size) return;

    float* row = out + static_cast<uint64_t>(n) * target_bins;

    float local = 0.0f;
    for (uint32_t j = threadIdx.x; j < target_bins; j += blockDim.x) local += row[j];

    extern __shared__ float sdata[];
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2u; s > 0u; s >>= 1u) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    const float sum = sdata[0];
    const float inv = (sum == 0.0f) ? 1.0f : (1.0f / sum);
    for (uint32_t j = threadIdx.x; j < target_bins; j += blockDim.x) row[j] *= inv;
}

}  // namespace

void resample_histogram_bilinear(
    cudaStream_t stream,
    const float* in,
    float* out,
    uint32_t batch_size,
    uint32_t source_bins,
    uint32_t target_bins) {
    if (batch_size == 0u || target_bins == 0u || source_bins == 0u) return;

    const uint64_t total = static_cast<uint64_t>(batch_size) * target_bins;
    const uint32_t threads = 256u;
    const uint32_t blocks = static_cast<uint32_t>((total + threads - 1u) / threads);
    resample_interp_kernel<<<blocks, threads, 0, stream>>>(in, out, batch_size, source_bins, target_bins);

    // Power-of-two thread count <= 256 for the shared-memory reduction.
    uint32_t nthreads = 1u;
    while (nthreads < target_bins && nthreads < 256u) nthreads <<= 1u;
    normalize_rows_kernel<<<batch_size, nthreads, nthreads * sizeof(float), stream>>>(out, batch_size, target_bins);
}

}  // namespace utils
}  // namespace rfnn::tcnn
