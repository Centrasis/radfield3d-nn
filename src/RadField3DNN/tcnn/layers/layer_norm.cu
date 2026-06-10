#include "radfield3d-nn/tcnn/layers/layer_norm.h"
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/cpp_api.h>


rfnn::tcnn::LayerNorm::LayerNorm(uint32_t channels, float eps)
    : channels(channels),
      eps(eps),
      NEXT_MULTIPLE_FOR_TYPE((sizeof(::tcnn::network_precision_t) == 2) ? 8 : 4)
{
    ::tcnn::cpp::rtc_set_cache_dir("/tmp/rtc_ln");
    this->set_jit_fusion(true);
}

size_t rfnn::tcnn::LayerNorm::n_params() const {
    return 2u * this->channels;
}

std::vector<std::pair<uint32_t, uint32_t>> rfnn::tcnn::LayerNorm::layer_sizes() const {
    return { {this->channels, 2u} };  // (gamma, beta) per channel
}

void rfnn::tcnn::LayerNorm::set_params_impl(::tcnn::network_precision_t* params, ::tcnn::network_precision_t* inference_params, ::tcnn::network_precision_t* gradients) {
    this->params_ptr = params;
    this->inference_params_ptr = inference_params;
    this->gradients_ptr = gradients;
}

void rfnn::tcnn::LayerNorm::initialize_params(::tcnn::pcg32& /*rnd*/, float* params_full_precision, float /*scale*/) {
    // gamma = 0 (so 1 + gamma = 1), beta = 0  →  identity at init.
    // params_full_precision is a device pointer (allocated via GPUMemory in
    // the trainer); writing to it from host would be an illegal access and
    // hang the next CUDA call, so use cudaMemset.
    CUDA_CHECK_THROW(cudaMemset(params_full_precision, 0, this->n_params() * sizeof(float)));
}

uint32_t rfnn::tcnn::LayerNorm::device_function_fwd_ctx_bytes() const {
    // Stash mean (1), inv_std (1), and normalized x (C) per lane, all half.
    // That gives us O(C) memory but enables the backward to avoid recomputing
    // any reduction. Lane-aligned to NEXT_MULTIPLE_FOR_TYPE.
    const uint32_t stash_halves = this->channels + 2u;
    return ::tcnn::next_multiple<uint32_t>(stash_halves * (uint32_t)sizeof(::tcnn::network_precision_t), NEXT_MULTIPLE_FOR_TYPE);
}

std::string rfnn::tcnn::LayerNorm::generate_device_function(const std::string& fn_name) const {
    const uint32_t stash_bytes = this->device_function_fwd_ctx_bytes();

    std::string body = fmt::format(R"(
        // Welford-free, single-pass mean/var over the channel axis. C is small
        // (one warp lane handles the whole sample), so a plain sum is fine.
        const ::tcnn::network_precision_t* gamma = params;
        const ::tcnn::network_precision_t* beta  = params + {C};

        float mean = 0.f;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {C}; i++) mean += (float)input[i];
        mean /= (float){C};

        float var = 0.f;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {C}; i++) {{
            float d = (float)input[i] - mean;
            var += d * d;
        }}
        var /= (float){C};
        const float inv_std = rsqrtf(var + {EPS}f);

        ::tcnn::tvec<::tcnn::network_precision_t, {C}> out;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {C}; i++) {{
            float xn = ((float)input[i] - mean) * inv_std;
            float g  = 1.f + (float)gamma[i];
            float b  = (float)beta[i];
            out[i] = (::tcnn::network_precision_t)(xn * g + b);
        }}

        if (fwd_ctx) {{
            ::tcnn::network_precision_t* stash = (::tcnn::network_precision_t*)(fwd_ctx + lane_id() * {STASH_BYTES});
            stash[0] = (::tcnn::network_precision_t)mean;
            stash[1] = (::tcnn::network_precision_t)inv_std;
            TCNN_PRAGMA_UNROLL
            for (int i = 0; i < {C}; i++) {{
                float xn = ((float)input[i] - mean) * inv_std;
                stash[2 + i] = (::tcnn::network_precision_t)xn;
            }}
        }}

        return out;
    )",
        fmt::arg("C", this->channels),
        fmt::arg("EPS", this->eps),
        fmt::arg("STASH_BYTES", stash_bytes)
    );

    return this->generate_device_function_from_body(fn_name, body);
}

std::string rfnn::tcnn::LayerNorm::generate_backward_device_function(const std::string& fn_name, uint32_t /*n_threads*/) const {
    const uint32_t stash_bytes = this->device_function_fwd_ctx_bytes();

    // Standard LN backward (with learnable affine):
    //   dL/dxhat[i] = dL/dy[i] * (1 + gamma[i])
    //   dL/dx[i]    = (inv_std / C) * (C*dL/dxhat[i] - sum(dL/dxhat) - xhat[i] * sum(dL/dxhat * xhat))
    //   dL/dgamma[i] = dL/dy[i] * xhat[i]
    //   dL/dbeta[i]  = dL/dy[i]
    std::string body = fmt::format(R"(
        const ::tcnn::network_precision_t* stash = (const ::tcnn::network_precision_t*)(fwd_ctx + lane_id() * {STASH_BYTES});
        const float inv_std = (float)stash[1];

        const ::tcnn::network_precision_t* gamma = params;

        float dL_dxhat[{C}];
        float sum_dxhat = 0.f;
        float sum_dxhat_xhat = 0.f;

        ::tcnn::tvec<::tcnn::network_precision_t, {C}> dL_dgamma_local;
        ::tcnn::tvec<::tcnn::network_precision_t, {C}> dL_dbeta_local;

        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {C}; i++) {{
            const float g = 1.f + (float)gamma[i];
            const float xhat = (float)stash[2 + i];
            const float dy = (float)dL_dy[i];

            dL_dxhat[i] = dy * g;
            sum_dxhat += dL_dxhat[i];
            sum_dxhat_xhat += dL_dxhat[i] * xhat;

            dL_dgamma_local[i] = (::tcnn::network_precision_t)(dy * xhat);
            dL_dbeta_local[i]  = (::tcnn::network_precision_t)dy;
        }}

        if (dL_dparams) {{
            ::tcnn::atomic_add(dL_dparams,        dL_dgamma_local);
            ::tcnn::atomic_add(dL_dparams + {C},  dL_dbeta_local);
        }}

        if (dL_dx) {{
            const float inv_C = 1.f / (float){C};
            TCNN_PRAGMA_UNROLL
            for (int i = 0; i < {C}; i++) {{
                const float xhat = (float)stash[2 + i];
                const float dx = inv_std * (dL_dxhat[i] - inv_C * (sum_dxhat + xhat * sum_dxhat_xhat));
                (*dL_dx)[i] = dx;
            }}
        }}
    )",
        fmt::arg("C", this->channels),
        fmt::arg("STASH_BYTES", stash_bytes)
    );

    return this->generate_backward_device_function_from_body(fn_name, body);
}
