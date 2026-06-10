#include "radfield3d-nn/tcnn/layers/film.h"
#include "radfield3d-nn/tcnn/blocks/mlp_select.h"
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/network.h>


rfnn::tcnn::FiLM::FiLM(uint32_t feature_channels, uint32_t condition_channels, const std::string& non_linearity)
    : feature_channels(feature_channels),
      condition_channels(condition_channels),
      NEXT_MULTIPLE_FOR_TYPE((sizeof(::tcnn::network_precision_t) == 2) ? 8 : 4),
      use_relu(non_linearity == "ReLU"),
      use_silu(non_linearity == "SiLU")
{
    if (non_linearity != "ReLU" && non_linearity != "SiLU" && non_linearity != "None") {
        throw std::runtime_error("FiLM: unsupported non_linearity '" + non_linearity + "'. Use 'ReLU', 'SiLU' or 'None'.");
    }

    // CutlassMLP's MMA fragments require the input dim to be a multiple of 16,
    // so pad the condition input (which is typically narrow) to that boundary.
    this->condition_padded = ::tcnn::next_multiple<uint32_t>(condition_channels, 16u);

    // CutlassMLP tolerates small n_input_dims (the condition is typically much
    // narrower than the feature, so FullyFusedMLP's neuron-count alignment
    // requirements are unsuitable here).
    // gamma/beta predictor = a SINGLE linear projection of the condition,
    // matching the original FiLM (Perez et al. 2017) and PBRFNet's
    // `nn.Linear(condition_channels, 2*out_channels)`. n_hidden_layers=0 ⇒ one
    // weight matrix (condition_padded → 2*feature_channels), no hidden activation.
    this->mlp_gamma_beta.reset(
        ::tcnn::create_network<::tcnn::network_precision_t>(
            ::tcnn::json{
                {"otype", select_mlp_otype(this->condition_padded, 2u * feature_channels, 0u)},
                {"n_input_dims", this->condition_padded},
                {"n_neurons", feature_channels},
                {"n_output_dims", 2 * feature_channels},
                {"n_hidden_layers", 0},
                {"activation", "None"},
                {"output_activation", "None"}
            }
        )
    );

    this->set_jit_fusion(true);
    this->mlp_gamma_beta->set_jit_fusion(true);
}

std::string rfnn::tcnn::FiLM::generate_device_function(const std::string& fn_name) const {
    const uint32_t mlp_ctx_bytes = ::tcnn::next_multiple<uint32_t>(
        this->mlp_gamma_beta->device_function_fwd_ctx_bytes(), NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t stash_bytes_per_lane = ::tcnn::next_multiple<uint32_t>(
        (3u * this->feature_channels + 1u) * (uint32_t)sizeof(::tcnn::network_precision_t), NEXT_MULTIPLE_FOR_TYPE);

    std::string preamble = fmt::format(R"(
            {MLP_FUNC}

        )",
        fmt::arg("MLP_FUNC", this->mlp_gamma_beta->generate_device_function(fn_name + "_mlp_gamma_beta"))
    );

    std::string body = fmt::format(R"(
        ::tcnn::tvec<::tcnn::network_precision_t, {C_PAD}> condition_hvec;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {C_PAD}; i++) condition_hvec[i] = (::tcnn::network_precision_t)0.f;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {C}; i++) condition_hvec[i] = (::tcnn::network_precision_t)input[{F} + i];

        auto gamma_beta = {FN_NAME}_mlp_gamma_beta(condition_hvec, params, fwd_ctx ? fwd_ctx + WARP_SIZE * 0 : nullptr);

        // LayerNorm over the {F} feature channels — mirrors PBRFNet's
        // FiLM(norm="layer"). fp32 stats (fp16-safe, same as rfnn::tcnn::LayerNorm);
        // affine-less because the gamma/beta modulation below IS the per-channel
        // affine. Normalizing the feature + bounding gamma to [0,2] keeps the
        // conditioned trunk activations finite (the unbounded 1+gamma_raw and
        // missing norm let x*gamma blow up -> fp16 overflow / NaN).
        float ln_mean = 0.f;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {F}; i++) ln_mean += (float)input[i];
        ln_mean /= (float){F};
        float ln_var = 0.f;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {F}; i++) {{ float d = (float)input[i] - ln_mean; ln_var += d * d; }}
        ln_var /= (float){F};
        const float ln_inv_std = rsqrtf(ln_var + 1e-5f);

        ::tcnn::tvec<::tcnn::network_precision_t, {F}> out;
        float xn_arr[{F}];
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {F}; i++) {{
            const float xn      = ((float)input[i] - ln_mean) * ln_inv_std;
            xn_arr[i]           = xn;
            const float gamma_i = 1.f + tanhf((float)gamma_beta[i]);   // bounded [0,2]
            const float beta_i  = (float)gamma_beta[{F} + i];
            float v = xn * gamma_i + beta_i;
            {RELU}
            out[i] = (::tcnn::network_precision_t)v;
        }}

        if (fwd_ctx) {{
            ::tcnn::network_precision_t* stash = (::tcnn::network_precision_t*)(fwd_ctx + WARP_SIZE * {MLP_CTX_BYTES} + lane_id() * {STASH_BYTES});
            TCNN_PRAGMA_UNROLL
            for (int i = 0; i < {F}; i++) stash[i] = (::tcnn::network_precision_t)xn_arr[i];      // normalized feature
            TCNN_PRAGMA_UNROLL
            for (int i = 0; i < 2 * {F}; i++) stash[{F} + i] = gamma_beta[i];
            stash[3 * {F}] = (::tcnn::network_precision_t)ln_inv_std;
        }}

        return out;
    )",
        fmt::arg("FN_NAME", fn_name),
        fmt::arg("F", this->feature_channels),
        fmt::arg("C", this->condition_channels),
        fmt::arg("C_PAD", this->condition_padded),
        fmt::arg("MLP_CTX_BYTES", mlp_ctx_bytes),
        fmt::arg("STASH_BYTES", stash_bytes_per_lane),
        fmt::arg("RELU", this->use_relu
            ? "v = v > 0.f ? v : 0.f;"
            : (this->use_silu
                ? "{ float s = 1.f / (1.f + __expf(-v)); v = v * s; }"
                : ""))
    );

    return fmt::format("{}{}", preamble, this->generate_device_function_from_body(fn_name, body));
}

std::string rfnn::tcnn::FiLM::generate_backward_device_function(const std::string& fn_name, uint32_t n_threads) const {
    const uint32_t mlp_ctx_bytes = ::tcnn::next_multiple<uint32_t>(
        this->mlp_gamma_beta->device_function_fwd_ctx_bytes(), NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t stash_bytes_per_lane = ::tcnn::next_multiple<uint32_t>(
        (3u * this->feature_channels + 1u) * (uint32_t)sizeof(::tcnn::network_precision_t), NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t mlp_padded_out = this->mlp_gamma_beta->padded_output_width();

    std::string preamble = fmt::format(R"(
            {MLP_BWD_FUNC}

        )",
        fmt::arg("MLP_BWD_FUNC", this->mlp_gamma_beta->generate_backward_device_function(fn_name + "_mlp_gamma_beta_bwd", n_threads))
    );

    std::string body = fmt::format(R"(
        const ::tcnn::network_precision_t* stash = (const ::tcnn::network_precision_t*)(fwd_ctx + WARP_SIZE * {MLP_CTX_BYTES} + lane_id() * {STASH_BYTES});
        const ::tcnn::network_precision_t* xn_stash         = stash;            // normalized feature
        const ::tcnn::network_precision_t* gamma_beta_stash = stash + {F};
        const float ln_inv_std = (float)stash[3 * {F}];

        ::tcnn::tvec<::tcnn::network_precision_t, {MLP_OUT}> dL_dgamma_beta;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {MLP_OUT}; i++) dL_dgamma_beta[i] = (::tcnn::network_precision_t)0.f;

        // Pass 1: gradient through the activation, the gamma/beta modulation
        // (gamma = 1 + tanh(.)), and accumulate the two LayerNorm reduction sums.
        float dL_dxn[{F}];
        float sum_dxn = 0.f;
        float sum_dxn_xn = 0.f;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {F}; i++) {{
            const float xn_i    = (float)xn_stash[i];
            const float t_i     = tanhf((float)gamma_beta_stash[i]);
            const float gamma_i = 1.f + t_i;
            const float beta_i  = (float)gamma_beta_stash[{F} + i];
            const float v       = xn_i * gamma_i + beta_i;

            float dL_dv = (float)dL_dy[i];
            {RELU_MASK}

            dL_dgamma_beta[i]       = (::tcnn::network_precision_t)(dL_dv * xn_i * (1.f - t_i * t_i)); // d/dgamma_raw via tanh'
            dL_dgamma_beta[{F} + i] = (::tcnn::network_precision_t)dL_dv;                               // d/dbeta

            const float dxn = dL_dv * gamma_i;   // dL/d(normalized feature)
            dL_dxn[i]   = dxn;
            sum_dxn    += dxn;
            sum_dxn_xn += dxn * xn_i;
        }}

        // Pass 2: LayerNorm backward (fp32):
        // dL/dx[i] = (inv_std/F) * (F*dL/dxn[i] - sum(dL/dxn) - xn[i]*sum(dL/dxn*xn))
        if (dL_dx) {{
            TCNN_PRAGMA_UNROLL
            for (int i = 0; i < {F}; i++) {{
                const float xn_i = (float)xn_stash[i];
                (*dL_dx)[i] = (ln_inv_std / (float){F}) * ((float){F} * dL_dxn[i] - sum_dxn - xn_i * sum_dxn_xn);
            }}
        }}

        ::tcnn::tvec<::tcnn::network_precision_t, {C_PAD}> dL_dcondition;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {C_PAD}; i++) dL_dcondition[i] = (::tcnn::network_precision_t)0.f;

        {FN_NAME}_mlp_gamma_beta_bwd(
            dL_dgamma_beta,
            params,
            fwd_ctx + WARP_SIZE * 0,
            (dL_dparams) ? dL_dparams : nullptr,
            &dL_dcondition
        );

        if (dL_dx) {{
            TCNN_PRAGMA_UNROLL
            for (int i = 0; i < {C}; i++) {{
                (*dL_dx)[{F} + i] = (float)dL_dcondition[i];
            }}
        }}
    )",
        fmt::arg("FN_NAME", fn_name),
        fmt::arg("F", this->feature_channels),
        fmt::arg("C", this->condition_channels),
        fmt::arg("C_PAD", this->condition_padded),
        fmt::arg("MLP_OUT", mlp_padded_out),
        fmt::arg("MLP_CTX_BYTES", mlp_ctx_bytes),
        fmt::arg("STASH_BYTES", stash_bytes_per_lane),
        fmt::arg("RELU_MASK", this->use_relu
            ? "if (!(v > 0.f)) dL_dv = 0.f;"
            : (this->use_silu
                ? "{ float s = 1.f / (1.f + __expf(-v)); float g = s * (1.f + v * (1.f - s)); dL_dv = dL_dv * g; }"
                : ""))
    );

    return fmt::format("{}{}", preamble, this->generate_backward_device_function_from_body(fn_name, body));
}

void rfnn::tcnn::FiLM::initialize_params(::tcnn::pcg32& rnd, float* params_full_precision, float scale) {
    // Identity start (matches PBRFNet `FiLM.initialize`: gamma_beta ~N(0,1e-3)).
    // The γ/β predictor is bias-less, so a tiny weight init makes the output ≈ 0
    // → gamma = 1 + tanh(0) = 1, beta = 0 → FiLM begins as an (approximate)
    // identity. We shrink tcnn's xavier init via the `scale` argument (×0.012 →
    // xavier-std ~0.07 → ~1e-3) — small random (not 0) so per-channel γ/β still
    // break symmetry. NB: `params_full_precision` is DEVICE memory, so we MUST
    // scale through tcnn's init path, not a host loop.
    this->mlp_gamma_beta->initialize_params(rnd, params_full_precision, scale * 0.012f);
}

void rfnn::tcnn::FiLM::set_params_impl(::tcnn::network_precision_t* params, ::tcnn::network_precision_t* inference_params, ::tcnn::network_precision_t* gradients) {
    this->mlp_gamma_beta->set_params(params, inference_params, gradients);
}

size_t rfnn::tcnn::FiLM::n_params() const {
    return this->mlp_gamma_beta->n_params();
}

std::vector<std::pair<uint32_t, uint32_t>> rfnn::tcnn::FiLM::layer_sizes() const {
    return this->mlp_gamma_beta->layer_sizes();
}

uint32_t rfnn::tcnn::FiLM::device_function_fwd_ctx_bytes() const {
    const uint32_t mlp_ctx = ::tcnn::next_multiple<uint32_t>(
        this->mlp_gamma_beta->device_function_fwd_ctx_bytes(), NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t stash = ::tcnn::next_multiple<uint32_t>(
        (3u * this->feature_channels + 1u) * (uint32_t)sizeof(::tcnn::network_precision_t), NEXT_MULTIPLE_FOR_TYPE);
    return mlp_ctx + stash;
}

uint32_t rfnn::tcnn::FiLM::backward_device_function_shmem_bytes(uint32_t n_threads, ::tcnn::GradientMode param_gradients_mode) const {
    return this->mlp_gamma_beta->backward_device_function_shmem_bytes(n_threads, param_gradients_mode);
}

nlohmann::json rfnn::tcnn::FiLM::hyperparams() const {
    return {
        {"otype", "FiLM"},
        {"feature_channels", this->feature_channels},
        {"condition_channels", this->condition_channels},
        {"non_linearity", this->use_relu ? "ReLU" : "None"},
        {"mlp_gamma_beta", this->mlp_gamma_beta->hyperparams()},
    };
}

void rfnn::tcnn::FiLM::convert_params_to_jit_layout(cudaStream_t stream, bool use_inference_params) {
    this->mlp_gamma_beta->convert_params_to_jit_layout(stream, use_inference_params);
}

void rfnn::tcnn::FiLM::convert_params_from_jit_layout(cudaStream_t stream, bool use_inference_params) {
    this->mlp_gamma_beta->convert_params_from_jit_layout(stream, use_inference_params);
}
