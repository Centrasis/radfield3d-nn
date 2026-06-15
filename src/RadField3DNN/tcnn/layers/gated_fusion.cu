#include "radfield3d-nn/tcnn/layers/gated_fusion.h"
#include "radfield3d-nn/tcnn/blocks/mlp_select.h"
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/network.h>


rfnn::tcnn::GatedFusion::GatedFusion(uint32_t feature_channels, uint32_t condition_channels, const std::string& non_linearity)
    : feature_channels(feature_channels),
      condition_channels(condition_channels),
      NEXT_MULTIPLE_FOR_TYPE((sizeof(::tcnn::network_precision_t) == 2) ? 8 : 4),
      use_relu(non_linearity == "ReLU"),
      use_silu(non_linearity == "SiLU")
{
    if (non_linearity != "ReLU" && non_linearity != "SiLU" && non_linearity != "None") {
        throw std::runtime_error("GatedFusion: unsupported non_linearity '" + non_linearity + "'. Use 'ReLU', 'SiLU' or 'None'.");
    }

    // CutlassMLP's MMA fragments require the input dim to be a multiple of 16,
    // so pad the (typically narrow) condition input to that boundary. Mirrors FiLM.
    this->condition_padded = ::tcnn::next_multiple<uint32_t>(condition_channels, 16u);

    // condition -> (gate_logits[F], candidate[F]). Same shape/param-layout as
    // FiLM's gamma/beta predictor, so GatedFusion is a drop-in conditioner.
    this->mlp_gate_candidate.reset(
        ::tcnn::create_network<::tcnn::network_precision_t>(
            ::tcnn::json{
                // FullyFusedMLP when in/out <= 128 (and a hidden layer); else
                // CutlassMLP. The 2*feature_channels output means d_model >= 64
                // falls back to CutlassMLP, matching FiLM's choice.
                {"otype", select_mlp_otype(this->condition_padded, 2u * feature_channels, 1u)},
                {"n_input_dims", this->condition_padded},
                {"n_neurons", feature_channels},
                {"n_output_dims", 2 * feature_channels},
                {"n_hidden_layers", 1},
                {"activation", non_linearity},
                {"output_activation", "None"}
            }
        )
    );

    this->set_jit_fusion(true);
    this->mlp_gate_candidate->set_jit_fusion(true);
}

std::string rfnn::tcnn::GatedFusion::generate_device_function(const std::string& fn_name) const {
    const uint32_t mlp_ctx_bytes = ::tcnn::next_multiple<uint32_t>(
        this->mlp_gate_candidate->device_function_fwd_ctx_bytes(), NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t stash_bytes_per_lane = ::tcnn::next_multiple<uint32_t>(
        3u * this->feature_channels * (uint32_t)sizeof(::tcnn::network_precision_t), NEXT_MULTIPLE_FOR_TYPE);

    std::string preamble = fmt::format(R"(
            {MLP_FUNC}

        )",
        fmt::arg("MLP_FUNC", this->mlp_gate_candidate->generate_device_function(fn_name + "_mlp_gate_candidate"))
    );

    std::string body = fmt::format(R"(
        ::tcnn::tvec<::tcnn::network_precision_t, {C_PAD}> condition_hvec;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {C_PAD}; i++) condition_hvec[i] = (::tcnn::network_precision_t)0.f;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {C}; i++) condition_hvec[i] = (::tcnn::network_precision_t)input[{F} + i];

        auto gate_cand = {FN_NAME}_mlp_gate_candidate(condition_hvec, params, fwd_ctx ? fwd_ctx + WARP_SIZE * 0 : nullptr);

        ::tcnn::tvec<::tcnn::network_precision_t, {F}> out;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {F}; i++) {{
            const ::tcnn::network_precision_t feature_i   = (::tcnn::network_precision_t)input[i];
            const ::tcnn::network_precision_t candidate_i = gate_cand[{F} + i];
            // hardsigmoid in float: clamp(x/6 + 0.5, 0, 1) in [0, 1]
            const float gl = (float)gate_cand[i];
            const float g  = fminf(fmaxf(gl * (1.f / 6.f) + 0.5f, 0.f), 1.f);
            const ::tcnn::network_precision_t g_h = (::tcnn::network_precision_t)g;
            ::tcnn::network_precision_t v = g_h * feature_i + ((::tcnn::network_precision_t)1.f - g_h) * candidate_i;
            {RELU}
            out[i] = v;
        }}

        if (fwd_ctx) {{
            ::tcnn::network_precision_t* stash = (::tcnn::network_precision_t*)(fwd_ctx + WARP_SIZE * {MLP_CTX_BYTES} + lane_id() * {STASH_BYTES});
            TCNN_PRAGMA_UNROLL
            for (int i = 0; i < {F}; i++) stash[i] = (::tcnn::network_precision_t)input[i];
            TCNN_PRAGMA_UNROLL
            for (int i = 0; i < 2 * {F}; i++) stash[{F} + i] = gate_cand[i];
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
            ? "v = v > (::tcnn::network_precision_t)0.f ? v : (::tcnn::network_precision_t)0.f;"
            : (this->use_silu
                ? "{ float fv = (float)v; float s = 1.f / (1.f + __expf(-fv)); v = (::tcnn::network_precision_t)(fv * s); }"
                : ""))
    );

    return fmt::format("{}{}", preamble, this->generate_device_function_from_body(fn_name, body));
}

std::string rfnn::tcnn::GatedFusion::generate_backward_device_function(const std::string& fn_name, uint32_t n_threads) const {
    const uint32_t mlp_ctx_bytes = ::tcnn::next_multiple<uint32_t>(
        this->mlp_gate_candidate->device_function_fwd_ctx_bytes(), NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t stash_bytes_per_lane = ::tcnn::next_multiple<uint32_t>(
        3u * this->feature_channels * (uint32_t)sizeof(::tcnn::network_precision_t), NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t mlp_padded_out = this->mlp_gate_candidate->padded_output_width();

    std::string preamble = fmt::format(R"(
            {MLP_BWD_FUNC}

        )",
        fmt::arg("MLP_BWD_FUNC", this->mlp_gate_candidate->generate_backward_device_function(fn_name + "_mlp_gate_candidate_bwd", n_threads))
    );

    std::string body = fmt::format(R"(
        const ::tcnn::network_precision_t* stash = (const ::tcnn::network_precision_t*)(fwd_ctx + WARP_SIZE * {MLP_CTX_BYTES} + lane_id() * {STASH_BYTES});
        const ::tcnn::network_precision_t* feature_stash   = stash;
        const ::tcnn::network_precision_t* gate_cand_stash = stash + {F};

        ::tcnn::tvec<::tcnn::network_precision_t, {MLP_OUT}> dL_dgate_cand;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {MLP_OUT}; i++) dL_dgate_cand[i] = (::tcnn::network_precision_t)0.f;

        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {F}; i++) {{
            const ::tcnn::network_precision_t feature_i   = feature_stash[i];
            const ::tcnn::network_precision_t candidate_i = gate_cand_stash[{F} + i];
            const float gl = (float)gate_cand_stash[i];
            const float g  = fminf(fmaxf(gl * (1.f / 6.f) + 0.5f, 0.f), 1.f);
            const ::tcnn::network_precision_t g_h = (::tcnn::network_precision_t)g;
            const ::tcnn::network_precision_t pre = g_h * feature_i + ((::tcnn::network_precision_t)1.f - g_h) * candidate_i;

            ::tcnn::network_precision_t dL_dpre = dL_dy[i];
            {RELU_MASK}

            // out = g*(feature - candidate) + candidate
            // d out/d gate_logit = (feature - candidate) * hardsigmoid'(gate_logit)
            const float hsg = (fabsf(gl) < 3.f) ? (1.f / 6.f) : 0.f;
            dL_dgate_cand[i]        = (::tcnn::network_precision_t)((float)dL_dpre * (float)(feature_i - candidate_i) * hsg);
            dL_dgate_cand[{F} + i]  = dL_dpre * ((::tcnn::network_precision_t)1.f - g_h);   // d out/d candidate = 1 - g

            if (dL_dx) {{
                (*dL_dx)[i] = (float)(dL_dpre * g_h);   // d out/d feature = g
            }}
        }}

        ::tcnn::tvec<::tcnn::network_precision_t, {C_PAD}> dL_dcondition;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {C_PAD}; i++) dL_dcondition[i] = (::tcnn::network_precision_t)0.f;

        {FN_NAME}_mlp_gate_candidate_bwd(
            dL_dgate_cand,
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
            ? "if (!(pre > (::tcnn::network_precision_t)0.f)) dL_dpre = (::tcnn::network_precision_t)0.f;"
            : (this->use_silu
                ? "{ float fp = (float)pre; float s = 1.f / (1.f + __expf(-fp)); float gd = s * (1.f + fp * (1.f - s)); dL_dpre = (::tcnn::network_precision_t)((float)dL_dpre * gd); }"
                : ""))
    );

    return fmt::format("{}{}", preamble, this->generate_backward_device_function_from_body(fn_name, body));
}

void rfnn::tcnn::GatedFusion::initialize_params(::tcnn::pcg32& rnd, float* params_full_precision, float scale) {
    this->mlp_gate_candidate->initialize_params(rnd, params_full_precision, scale);
}

void rfnn::tcnn::GatedFusion::set_params_impl(::tcnn::network_precision_t* params, ::tcnn::network_precision_t* inference_params, ::tcnn::network_precision_t* gradients) {
    this->mlp_gate_candidate->set_params(params, inference_params, gradients);
}

size_t rfnn::tcnn::GatedFusion::n_params() const {
    return this->mlp_gate_candidate->n_params();
}

std::vector<std::pair<uint32_t, uint32_t>> rfnn::tcnn::GatedFusion::layer_sizes() const {
    return this->mlp_gate_candidate->layer_sizes();
}

uint32_t rfnn::tcnn::GatedFusion::device_function_fwd_ctx_bytes() const {
    const uint32_t mlp_ctx = ::tcnn::next_multiple<uint32_t>(
        this->mlp_gate_candidate->device_function_fwd_ctx_bytes(), NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t stash = ::tcnn::next_multiple<uint32_t>(
        3u * this->feature_channels * (uint32_t)sizeof(::tcnn::network_precision_t), NEXT_MULTIPLE_FOR_TYPE);
    return mlp_ctx + stash;
}

uint32_t rfnn::tcnn::GatedFusion::backward_device_function_shmem_bytes(uint32_t n_threads, ::tcnn::GradientMode param_gradients_mode) const {
    return this->mlp_gate_candidate->backward_device_function_shmem_bytes(n_threads, param_gradients_mode);
}

nlohmann::json rfnn::tcnn::GatedFusion::hyperparams() const {
    return {
        {"otype", "GatedFusion"},
        {"feature_channels", this->feature_channels},
        {"condition_channels", this->condition_channels},
        {"non_linearity", this->use_relu ? "ReLU" : (this->use_silu ? "SiLU" : "None")},
        {"mlp_gate_candidate", this->mlp_gate_candidate->hyperparams()},
    };
}

void rfnn::tcnn::GatedFusion::convert_params_to_jit_layout(cudaStream_t stream, bool use_inference_params) {
    this->mlp_gate_candidate->convert_params_to_jit_layout(stream, use_inference_params);
}

void rfnn::tcnn::GatedFusion::convert_params_from_jit_layout(cudaStream_t stream, bool use_inference_params) {
    this->mlp_gate_candidate->convert_params_from_jit_layout(stream, use_inference_params);
}
