#include "radfield3d-nn/tcnn/encodings/global_parameters.h"

rfnn::tcnn::ParameterSetEncoding::ParameterSetEncoding(const std::vector<ParameterSet>& parameter_sets, uint32_t encoded_dims)
    : encoded_dims(encoded_dims),
      NEXT_MULTIPLE_FOR_TYPE((sizeof(::tcnn::network_precision_t) == 2) ? 8 : 4)
{
    const std::string network_type = (encoded_dims > 128) ? "CutlassMLP" : "FullyFusedMLP";

    this->encoding_mlp_input_width = 0;
    for (auto& desc : parameter_sets) {
        std::unique_ptr<::tcnn::Encoding<::tcnn::network_precision_t>> encoding;

        if (desc.encoding == "spherical_harmonics") {
            assert(desc.dimensions == 3);   // SH only works on 3 dimensions in tcnn
            encoding.reset(
                ::tcnn::create_encoding<::tcnn::network_precision_t>(
                    3,
                    ::tcnn::json{
                        {"otype", "SphericalHarmonics"},
                        {"n_dims_to_encode", 3},
                        {"degree", 4}   // Fixed for SRBF and following
                    }
                )
            );
        }

        if (desc.encoding == "fourier") {
            encoding.reset(
                ::tcnn::create_encoding<::tcnn::network_precision_t>(
                    desc.dimensions,
                    ::tcnn::json{
                        {"otype", "Frequency"},
                        {"n_frequencies", desc.dimensions * 4},
                        {"n_dims_to_encode", desc.dimensions}
                    }
                )
            );
        }

        if (desc.encoding == "one_blob") {
            assert(desc.dimensions == 1);
            encoding.reset(
                ::tcnn::create_encoding<::tcnn::network_precision_t>(
                    1,
                    ::tcnn::json{
                        {"otype", "OneBlob"},
                        {"n_bins", desc.feature_dimensions}
                    }
                )
            );
        }

        if (desc.encoding == "None") {
            // Pass-through: the caller supplies already-encoded values that
            // should reach the encoding MLP as-is (just float -> half cast).
            // tcnn's "Identity" encoding implements exactly this.
            encoding.reset(
                ::tcnn::create_encoding<::tcnn::network_precision_t>(
                    desc.dimensions,
                    ::tcnn::json{
                        {"otype", "Identity"},
                        {"n_dims_to_encode", desc.dimensions}
                    }
                )
            );
        }

        if (!encoding)
            throw std::runtime_error("Unknown encoding requested: " + desc.encoding);

        ParameterSet new_desc = desc;
        new_desc.feature_dimensions = encoding->padded_output_width();

        this->parameter_sets.push_back(
            {
                new_desc,
                std::move(encoding)
            }
        );

        this->encoding_mlp_input_width += new_desc.feature_dimensions;
    }
    this->encoding_mlp_input_width = ::tcnn::next_multiple<uint32_t>(this->encoding_mlp_input_width, NEXT_MULTIPLE_FOR_TYPE);

    this->encoding_mlp.reset(
        ::tcnn::create_network<::tcnn::network_precision_t>(
            ::tcnn::json{
                {"otype", network_type},
                {"n_input_dims", this->encoding_mlp_input_width},
                {"n_neurons", encoded_dims},
                {"n_output_dims", encoded_dims},
                {"n_hidden_layers", 2},
                // SiLU mirrors PBRFNet's RFBackboneModel (nn.SiLU throughout).
                // ReLU has dead-neuron risk in fp16: once a unit's pre-act goes
                // negative the gradient is zero and the unit dies permanently.
                {"activation", "SiLU"},
                {"output_activation", "None"}
            }
        )
    );

    this->set_jit_fusion(true);
}

std::vector<std::pair<uint32_t, uint32_t>> rfnn::tcnn::ParameterSetEncoding::layer_sizes() const {
    return this->encoding_mlp->layer_sizes();
}

std::string rfnn::tcnn::ParameterSetEncoding::generate_device_function(const std::string& fn_name) const {
    std::string preamble = fmt::format(R"(
            {ENCODING_MLP_FUNC}

        )",
            fmt::arg("ENCODING_MLP_FUNC", this->encoding_mlp->generate_device_function(fn_name + "_mlp"))
        );

    for (size_t encoding_idx = 0; encoding_idx < this->parameter_sets.size(); encoding_idx++) {
        preamble += this->parameter_sets[encoding_idx].second->generate_device_function(fn_name + "_encoding_" + std::to_string(encoding_idx));
        preamble += "\n\n";
    }

    uint32_t input_offset = 0;
    uint32_t params_offset = 0;
    uint32_t ctx_offset = 0;
    std::string concat_expr;

    if (this->parameter_sets.size() == 1) {
        const ParameterSet& ps = this->parameter_sets[0].first;
        auto enc = this->parameter_sets[0].second.get();
        concat_expr = fmt::format(
            "{FN_NAME}_encoding_0(::tcnn::tvec<float, {IN_DIM}>{{input[0], input[1], input[2]}}, params + 0, fwd_ctx ? fwd_ctx : nullptr)",
            fmt::arg("FN_NAME", fn_name),
            fmt::arg("IN_DIM", ps.dimensions)
        );
        params_offset = ::tcnn::next_multiple<uint32_t>(enc->n_params(), NEXT_MULTIPLE_FOR_TYPE);
        ctx_offset    = ::tcnn::next_multiple<uint32_t>(enc->device_function_fwd_ctx_bytes(), NEXT_MULTIPLE_FOR_TYPE);
    } else {
        std::string lambda_body;
        size_t concat_off = 0;
        for (size_t e = 0; e < this->parameter_sets.size(); e++) {
            const ParameterSet& ps = this->parameter_sets[e].first;
            auto enc = this->parameter_sets[e].second.get();
            lambda_body += fmt::format(R"(
                {{
                    ::tcnn::tvec<float, {IN_DIM}> in;
                    TCNN_PRAGMA_UNROLL
                    for (int i = 0; i < {IN_DIM}; ++i) in[i] = input[{IN_OFF} + i];
                    auto enc_out = {FN_NAME}_encoding_{ENC_IDX}(in, params + {P_OFF}, fwd_ctx ? fwd_ctx + WARP_SIZE * {C_OFF} : nullptr);
                    TCNN_PRAGMA_UNROLL
                    for (int i = 0; i < {D}; ++i) out[{CC_OFF} + i] = enc_out[i];
                }}
            )",
                fmt::arg("IN_DIM", ps.dimensions),
                fmt::arg("IN_OFF", input_offset),
                fmt::arg("FN_NAME", fn_name),
                fmt::arg("ENC_IDX", e),
                fmt::arg("P_OFF", params_offset),
                fmt::arg("C_OFF", ctx_offset),
                fmt::arg("D", ps.feature_dimensions),
                fmt::arg("CC_OFF", concat_off)
            );
            input_offset  += ps.dimensions;
            params_offset += ::tcnn::next_multiple<uint32_t>(enc->n_params(), NEXT_MULTIPLE_FOR_TYPE);
            ctx_offset    += ::tcnn::next_multiple<uint32_t>(enc->device_function_fwd_ctx_bytes(), NEXT_MULTIPLE_FOR_TYPE);
            concat_off    += ps.feature_dimensions;
        }
        concat_expr = fmt::format(R"(([&]() -> ::tcnn::tvec<::tcnn::network_precision_t, {W}> {{
            ::tcnn::tvec<::tcnn::network_precision_t, {W}> out;
            TCNN_PRAGMA_UNROLL
            for (int i = 0; i < {W}; ++i) out[i] = (::tcnn::network_precision_t)0;
            {LAMBDA_BODY}
            return out;
        }}()))",
            fmt::arg("W", this->encoding_mlp_input_width),
            fmt::arg("LAMBDA_BODY", lambda_body)
        );
    }

    std::string body = fmt::format(R"(
        return {FN_NAME}_mlp({ENCODED}, params + {P_OFF}, fwd_ctx ? fwd_ctx + WARP_SIZE * {C_OFF} : nullptr);
    )",
        fmt::arg("FN_NAME", fn_name),
        fmt::arg("ENCODED", concat_expr),
        fmt::arg("P_OFF", params_offset),
        fmt::arg("C_OFF", ctx_offset)
    );

    return fmt::format("{}{}", preamble, this->generate_device_function_from_body(fn_name, body));
}


std::string rfnn::tcnn::ParameterSetEncoding::generate_backward_device_function(const std::string& fn_name, uint32_t n_threads) const {
    std::string preamble = fmt::format(R"(
            {ENCODING_MLP_FUNC}

        )",
            fmt::arg("ENCODING_MLP_FUNC", this->encoding_mlp->generate_backward_device_function(fn_name + "_mlp_bwd", n_threads))
        );

    for (size_t encoding_idx = 0; encoding_idx < this->parameter_sets.size(); encoding_idx++) {
        preamble += this->parameter_sets[encoding_idx].second->generate_backward_device_function(fn_name + "_encoding_bwd_" + std::to_string(encoding_idx), n_threads);
        preamble += "\n\n";
    }

    uint32_t enc_mlp_offset = this->encoding_mlp_input_width;
    uint32_t params_offset = this->n_params() - ::tcnn::next_multiple<uint32_t>(this->encoding_mlp->n_params(), NEXT_MULTIPLE_FOR_TYPE);
    uint32_t ctx_offset = this->device_function_fwd_ctx_bytes() - ::tcnn::next_multiple<uint32_t>(this->encoding_mlp->device_function_fwd_ctx_bytes(), NEXT_MULTIPLE_FOR_TYPE);
    uint32_t input_offset = this->input_width();

    std::string body = fmt::format(R"(
        if (dL_dx) {{
            TCNN_PRAGMA_UNROLL
            for (int i = 0; i < {D_IN_TOTAL}; i++) (*dL_dx)[i] = 0.f;
        }}
        ::tcnn::tvec<::tcnn::network_precision_t, {D_MLP}> dL_dEncoding_mlp;
        {FN_NAME}_mlp_bwd(dL_dy, params + {PARAMS_OFFSET}, fwd_ctx + WARP_SIZE * {ENC_FWD_CTX_OFFSET}, (dL_dparams) ? dL_dparams + {PARAMS_OFFSET} : nullptr, &dL_dEncoding_mlp);
    )",
        fmt::arg("D_MLP", this->encoding_mlp_input_width),
        fmt::arg("D_IN_TOTAL", this->input_width()),
        fmt::arg("FN_NAME", fn_name),
        fmt::arg("PARAMS_OFFSET", params_offset),
        fmt::arg("ENC_FWD_CTX_OFFSET", ctx_offset)
    );

    for (long encoding_idx = this->parameter_sets.size() - 1; encoding_idx >= 0; encoding_idx--) {
        const ParameterSet& params = this->parameter_sets[encoding_idx].first;
        auto encoding = this->parameter_sets[encoding_idx].second.get();

        enc_mlp_offset -= params.feature_dimensions;
        params_offset -= ::tcnn::next_multiple<uint32_t>(encoding->n_params(), NEXT_MULTIPLE_FOR_TYPE);
        ctx_offset -= ::tcnn::next_multiple<uint32_t>(encoding->device_function_fwd_ctx_bytes(), NEXT_MULTIPLE_FOR_TYPE);
        input_offset -= params.dimensions;

        body += fmt::format(R"(
            ::tcnn::tvec<float, {D_IN}> dL_dEncodingInput_{ENC_IDX} = {{0.f}};
            ::tcnn::tvec<::tcnn::network_precision_t, {D_ENC}> dMLP_dEncoding_{ENC_IDX} = {{0}};

            TCNN_PRAGMA_UNROLL
            for (int i = 0; i < {D_ENC}; i++) {{
                dMLP_dEncoding_{ENC_IDX}[i] = dL_dEncoding_mlp[{ENC_MLP_OFFSET} + i];
            }}

            {FN_NAME}_encoding_bwd_{ENC_IDX}(dMLP_dEncoding_{ENC_IDX}, params + {PARAMS_OFFSET}, fwd_ctx + WARP_SIZE * {ENC_FWD_CTX_OFFSET}, (dL_dparams) ? dL_dparams + {PARAMS_OFFSET} : nullptr, &dL_dEncodingInput_{ENC_IDX});

            if (dL_dx) {{
                TCNN_PRAGMA_UNROLL
                for (int i = 0; i < {D_IN}; i++) {{
                    (*dL_dx)[{INPUT_OFFSET} + i] += dL_dEncodingInput_{ENC_IDX}[i];
                }}
            }}
        )",
            fmt::arg("D_IN", params.dimensions),
            fmt::arg("INPUT_OFFSET", input_offset),
            fmt::arg("D_ENC", encoding->padded_output_width()),
            fmt::arg("ENC_MLP_OFFSET", enc_mlp_offset),
            fmt::arg("ENC_IDX", encoding_idx),
            fmt::arg("FN_NAME", fn_name),
            fmt::arg("PARAMS_OFFSET", params_offset),
            fmt::arg("ENC_FWD_CTX_OFFSET", ctx_offset)
        );
    }

    std::string dev_function = fmt::format("{}{}", preamble, this->generate_backward_device_function_from_body(fn_name, body));
    return dev_function;
}

void rfnn::tcnn::ParameterSetEncoding::initialize_params(::tcnn::pcg32& rnd, float* params_full_precision, float scale) {
    float* current_params = params_full_precision;
    for (auto& params : this->parameter_sets) {
        params.second->initialize_params(rnd, current_params, scale);
        current_params += ::tcnn::next_multiple<uint32_t>(params.second->n_params(), NEXT_MULTIPLE_FOR_TYPE);
    }
    this->encoding_mlp->initialize_params(rnd, current_params, scale);
}

void rfnn::tcnn::ParameterSetEncoding::set_params_impl(::tcnn::network_precision_t* params, ::tcnn::network_precision_t* inference_params, ::tcnn::network_precision_t* gradients) {
    uint32_t offset = 0;
    for (auto& p_set : this->parameter_sets) {
        p_set.second->set_params(params + offset, inference_params + offset, gradients + offset);
        offset += ::tcnn::next_multiple<uint32_t>(p_set.second->n_params(), NEXT_MULTIPLE_FOR_TYPE);
    }
    this->encoding_mlp->set_params(params + offset, inference_params + offset, gradients + offset);
}

uint32_t rfnn::tcnn::ParameterSetEncoding::device_function_fwd_ctx_bytes() const {
    uint32_t bytes = 0;
    for (auto& p_set : this->parameter_sets)
        bytes += ::tcnn::next_multiple<uint32_t>(p_set.second->device_function_fwd_ctx_bytes(),  NEXT_MULTIPLE_FOR_TYPE);
    bytes += ::tcnn::next_multiple<uint32_t>(this->encoding_mlp->device_function_fwd_ctx_bytes(), NEXT_MULTIPLE_FOR_TYPE);
    return bytes;
}

size_t rfnn::tcnn::ParameterSetEncoding::n_params() const {
    size_t bytes = 0;
    for (auto& p_set : this->parameter_sets)
        bytes += ::tcnn::next_multiple<size_t>(p_set.second->n_params(),  NEXT_MULTIPLE_FOR_TYPE);
    bytes += ::tcnn::next_multiple<size_t>(this->encoding_mlp->n_params(), NEXT_MULTIPLE_FOR_TYPE);
    return bytes;
}

uint32_t rfnn::tcnn::ParameterSetEncoding::input_width() const {
    uint32_t n = 0;
    for (auto& p_set : this->parameter_sets)
        n += p_set.first.dimensions;
    return n;
}

uint32_t rfnn::tcnn::ParameterSetEncoding::backward_device_function_shmem_bytes(uint32_t n_threads, ::tcnn::GradientMode param_gradients_mode) const {
    uint32_t bytes = 0;
    for (auto& p_set : this->parameter_sets) {
        //bytes += p_set.second->backward_device_function_shmem_bytes(n_threads, param_gradients_mode);
        bytes = std::max(bytes, p_set.second->backward_device_function_shmem_bytes(n_threads, param_gradients_mode));
    }
    //bytes += this->encoding_mlp->backward_device_function_shmem_bytes(n_threads, param_gradients_mode);
    bytes = std::max(bytes, this->encoding_mlp->backward_device_function_shmem_bytes(n_threads, param_gradients_mode));
    return bytes;
}

nlohmann::json rfnn::tcnn::ParameterSetEncoding::hyperparams() const {
    return {
        {"otype", "ParameterSet"},
        {"encoding_mlp", this->encoding_mlp->hyperparams()},
    };  // TODO: Implement serialization
}

uint32_t rfnn::tcnn::ParameterSetEncoding::padded_output_width() const {
    return this->encoding_mlp->padded_output_width();
}

void rfnn::tcnn::ParameterSetEncoding::convert_params_to_jit_layout(cudaStream_t stream, bool use_inference_params) {
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    auto p = this->encoding_mlp->params();
    this->encoding_mlp->convert_params_to_jit_layout(stream, use_inference_params);
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    for (auto& p_set : this->parameter_sets)
        p_set.second->convert_params_to_jit_layout(stream, use_inference_params);
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
}

void rfnn::tcnn::ParameterSetEncoding::convert_params_from_jit_layout(cudaStream_t stream, bool use_inference_params) {
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    this->encoding_mlp->convert_params_from_jit_layout(stream, use_inference_params);
    for (auto& p_set : this->parameter_sets)
        p_set.second->convert_params_from_jit_layout(stream, use_inference_params);
}
