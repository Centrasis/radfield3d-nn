#include "radfield3d-nn/tcnn/encodings/sperf_beam_encoder.h"

// -----------------------------------------------------------------------------
// SPERFBeamEncoder (JIT-fused, distance-less variant of PBRFBeamEncoder)
// -----------------------------------------------------------------------------

using namespace rfnn::tcnn;


SPERFBeamEncoder::SPERFBeamEncoder(uint32_t spectrum_dim, uint32_t d_model)
    : spectrum_dim(spectrum_dim),
      d_model(d_model),
      NEXT_MULTIPLE_FOR_TYPE((sizeof(::tcnn::network_precision_t) == 2) ? 8 : 4)
{
    // 2-layer spectrum MLP at internal width = d_model, mirroring PBRFBeamEncoder.
    this->spectrum_enc_dim = d_model;
    this->spectrum_dim_padded = ::tcnn::next_multiple<uint32_t>(spectrum_dim, 16u);

    // No LayerNorm on the raw spectrum input — see PBRFBeamEncoder comment.
    this->spectrum_mlp1.reset(
        ::tcnn::create_network<::tcnn::network_precision_t>(
            ::tcnn::json{
                {"otype", "CutlassMLP"},
                {"n_input_dims", this->spectrum_dim_padded},
                {"n_neurons", d_model},
                {"n_output_dims", d_model},
                {"n_hidden_layers", 1},
                // SiLU mirrors PBRFNet (see beam_encoder.cu for rationale).
                {"activation", "SiLU"},
                {"output_activation", "None"}
            }
        )
    );

    this->spectrum_ln2.reset(new rfnn::tcnn::LayerNorm(d_model));
    this->spectrum_mlp2.reset(
        ::tcnn::create_network<::tcnn::network_precision_t>(
            ::tcnn::json{
                {"otype", "CutlassMLP"},
                {"n_input_dims", d_model},
                {"n_neurons", d_model},
                {"n_output_dims", spectrum_enc_dim},
                {"n_hidden_layers", 0},
                {"activation", "None"},
                {"output_activation", "None"}
            }
        )
    );

    // No distance one-blob: only SH(direction) + None(spectrum_encoded).
    std::vector<ParameterSetEncoding::ParameterSet> sets = {
        ParameterSetEncoding::ParameterSet("spherical_harmonics", 3),
        ParameterSetEncoding::ParameterSet("None",                spectrum_enc_dim, spectrum_enc_dim),
    };
    this->beam_params_encoding.reset(new ParameterSetEncoding(sets, d_model));

    this->beam_params_ln.reset(new rfnn::tcnn::LayerNorm(d_model));

    this->spectrum_mlp1->set_jit_fusion(true);
    this->spectrum_mlp2->set_jit_fusion(true);
    this->beam_params_encoding->set_jit_fusion(true);
    this->set_jit_fusion(true);
}

SPERFBeamEncoder::~SPERFBeamEncoder() = default;

size_t SPERFBeamEncoder::n_params() const {
    size_t n = 0;
    n += ::tcnn::next_multiple<size_t>(this->spectrum_mlp1->n_params(),        NEXT_MULTIPLE_FOR_TYPE);
    n += ::tcnn::next_multiple<size_t>(this->spectrum_ln2->n_params(),         NEXT_MULTIPLE_FOR_TYPE);
    n += ::tcnn::next_multiple<size_t>(this->spectrum_mlp2->n_params(),        NEXT_MULTIPLE_FOR_TYPE);
    n += ::tcnn::next_multiple<size_t>(this->beam_params_encoding->n_params(), NEXT_MULTIPLE_FOR_TYPE);
    n += ::tcnn::next_multiple<size_t>(this->beam_params_ln->n_params(),       NEXT_MULTIPLE_FOR_TYPE);
    return n;
}

std::vector<std::pair<uint32_t, uint32_t>> SPERFBeamEncoder::layer_sizes() const {
    std::vector<std::pair<uint32_t, uint32_t>> out;
    for (auto& l : this->spectrum_mlp1->layer_sizes())           out.push_back(l);
    for (auto& l : this->spectrum_ln2->layer_sizes())            out.push_back(l);
    for (auto& l : this->spectrum_mlp2->layer_sizes())           out.push_back(l);
    for (auto& l : this->beam_params_encoding->layer_sizes())    out.push_back(l);
    for (auto& l : this->beam_params_ln->layer_sizes())          out.push_back(l);
    return out;
}

void SPERFBeamEncoder::set_params_impl(::tcnn::network_precision_t* params, ::tcnn::network_precision_t* inference_params, ::tcnn::network_precision_t* gradients) {
    size_t offset = 0;
    auto bump = [&](size_t n) {
        offset += ::tcnn::next_multiple<size_t>(n, NEXT_MULTIPLE_FOR_TYPE);
    };

    this->spectrum_mlp1->set_params(params + offset, inference_params + offset, gradients + offset);
    bump(this->spectrum_mlp1->n_params());

    this->spectrum_ln2->set_params(params + offset, inference_params + offset, gradients + offset);
    bump(this->spectrum_ln2->n_params());

    this->spectrum_mlp2->set_params(params + offset, inference_params + offset, gradients + offset);
    bump(this->spectrum_mlp2->n_params());

    this->beam_params_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
    bump(this->beam_params_encoding->n_params());

    this->beam_params_ln->set_params(params + offset, inference_params + offset, gradients + offset);
    bump(this->beam_params_ln->n_params());
}

void SPERFBeamEncoder::initialize_params(::tcnn::pcg32& rnd, float* p, float scale) {
    float* cur = p;
    auto step = [&](auto& m) {
        m->initialize_params(rnd, cur, scale);
        cur += ::tcnn::next_multiple<size_t>(m->n_params(), NEXT_MULTIPLE_FOR_TYPE);
    };
    step(this->spectrum_mlp1);
    step(this->spectrum_ln2);
    step(this->spectrum_mlp2);
    step(this->beam_params_encoding);
    step(this->beam_params_ln);
}

uint32_t SPERFBeamEncoder::device_function_fwd_ctx_bytes() const {
    uint32_t bytes = 0;
    bytes += ::tcnn::next_multiple<uint32_t>(this->spectrum_mlp1->device_function_fwd_ctx_bytes(),        NEXT_MULTIPLE_FOR_TYPE);
    bytes += ::tcnn::next_multiple<uint32_t>(this->spectrum_ln2->device_function_fwd_ctx_bytes(),         NEXT_MULTIPLE_FOR_TYPE);
    bytes += ::tcnn::next_multiple<uint32_t>(this->spectrum_mlp2->device_function_fwd_ctx_bytes(),        NEXT_MULTIPLE_FOR_TYPE);
    bytes += ::tcnn::next_multiple<uint32_t>(this->beam_params_encoding->device_function_fwd_ctx_bytes(), NEXT_MULTIPLE_FOR_TYPE);
    bytes += ::tcnn::next_multiple<uint32_t>(this->beam_params_ln->device_function_fwd_ctx_bytes(),       NEXT_MULTIPLE_FOR_TYPE);
    return bytes;
}

uint32_t SPERFBeamEncoder::backward_device_function_shmem_bytes(uint32_t n_threads, ::tcnn::GradientMode mode) const {
    uint32_t bytes = 0;
    bytes = std::max(bytes, this->spectrum_mlp1->backward_device_function_shmem_bytes(n_threads, mode));
    bytes = std::max(bytes, this->spectrum_ln2->backward_device_function_shmem_bytes(n_threads, mode));
    bytes = std::max(bytes, this->spectrum_mlp2->backward_device_function_shmem_bytes(n_threads, mode));
    bytes = std::max(bytes, this->beam_params_encoding->backward_device_function_shmem_bytes(n_threads, mode));
    bytes = std::max(bytes, this->beam_params_ln->backward_device_function_shmem_bytes(n_threads, mode));
    return bytes;
}

void SPERFBeamEncoder::convert_params_to_jit_layout(cudaStream_t stream, bool use_inference_params) {
    this->spectrum_mlp1->convert_params_to_jit_layout(stream, use_inference_params);
    this->spectrum_mlp2->convert_params_to_jit_layout(stream, use_inference_params);
    this->beam_params_encoding->convert_params_to_jit_layout(stream, use_inference_params);
}

void SPERFBeamEncoder::convert_params_from_jit_layout(cudaStream_t stream, bool use_inference_params) {
    this->spectrum_mlp1->convert_params_from_jit_layout(stream, use_inference_params);
    this->spectrum_mlp2->convert_params_from_jit_layout(stream, use_inference_params);
    this->beam_params_encoding->convert_params_from_jit_layout(stream, use_inference_params);
}

std::string SPERFBeamEncoder::generate_device_function(const std::string& fn_name) const {
    const uint32_t MLP1_P   = 0;
    const uint32_t LN2_P    = MLP1_P   + ::tcnn::next_multiple<uint32_t>(this->spectrum_mlp1->n_params(),        NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t MLP2_P   = LN2_P    + ::tcnn::next_multiple<uint32_t>(this->spectrum_ln2->n_params(),         NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t PSE_P    = MLP2_P   + ::tcnn::next_multiple<uint32_t>(this->spectrum_mlp2->n_params(),        NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t OUT_LN_P = PSE_P    + ::tcnn::next_multiple<uint32_t>(this->beam_params_encoding->n_params(), NEXT_MULTIPLE_FOR_TYPE);

    const uint32_t MLP1_O   = 0;
    const uint32_t LN2_O    = MLP1_O   + ::tcnn::next_multiple<uint32_t>(this->spectrum_mlp1->device_function_fwd_ctx_bytes(),        NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t MLP2_O   = LN2_O    + ::tcnn::next_multiple<uint32_t>(this->spectrum_ln2->device_function_fwd_ctx_bytes(),         NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t PSE_O    = MLP2_O   + ::tcnn::next_multiple<uint32_t>(this->spectrum_mlp2->device_function_fwd_ctx_bytes(),        NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t OUT_LN_O = PSE_O    + ::tcnn::next_multiple<uint32_t>(this->beam_params_encoding->device_function_fwd_ctx_bytes(), NEXT_MULTIPLE_FOR_TYPE);

    std::string preamble = fmt::format(R"(
            {MLP1_FUNC}

            {LN2_FUNC}

            {MLP2_FUNC}

            {PSE_FUNC}

            {OUT_LN_FUNC}

        )",
        fmt::arg("MLP1_FUNC",   this->spectrum_mlp1->generate_device_function(fn_name + "_mlp1")),
        fmt::arg("LN2_FUNC",    this->spectrum_ln2->generate_device_function(fn_name + "_ln2")),
        fmt::arg("MLP2_FUNC",   this->spectrum_mlp2->generate_device_function(fn_name + "_mlp2")),
        fmt::arg("PSE_FUNC",    this->beam_params_encoding->generate_device_function(fn_name + "_pse")),
        fmt::arg("OUT_LN_FUNC", this->beam_params_ln->generate_device_function(fn_name + "_out_ln"))
    );

    // Input layout (float, total = 3 + spectrum_dim):
    //   input[0..3)            direction (3)
    //   input[3..3+spectrum_dim) spectrum
    // (No distance — that's what distinguishes SPERFBeamEncoder from PBRFBeamEncoder.)
    std::string body = fmt::format(R"(
        // 1) Extract and zero-pad the spectrum slice into a half-precision
        //    tvec — `spectrum_mlp1` is a Network<half,half> so the JIT
        //    signature expects network_precision_t input.
        ::tcnn::tvec<::tcnn::network_precision_t, {SPEC_PAD}> spec_in;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {SPEC_PAD}; i++) spec_in[i] = (::tcnn::network_precision_t)0.f;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {SPEC}; i++) spec_in[i] = (::tcnn::network_precision_t)input[3 + i];

        // 2) MLP1 (raw spectrum -> d_model)
        auto mlp1_out = {FN_NAME}_mlp1(spec_in, params + {MLP1_P}, fwd_ctx ? fwd_ctx + WARP_SIZE * {MLP1_O} : nullptr);

        // 3) Cast half -> float for LN2
        ::tcnn::tvec<float, {D}> ln2_in;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) ln2_in[i] = (float)mlp1_out[i];

        // 4) LN2 -> MLP2 (spectrum_encoded, half)
        auto ln2_out  = {FN_NAME}_ln2 (ln2_in,  params + {LN2_P},  fwd_ctx ? fwd_ctx + WARP_SIZE * {LN2_O}  : nullptr);
        auto mlp2_out = {FN_NAME}_mlp2(ln2_out, params + {MLP2_P}, fwd_ctx ? fwd_ctx + WARP_SIZE * {MLP2_O} : nullptr);

        // 5) Assemble [direction(3), spectrum_encoded(d_model)] (float) — no distance.
        ::tcnn::tvec<float, {CONCAT_W}> concat;
        concat[0] = input[0];
        concat[1] = input[1];
        concat[2] = input[2];
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) concat[3 + i] = (float)mlp2_out[i];

        // 6) Final beam parameter encoding (SH(direction) + None(spectrum_encoded))
        auto pse_out = {FN_NAME}_pse(concat, params + {PSE_P}, fwd_ctx ? fwd_ctx + WARP_SIZE * {PSE_O} : nullptr);

        // 7) LayerNorm on the d_model-wide ParameterSetEncoding output.
        ::tcnn::tvec<float, {D}> out_ln_in;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) out_ln_in[i] = (float)pse_out[i];
        return {FN_NAME}_out_ln(out_ln_in, params + {OUT_LN_P}, fwd_ctx ? fwd_ctx + WARP_SIZE * {OUT_LN_O} : nullptr);
    )",
        fmt::arg("FN_NAME",  fn_name),
        fmt::arg("SPEC",     this->spectrum_dim),
        fmt::arg("SPEC_PAD", this->spectrum_dim_padded),
        fmt::arg("D",        this->d_model),
        fmt::arg("CONCAT_W", 3u + this->spectrum_enc_dim),
        fmt::arg("MLP1_P",   MLP1_P),
        fmt::arg("LN2_P",    LN2_P),
        fmt::arg("MLP2_P",   MLP2_P),
        fmt::arg("PSE_P",    PSE_P),
        fmt::arg("OUT_LN_P", OUT_LN_P),
        fmt::arg("MLP1_O",   MLP1_O),
        fmt::arg("LN2_O",    LN2_O),
        fmt::arg("MLP2_O",   MLP2_O),
        fmt::arg("PSE_O",    PSE_O),
        fmt::arg("OUT_LN_O", OUT_LN_O)
    );

    return fmt::format("{}{}", preamble, this->generate_device_function_from_body(fn_name, body));
}

std::string SPERFBeamEncoder::generate_backward_device_function(const std::string& fn_name, uint32_t n_threads) const {
    const uint32_t MLP1_P   = 0;
    const uint32_t LN2_P    = MLP1_P   + ::tcnn::next_multiple<uint32_t>(this->spectrum_mlp1->n_params(),        NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t MLP2_P   = LN2_P    + ::tcnn::next_multiple<uint32_t>(this->spectrum_ln2->n_params(),         NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t PSE_P    = MLP2_P   + ::tcnn::next_multiple<uint32_t>(this->spectrum_mlp2->n_params(),        NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t OUT_LN_P = PSE_P    + ::tcnn::next_multiple<uint32_t>(this->beam_params_encoding->n_params(), NEXT_MULTIPLE_FOR_TYPE);

    const uint32_t MLP1_O   = 0;
    const uint32_t LN2_O    = MLP1_O   + ::tcnn::next_multiple<uint32_t>(this->spectrum_mlp1->device_function_fwd_ctx_bytes(),        NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t MLP2_O   = LN2_O    + ::tcnn::next_multiple<uint32_t>(this->spectrum_ln2->device_function_fwd_ctx_bytes(),         NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t PSE_O    = MLP2_O   + ::tcnn::next_multiple<uint32_t>(this->spectrum_mlp2->device_function_fwd_ctx_bytes(),        NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t OUT_LN_O = PSE_O    + ::tcnn::next_multiple<uint32_t>(this->beam_params_encoding->device_function_fwd_ctx_bytes(), NEXT_MULTIPLE_FOR_TYPE);

    std::string preamble = fmt::format(R"(
            {MLP1_BWD}

            {LN2_BWD}

            {MLP2_BWD}

            {PSE_BWD}

            {OUT_LN_BWD}

        )",
        fmt::arg("MLP1_BWD",   this->spectrum_mlp1->generate_backward_device_function(fn_name + "_mlp1_bwd", n_threads)),
        fmt::arg("LN2_BWD",    this->spectrum_ln2->generate_backward_device_function(fn_name + "_ln2_bwd",  n_threads)),
        fmt::arg("MLP2_BWD",   this->spectrum_mlp2->generate_backward_device_function(fn_name + "_mlp2_bwd", n_threads)),
        fmt::arg("PSE_BWD",    this->beam_params_encoding->generate_backward_device_function(fn_name + "_pse_bwd", n_threads)),
        fmt::arg("OUT_LN_BWD", this->beam_params_ln->generate_backward_device_function(fn_name + "_out_ln_bwd", n_threads))
    );

    std::string body = fmt::format(R"(
        // 0) Out-LN backward
        ::tcnn::tvec<float, {D}> dL_dpse_f;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) dL_dpse_f[i] = 0.f;
        {FN_NAME}_out_ln_bwd(dL_dy, params + {OUT_LN_P}, fwd_ctx + WARP_SIZE * {OUT_LN_O}, dL_dparams ? dL_dparams + {OUT_LN_P} : nullptr, &dL_dpse_f);

        ::tcnn::tvec<::tcnn::network_precision_t, {D}> dL_dpse;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) dL_dpse[i] = (::tcnn::network_precision_t)dL_dpse_f[i];

        // 1) PSE backward -> dL_dconcat (3 dir + D spec)
        ::tcnn::tvec<float, {CONCAT_W}> dL_dconcat;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {CONCAT_W}; i++) dL_dconcat[i] = 0.f;
        {FN_NAME}_pse_bwd(dL_dpse, params + {PSE_P}, fwd_ctx + WARP_SIZE * {PSE_O}, dL_dparams ? dL_dparams + {PSE_P} : nullptr, &dL_dconcat);

        // 2) Split: rows [3..3+D) are dL/d(mlp2_out)
        ::tcnn::tvec<::tcnn::network_precision_t, {D}> dL_dmlp2_out;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) dL_dmlp2_out[i] = (::tcnn::network_precision_t)dL_dconcat[3 + i];

        // 3) MLP2 backward
        ::tcnn::tvec<::tcnn::network_precision_t, {D}> dL_dln2_out;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) dL_dln2_out[i] = (::tcnn::network_precision_t)0.f;
        {FN_NAME}_mlp2_bwd(dL_dmlp2_out, params + {MLP2_P}, fwd_ctx + WARP_SIZE * {MLP2_O}, dL_dparams ? dL_dparams + {MLP2_P} : nullptr, &dL_dln2_out);

        // 4) LN2 backward
        ::tcnn::tvec<float, {D}> dL_dmlp1_out_f;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) dL_dmlp1_out_f[i] = 0.f;
        {FN_NAME}_ln2_bwd(dL_dln2_out, params + {LN2_P}, fwd_ctx + WARP_SIZE * {LN2_O}, dL_dparams ? dL_dparams + {LN2_P} : nullptr, &dL_dmlp1_out_f);

        // 5) Cast float -> half across the mlp1/ln2 boundary
        ::tcnn::tvec<::tcnn::network_precision_t, {D}> dL_dmlp1_out;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) dL_dmlp1_out[i] = (::tcnn::network_precision_t)dL_dmlp1_out_f[i];

        // 6) MLP1 backward -> dL_dspec_h (half; MLP1 is Network<half,half>).
        ::tcnn::tvec<::tcnn::network_precision_t, {SPEC_PAD}> dL_dspec_h;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {SPEC_PAD}; i++) dL_dspec_h[i] = (::tcnn::network_precision_t)0.f;
        {FN_NAME}_mlp1_bwd(dL_dmlp1_out, params + {MLP1_P}, fwd_ctx + WARP_SIZE * {MLP1_O}, dL_dparams ? dL_dparams + {MLP1_P} : nullptr, &dL_dspec_h);

        // 7) Assemble dL_dx: direction from dL_dconcat[0..3), spectrum cast
        //    back to float at the encoder-input boundary. No distance slot.
        if (dL_dx) {{
            (*dL_dx)[0] = dL_dconcat[0];
            (*dL_dx)[1] = dL_dconcat[1];
            (*dL_dx)[2] = dL_dconcat[2];
            TCNN_PRAGMA_UNROLL
            for (int i = 0; i < {SPEC}; i++) (*dL_dx)[3 + i] = (float)dL_dspec_h[i];
        }}
    )",
        fmt::arg("FN_NAME",  fn_name),
        fmt::arg("SPEC",     this->spectrum_dim),
        fmt::arg("SPEC_PAD", this->spectrum_dim_padded),
        fmt::arg("D",        this->d_model),
        fmt::arg("CONCAT_W", 3u + this->spectrum_enc_dim),
        fmt::arg("MLP1_P",   MLP1_P),
        fmt::arg("LN2_P",    LN2_P),
        fmt::arg("MLP2_P",   MLP2_P),
        fmt::arg("PSE_P",    PSE_P),
        fmt::arg("OUT_LN_P", OUT_LN_P),
        fmt::arg("MLP1_O",   MLP1_O),
        fmt::arg("LN2_O",    LN2_O),
        fmt::arg("MLP2_O",   MLP2_O),
        fmt::arg("PSE_O",    PSE_O),
        fmt::arg("OUT_LN_O", OUT_LN_O)
    );

    return fmt::format("{}{}", preamble, this->generate_backward_device_function_from_body(fn_name, body));
}
