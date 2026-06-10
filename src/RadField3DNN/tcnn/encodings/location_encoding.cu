#include "radfield3d-nn/tcnn/encodings/location_encoding.h"
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/network.h>
#include <radfield3d-nn/tcnn/helper_kernels.cuh>


rfnn::tcnn::LocationEncoding::LocationEncoding(uint32_t frequencies, uint32_t encoded_dims,
                                          rfnn::tcnn::LocationEncodingKind kind)
    : encoded_dims(encoded_dims),
      frequencies(frequencies),
      kind(kind)
{
    if (kind == rfnn::tcnn::LocationEncodingKind::Frequency && encoded_dims < 4) {
        throw std::invalid_argument(
            "LocationEncoding(Frequency): encoded_dims must be >= 4 — the MLP's "
            "output width must allow at least one learnable channel.");
    }

    const std::string network_type = (encoded_dims > 128) ? "CutlassMLP" : "FullyFusedMLP";

    if (kind == rfnn::tcnn::LocationEncodingKind::Frequency) {
        // Real Fourier output width: 2 sin/cos values per (axis, frequency) = 6F.
        this->freq_real_out_width = frequencies * 3u * 2u;

        // Make sure we have AT LEAST 3 alignment pad slots beyond the real
        // Fourier output, so we can drop xyz into them as a learnable skip
        // connection feeding into the inner MLP (mirrors Python's
        // SinusoidalFrequencyEncoding(append_input=True), where the next
        // Linear sees [Fourier(xyz), xyz] concatenated).
        //
        // next_multiple(6F + 3, 16) guarantees padded - 6F >= 3:
        //   * 6F=72 -> 75 -> 80   (spare 8)
        //   * 6F=48 -> 51 -> 64   (spare 16) — would have been 48 w/o the +3
        //   * 6F=96 -> 99 -> 112  (spare 16) — would have been 96 w/o the +3
        // The over-allocated slots stay at the Frequency encoder's default
        // pad value (1.0f); the inner MLP learns to ignore them just like
        // any constant input.
        this->padded_loc_enc_dims = ::tcnn::next_multiple<uint32_t>(this->freq_real_out_width + 3u, 16u);
        this->xyz_inject_offset   = this->freq_real_out_width;

        // Standalone Frequency encoder. set_padded_output_width is honoured
        // via the `alignment` arg to create_encoding; we round up using LCM
        // of the requested alignment and the encoder's required alignment
        // (8 for FP16) so the encoder's set_padded_output_width pre-check
        // (padded >= n_output_dims) is always satisfied.
        this->freq_encoding.reset(::tcnn::create_encoding<::tcnn::network_precision_t>(
            3u,
            ::tcnn::json{
                {"otype", "Frequency"},
                {"n_frequencies", frequencies},
                {"n_dims_to_encode", 3}
            },
            this->padded_loc_enc_dims  // alignment arg drives set_padded_output_width
        ));

        // The MLP takes the full padded encoding width (including the 3 xyz
        // slots) and projects to d_model. n_hidden_layers=1 (2 weight
        // matrices) — this IS the "first MLP block" in the PBRFNetCPP
        // architecture (xyz → fourier → mlp(2) → FiLM → ...), analog to
        // Python's `RFBackboneModel.block1`.
        this->freq_mlp.reset(::tcnn::create_network<::tcnn::network_precision_t>(
            ::tcnn::json{
                {"otype", network_type},
                {"n_input_dims", this->padded_loc_enc_dims},
                {"n_neurons", encoded_dims},
                {"n_output_dims", encoded_dims},
                {"n_hidden_layers", 1},
                {"activation", "SiLU"},
                {"output_activation", "None"}
            }
        ));

        this->freq_encoding->set_jit_fusion(true);
        this->freq_mlp->set_jit_fusion(true);
    } else {
        // HashGrid (Instant-NGP). `frequencies` is reinterpreted as
        // `n_levels`. Default sizing from Müller 2022 §3.2:
        //   features_per_level=2, base_resolution=16, log2_hashmap=19,
        //   per_level_scale=2.0 (geometric growth across levels).
        const uint32_t n_levels = frequencies;
        const uint32_t feats_per_level = 2;
        const uint32_t raw = n_levels * feats_per_level;
        this->padded_loc_enc_dims = ::tcnn::next_multiple<uint32_t>(raw, 16u);
        ::tcnn::json encoding_cfg = ::tcnn::json{
            {"otype", "HashGrid"},
            {"n_levels", n_levels},
            {"n_features_per_level", feats_per_level},
            {"base_resolution", 16},
            {"log2_hashmap_size", 19},
            {"per_level_scale", 2.0},
            {"interpolation", "Linear"},
            {"alignment", 16}
        };

        this->mlp_encoding_block.reset(
            new ::tcnn::NetworkWithInputEncoding<::tcnn::network_precision_t>(
                3,
                encoded_dims,
                encoding_cfg,
                ::tcnn::json{
                    {"otype", network_type},
                    {"n_input_dims", this->padded_loc_enc_dims},
                    {"n_neurons", encoded_dims},
                    {"n_output_dims", encoded_dims},
                    {"n_hidden_layers", 2},
                    {"activation", "SiLU"},
                    {"output_activation", "None"}
                }
            )
        );
        this->mlp_encoding_block->set_jit_fusion(true);
    }

    this->set_jit_fusion(true);
}

std::string rfnn::tcnn::LocationEncoding::generate_device_function(const std::string& fn_name) const {
    // HashGrid: passthrough — the hash lookup is already an explicit
    // function of position, no append-input skip is needed.
    if (this->kind == rfnn::tcnn::LocationEncodingKind::HashGrid) {
        return this->mlp_encoding_block->generate_device_function(fn_name);
    }

    // Frequency: call the encoder, overwrite the first 3 alignment-pad
    // slots with raw xyz, then call the MLP. This matches the Python
    // SinusoidalFrequencyEncoding(append_input=True) behaviour where the
    // MLP's input is [sin/cos features, raw xyz] concatenated.
    std::string enc_fn = this->freq_encoding->generate_device_function(fn_name + "_enc");
    std::string mlp_fn = this->freq_mlp     ->generate_device_function(fn_name + "_mlp");

    std::string body = fmt::format(R"(
        // 1) Fourier encoding (xyz -> {PADDED} half channels).
        //    The Frequency encoder fills [{REAL_OUT}..{PADDED}) with 1.0f as
        //    alignment-pad — those are the slots we overwrite below.
        auto enc_out = {FN_NAME}_enc(input, params /* freq has 0 params */, fwd_ctx);

        // 2) Inject raw xyz into the FIRST 3 alignment-pad slots so the
        //    inner MLP gets [Fourier(xyz), xyz, …pad…] as input.
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < 3; i++) enc_out[{XYZ_OFF} + i] = (::tcnn::network_precision_t)input[i];

        // 3) Project through the inner MLP. params layout: {{ MLP params }}
        //    (Frequency encoding is parameterless).
        return {FN_NAME}_mlp(enc_out, params, fwd_ctx ? fwd_ctx + WARP_SIZE * {ENC_CTX} : nullptr);
    )",
        fmt::arg("FN_NAME",  fn_name),
        fmt::arg("PADDED",   this->padded_loc_enc_dims),
        fmt::arg("REAL_OUT", this->freq_real_out_width),
        fmt::arg("XYZ_OFF",  this->xyz_inject_offset),
        fmt::arg("ENC_CTX",  ::tcnn::next_multiple<uint32_t>(this->freq_encoding->device_function_fwd_ctx_bytes(),
                                                          (sizeof(::tcnn::network_precision_t) == 2) ? 8u : 4u))
    );

    return fmt::format("{}{}{}", enc_fn, mlp_fn,
                       this->generate_device_function_from_body(fn_name, body));
}

std::string rfnn::tcnn::LocationEncoding::generate_backward_device_function(const std::string& name, uint32_t n_threads) const {
    if (this->kind == rfnn::tcnn::LocationEncodingKind::HashGrid) {
        return this->mlp_encoding_block->generate_backward_device_function(name, n_threads);
    }

    // Frequency backward: split dL_dy into the path through the MLP, then
    // through the encoder for the real Fourier slots, then add a direct
    // copy of the xyz-skip slots' gradients to dL_dxyz.
    std::string enc_bwd = this->freq_encoding->generate_backward_device_function(name + "_enc_bwd", n_threads);
    std::string mlp_bwd = this->freq_mlp     ->generate_backward_device_function(name + "_mlp_bwd", n_threads);

    std::string body = fmt::format(R"(
        // 1) MLP backward: dL_dy ({D} half) -> dL_denc ({PADDED} half).
        //    The MLP backward writes the gradient for every input channel,
        //    including the 3 we hijacked for xyz and the >=N spare pad slots
        //    (which the Frequency forward filled with constant 1.0f — their
        //    gradient is real but the forward had no learnable contribution
        //    from them, so it just gets discarded after this step).
        ::tcnn::tvec<::tcnn::network_precision_t, {PADDED}> dL_denc;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {PADDED}; i++) dL_denc[i] = (::tcnn::network_precision_t)0.f;
        {FN_NAME}_mlp_bwd(dL_dy, params, fwd_ctx + WARP_SIZE * {ENC_CTX}, dL_dparams, &dL_denc);

        // 2) Capture the xyz-skip gradient BEFORE feeding dL_denc into the
        //    encoder backward. The encoder backward only reads
        //    dL_denc[0..{REAL_OUT}) (see frequency.h:N_DIMS*N_FREQUENCIES*2
        //    loop), so the [{XYZ_OFF}..{XYZ_OFF}+3) slots are naturally
        //    ignored on the encoder path — but we DO need them here as the
        //    direct skip-connection gradient back to dL_dxyz.
        ::tcnn::tvec<float, 3> dL_dxyz_skip;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < 3; i++) dL_dxyz_skip[i] = (float)dL_denc[{XYZ_OFF} + i];

        // 3) Encoder backward: reads dL_denc[0..{REAL_OUT}) only. The
        //    Frequency encoder is parameterless so dL_dparams is unused;
        //    the encoder writes its dL_dxyz_freq output.
        ::tcnn::tvec<float, 3> dL_dxyz_freq;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < 3; i++) dL_dxyz_freq[i] = 0.f;
        {FN_NAME}_enc_bwd(dL_denc, params, fwd_ctx, nullptr /* freq has 0 params */, &dL_dxyz_freq);

        // 4) Combine: dL_dxyz = (Fourier path) + (xyz-skip path).
        if (dL_dx) {{
            TCNN_PRAGMA_UNROLL
            for (int i = 0; i < 3; i++) (*dL_dx)[i] = dL_dxyz_freq[i] + dL_dxyz_skip[i];
        }}
    )",
        fmt::arg("FN_NAME",  name),
        fmt::arg("D",        this->encoded_dims),
        fmt::arg("PADDED",   this->padded_loc_enc_dims),
        fmt::arg("REAL_OUT", this->freq_real_out_width),
        fmt::arg("XYZ_OFF",  this->xyz_inject_offset),
        fmt::arg("ENC_CTX",  ::tcnn::next_multiple<uint32_t>(this->freq_encoding->device_function_fwd_ctx_bytes(),
                                                          (sizeof(::tcnn::network_precision_t) == 2) ? 8u : 4u))
    );

    return fmt::format("{}{}{}", enc_bwd, mlp_bwd,
                       this->generate_backward_device_function_from_body(name, body));
}

size_t rfnn::tcnn::LocationEncoding::n_params() const {
    if (this->kind == rfnn::tcnn::LocationEncodingKind::HashGrid) {
        return this->mlp_encoding_block->n_params();
    }
    // Frequency: encoder has 0 params, all params come from the MLP.
    return this->freq_mlp->n_params();
}

void rfnn::tcnn::LocationEncoding::set_params_impl(::tcnn::network_precision_t* params, ::tcnn::network_precision_t* inference_params, ::tcnn::network_precision_t* gradients) {
    if (this->kind == rfnn::tcnn::LocationEncodingKind::HashGrid) {
        this->mlp_encoding_block->set_params(params, inference_params, gradients);
        return;
    }
    // Frequency encoder is parameterless; the whole param slab is the MLP's.
    this->freq_mlp->set_params(params, inference_params, gradients);
}

std::vector<std::pair<uint32_t, uint32_t>> rfnn::tcnn::LocationEncoding::layer_sizes() const {
    if (this->kind == rfnn::tcnn::LocationEncodingKind::HashGrid) {
        return this->mlp_encoding_block->layer_sizes();
    }
    return this->freq_mlp->layer_sizes();
}

void rfnn::tcnn::LocationEncoding::initialize_params(::tcnn::pcg32& rnd, float* params_full_precision, float scale) {
    if (this->kind == rfnn::tcnn::LocationEncodingKind::HashGrid) {
        this->mlp_encoding_block->initialize_params(rnd, params_full_precision, scale);
        return;
    }
    this->freq_mlp->initialize_params(rnd, params_full_precision, scale);
}

uint32_t rfnn::tcnn::LocationEncoding::device_function_fwd_ctx_bytes() const {
    if (this->kind == rfnn::tcnn::LocationEncodingKind::HashGrid) {
        return this->mlp_encoding_block->device_function_fwd_ctx_bytes();
    }
    // Frequency: encoder ctx (xyz stash for backward) + MLP ctx slabs.
    const uint32_t MULT = (sizeof(::tcnn::network_precision_t) == 2) ? 8u : 4u;
    return ::tcnn::next_multiple<uint32_t>(this->freq_encoding->device_function_fwd_ctx_bytes(), MULT)
         + ::tcnn::next_multiple<uint32_t>(this->freq_mlp     ->device_function_fwd_ctx_bytes(), MULT);
}

std::shared_ptr<::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>> rfnn::tcnn::LocationEncoding::encode_locations(const ::tcnn::GPUMatrixDynamic<float>& input)
{
    const uint32_t B = input.n();
    std::shared_ptr<::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>> encoded = std::make_shared<::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>>(this->encoded_dims, B);

    this->forward(
        input,
        encoded.get(),
        true,
        false
    );

    return encoded;
}
