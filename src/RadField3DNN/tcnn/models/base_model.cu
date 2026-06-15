#include "radfield3d-nn/tcnn/base_model.h"
#include "radfield3d-nn/tcnn/blocks/mlp_select.h"
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <cstdlib>
#include <iostream>


using namespace rfnn::tcnn;

// -----------------------------------------------------------------------------
// BaseRadiationPredictionModel (main forward)
// -----------------------------------------------------------------------------

BaseRadiationPredictionModel::BaseRadiationPredictionModel(uint32_t d_model, uint32_t location_encoding_dim, float flux_offset, int flux_activation,
                                                            LocationEncodingKind location_encoding_kind,
                                                            float flux_clamp_min, float flux_clamp_max,
                                                            uint32_t trunk_hidden_layers,
                                                            BeamFusionKind beam_fusion)
    : d_model(d_model),
      flux_offset(flux_offset),
      flux_activation(flux_activation),
      flux_clamp_min(flux_clamp_min),
      flux_clamp_max(flux_clamp_max),
      location_encoding_kind(location_encoding_kind),
      trunk_hidden_layers(trunk_hidden_layers),
      beam_fusion_kind(beam_fusion),
      NEXT_MULTIPLE_FOR_TYPE((sizeof(::tcnn::network_precision_t) == 2) ? 8 : 4)
{
    // Per-MLP backend choice: FullyFusedMLP when in/out <= 128 and there is a
    // hidden layer, else CutlassMLP (see mlp_select.h). mlp_block and mlp_post
    // are both square (d_model -> d_model) — they stack into the flat
    // pure-Python `block2` (no concat-skip).
    const std::string mlp_block_type = select_mlp_otype(d_model, d_model, trunk_hidden_layers);
    const std::string mlp_post_type  = select_mlp_otype(d_model, d_model, trunk_hidden_layers);

    this->location_encoding.reset(new rfnn::tcnn::LocationEncoding(location_encoding_dim, d_model, location_encoding_kind));

    this->beam_conditioner1 = rfnn::tcnn::create_beam_fusion(beam_fusion, d_model, d_model, "SiLU");

    // mlp_block + mlp_post = the flat pure-Python `block2` (4 Linears, SiLU
    // between all, no activation before FiLM2; NO mid-trunk concat-skip).
    // mlp_block ends with SiLU (output_activation SiLU) so the boundary between
    // the two stacked MLPs has an activation, making
    // [mlp_block: Lin,SiLU,Lin,SiLU][mlp_post: Lin,SiLU,Lin] = block2.
    // Input is x1 = FILM1(loc_enc, beam).
    this->mlp_block.reset(
        ::tcnn::create_network<::tcnn::network_precision_t>(
            ::tcnn::json{
                {"otype", mlp_block_type},
                {"n_input_dims", d_model},
                {"n_neurons", d_model},
                {"n_output_dims", d_model},
                {"n_hidden_layers", trunk_hidden_layers},
                {"activation", "SiLU"},
                {"output_activation", "SiLU"}
            }
        )
    );

    // Second half of `block2`. Input is mlp_block's output (d_model) — the
    // loc_enc concat-skip is REMOVED to mirror pure-Python (loc_enc only feeds
    // FiLM1 and the terminal `x3 + loc_enc` skip, like Python's `x2 + x1`).
    this->mlp_post.reset(
        ::tcnn::create_network<::tcnn::network_precision_t>(
            ::tcnn::json{
                {"otype", mlp_post_type},
                {"n_input_dims", d_model},
                {"n_neurons", d_model},
                {"n_output_dims", d_model},
                {"n_hidden_layers", trunk_hidden_layers},
                {"activation", "SiLU"},
                {"output_activation", "None"}
            }
        )
    );

    this->beam_conditioner2 = rfnn::tcnn::create_beam_fusion(beam_fusion, d_model, d_model, "SiLU");

    this->mlp_spectrum_decode.reset(
        ::tcnn::create_network<::tcnn::network_precision_t>(
            ::tcnn::json{
                {"otype", select_mlp_otype(d_model, 32u, 1u)},
                {"n_input_dims", d_model},
                // Hidden width d_model/2 to MIRROR the pure-Python spectra_decoder
                // (Linear(d_model, d_model/2) -> SiLU -> Linear(d_model/2, 32)).
                // The previous d_model-wide hidden let the spectrum head absorb
                // the beam-driven spectrum WITHOUT engaging the spatial trunk, so
                // its trunk-gradient norm stayed tiny and DB-MTL up-weighted
                // spectrum (inverting the pure-Python balance, overflowing fp16).
                {"n_neurons", (d_model / 2u >= 16u) ? (d_model / 2u) : 16u},
                {"n_output_dims", 32u},
                // n_hidden=1 (2 weights) per the uniform 2-layer-per-MLP
                // architecture spec. The output Linear maps d_model → 32
                // bins; the softplus + sum-norm activation in the JIT body
                // provides the histogram structure.
                {"n_hidden_layers", 1},
                {"activation", "SiLU"},
                {"output_activation", "None"}
            }
        )
    );

    // Single (joined) flux projector — a bias-less MLP. n_hidden=1
    // (2 weight matrices) gives the head a SiLU non-linearity in front of its
    // final scalar projection so it can model the bimodal "in beam / out of
    // beam" decision. (The historic second "direct" head was removed: the
    // single-head joined-flux net matches the pure-Python PBRFNet that trains
    // best — the split-flux two-head variant underperformed it.)
    this->flux_projector.reset(
        ::tcnn::create_network<::tcnn::network_precision_t>(
            ::tcnn::json{
                {"otype", select_mlp_otype(d_model, 16u, 1u)},
                {"n_input_dims", d_model},
                {"n_neurons", d_model},
                {"n_output_dims", 16u},   // logically 1; padded for tile alignment
                {"n_hidden_layers", 1},
                {"activation", "SiLU"},
                {"output_activation", "None"}
            }
        )
    );

    this->total_parameter_count = 0;
    this->total_parameter_count += location_encoding->n_params();
    this->total_parameter_count += beam_conditioner1->n_params();
    this->total_parameter_count += mlp_block->n_params();
    this->total_parameter_count += mlp_post->n_params();
    this->total_parameter_count += beam_conditioner2->n_params();
    this->total_parameter_count += mlp_spectrum_decode->n_params();
    this->total_parameter_count += flux_projector->n_params();

    this->set_jit_fusion(true);
}

BaseRadiationPredictionModel::~BaseRadiationPredictionModel() = default;

size_t BaseRadiationPredictionModel::n_params() const {
    return this->total_parameter_count;
}

size_t BaseRadiationPredictionModel::output_head_param_offset() const {
    // Mirror the padded offset accumulation in set_params_impl, summing every
    // sub-module BEFORE the output heads (mlp_spectrum_decode, flux_projector).
    auto pad = [this](size_t n) { return ::tcnn::next_multiple<size_t>(n, NEXT_MULTIPLE_FOR_TYPE); };
    size_t offset = 0;
    offset += pad(this->location_encoding->n_params());
    offset += pad(this->beam_conditioner1->n_params());
    offset += pad(this->mlp_block->n_params());
    offset += pad(this->mlp_post->n_params());
    offset += pad(this->beam_conditioner2->n_params());
    return offset;
}

void BaseRadiationPredictionModel::set_params_impl(::tcnn::network_precision_t* params, ::tcnn::network_precision_t* inference_params, ::tcnn::network_precision_t* gradients) {
    size_t offset = 0;
    auto advance_offset = [&](size_t n) {
        offset += ::tcnn::next_multiple<size_t>(n, NEXT_MULTIPLE_FOR_TYPE);
    };

    this->location_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
    advance_offset(this->location_encoding->n_params());

    this->beam_conditioner1->set_params(params + offset, inference_params + offset, gradients + offset);
    advance_offset(this->beam_conditioner1->n_params());

    this->mlp_block->set_params(params + offset, inference_params + offset, gradients + offset);
    advance_offset(this->mlp_block->n_params());

    this->mlp_post->set_params(params + offset, inference_params + offset, gradients + offset);
    advance_offset(this->mlp_post->n_params());

    this->beam_conditioner2->set_params(params + offset, inference_params + offset, gradients + offset);
    advance_offset(this->beam_conditioner2->n_params());

    this->mlp_spectrum_decode->set_params(params + offset, inference_params + offset, gradients + offset);
    advance_offset(this->mlp_spectrum_decode->n_params());

    this->flux_projector->set_params(params + offset, inference_params + offset, gradients + offset);
    advance_offset(this->flux_projector->n_params());
}

std::vector<std::pair<uint32_t, uint32_t>> BaseRadiationPredictionModel::layer_sizes() const {
    std::vector<std::pair<uint32_t, uint32_t>> out;
    for (auto& l : this->location_encoding->layer_sizes())     out.push_back(l);
    for (auto& l : this->beam_conditioner1->layer_sizes())     out.push_back(l);
    for (auto& l : this->mlp_block->layer_sizes())             out.push_back(l);
    for (auto& l : this->mlp_post->layer_sizes())           out.push_back(l);
    for (auto& l : this->beam_conditioner2->layer_sizes())     out.push_back(l);
    for (auto& l : this->mlp_spectrum_decode->layer_sizes())   out.push_back(l);
    for (auto& l : this->flux_projector->layer_sizes())        out.push_back(l);
    return out;
}

void BaseRadiationPredictionModel::initialize_params(::tcnn::pcg32& rnd, float* p, float scale) {
    float* cur = p;
    auto step = [&](auto& m) {
        m->initialize_params(rnd, cur, scale);
        cur += ::tcnn::next_multiple<size_t>(m->n_params(), NEXT_MULTIPLE_FOR_TYPE);
    };
    step(this->location_encoding);
    step(this->beam_conditioner1);
    step(this->mlp_block);
    step(this->mlp_post);
    step(this->beam_conditioner2);
    step(this->mlp_spectrum_decode);
    step(this->flux_projector);
}

uint32_t BaseRadiationPredictionModel::device_function_fwd_ctx_bytes() const {
    uint32_t bytes = 0;
    bytes += ::tcnn::next_multiple<uint32_t>(this->location_encoding->device_function_fwd_ctx_bytes(),     NEXT_MULTIPLE_FOR_TYPE);
    bytes += ::tcnn::next_multiple<uint32_t>(this->beam_conditioner1->device_function_fwd_ctx_bytes(),     NEXT_MULTIPLE_FOR_TYPE);
    bytes += ::tcnn::next_multiple<uint32_t>(this->mlp_block->device_function_fwd_ctx_bytes(),             NEXT_MULTIPLE_FOR_TYPE);
    bytes += ::tcnn::next_multiple<uint32_t>(this->mlp_post->device_function_fwd_ctx_bytes(),           NEXT_MULTIPLE_FOR_TYPE);
    bytes += ::tcnn::next_multiple<uint32_t>(this->beam_conditioner2->device_function_fwd_ctx_bytes(),     NEXT_MULTIPLE_FOR_TYPE);
    bytes += ::tcnn::next_multiple<uint32_t>(this->mlp_spectrum_decode->device_function_fwd_ctx_bytes(),   NEXT_MULTIPLE_FOR_TYPE);
    bytes += ::tcnn::next_multiple<uint32_t>(this->flux_projector->device_function_fwd_ctx_bytes(),         NEXT_MULTIPLE_FOR_TYPE);
    // act_ctx slab: 32 spec pre-activation logits + 1 flux-y. The flux y slot is
    // used only by the SoftClip backward (Jacobian 2*y*(1-y)); the hard-clamp
    // backward is gradient-conserving and ignores it (negligible stash cost).
    bytes += ::tcnn::next_multiple<uint32_t>(static_cast<uint32_t>(33u * sizeof(float)), NEXT_MULTIPLE_FOR_TYPE);
    return bytes;
}

uint32_t BaseRadiationPredictionModel::backward_device_function_shmem_bytes(uint32_t n_threads, ::tcnn::GradientMode mode) const {
    uint32_t bytes = 0;
    bytes = std::max(bytes, this->location_encoding->backward_device_function_shmem_bytes(n_threads, mode));
    bytes = std::max(bytes, this->beam_conditioner1->backward_device_function_shmem_bytes(n_threads, mode));
    bytes = std::max(bytes, this->mlp_block->backward_device_function_shmem_bytes(n_threads, mode));
    bytes = std::max(bytes, this->mlp_post->backward_device_function_shmem_bytes(n_threads, mode));
    bytes = std::max(bytes, this->beam_conditioner2->backward_device_function_shmem_bytes(n_threads, mode));
    bytes = std::max(bytes, this->mlp_spectrum_decode->backward_device_function_shmem_bytes(n_threads, mode));
    bytes = std::max(bytes, this->flux_projector->backward_device_function_shmem_bytes(n_threads, mode));
    return bytes;
}

void BaseRadiationPredictionModel::convert_params_to_jit_layout(cudaStream_t stream, bool use_inference_params) {
    this->location_encoding->convert_params_to_jit_layout(stream, use_inference_params);
    this->mlp_block->convert_params_to_jit_layout(stream, use_inference_params);
    this->mlp_post->convert_params_to_jit_layout(stream, use_inference_params);
    this->mlp_spectrum_decode->convert_params_to_jit_layout(stream, use_inference_params);
    this->flux_projector->convert_params_to_jit_layout(stream, use_inference_params);
}

void BaseRadiationPredictionModel::convert_params_from_jit_layout(cudaStream_t stream, bool use_inference_params) {
    this->location_encoding->convert_params_from_jit_layout(stream, use_inference_params);
    this->mlp_block->convert_params_from_jit_layout(stream, use_inference_params);
    this->mlp_post->convert_params_from_jit_layout(stream, use_inference_params);
    this->mlp_spectrum_decode->convert_params_from_jit_layout(stream, use_inference_params);
    this->flux_projector->convert_params_from_jit_layout(stream, use_inference_params);
}

std::string BaseRadiationPredictionModel::generate_device_function(const std::string& fn_name) const {
    const uint32_t loc_ctx    = ::tcnn::next_multiple<uint32_t>(this->location_encoding->device_function_fwd_ctx_bytes(),     NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t film1_ctx  = ::tcnn::next_multiple<uint32_t>(this->beam_conditioner1->device_function_fwd_ctx_bytes(),     NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t block_ctx  = ::tcnn::next_multiple<uint32_t>(this->mlp_block->device_function_fwd_ctx_bytes(),             NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t post_ctx   = ::tcnn::next_multiple<uint32_t>(this->mlp_post->device_function_fwd_ctx_bytes(),               NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t film2_ctx  = ::tcnn::next_multiple<uint32_t>(this->beam_conditioner2->device_function_fwd_ctx_bytes(),     NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t spec_ctx   = ::tcnn::next_multiple<uint32_t>(this->mlp_spectrum_decode->device_function_fwd_ctx_bytes(),   NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t flu_ctx    = ::tcnn::next_multiple<uint32_t>(this->flux_projector->device_function_fwd_ctx_bytes(),        NEXT_MULTIPLE_FOR_TYPE);

    const uint32_t loc_off    = 0;
    const uint32_t film1_off  = loc_off    + loc_ctx;
    const uint32_t block_off  = film1_off  + film1_ctx;
    const uint32_t post_off   = block_off  + block_ctx;
    const uint32_t film2_off  = post_off   + post_ctx;
    const uint32_t spec_off   = film2_off  + film2_ctx;
    const uint32_t flu_off    = spec_off   + spec_ctx;
    // Activation context slab: 32 spec pre-activation logits + 1 flux y per thread.
    const uint32_t act_off    = flu_off    + flu_ctx;

    const uint32_t loc_params    = ::tcnn::next_multiple<uint32_t>(this->location_encoding->n_params(),       NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t film1_params  = ::tcnn::next_multiple<uint32_t>(this->beam_conditioner1->n_params(),       NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t block_params  = ::tcnn::next_multiple<uint32_t>(this->mlp_block->n_params(),               NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t post_params   = ::tcnn::next_multiple<uint32_t>(this->mlp_post->n_params(),                NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t film2_params  = ::tcnn::next_multiple<uint32_t>(this->beam_conditioner2->n_params(),       NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t spec_params   = ::tcnn::next_multiple<uint32_t>(this->mlp_spectrum_decode->n_params(),     NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t flu_params    = ::tcnn::next_multiple<uint32_t>(this->flux_projector->n_params(),          NEXT_MULTIPLE_FOR_TYPE);

    const uint32_t loc_p    = 0;
    const uint32_t film1_p  = loc_p    + loc_params;
    const uint32_t block_p  = film1_p  + film1_params;
    const uint32_t post_p   = block_p  + block_params;
    const uint32_t film2_p  = post_p   + post_params;
    const uint32_t spec_p   = film2_p   + film2_params;
    const uint32_t flu_p    = spec_p   + spec_params;

    std::string preamble = fmt::format(R"(
            {LOC_FUNC}

            {FILM1_FUNC}

            {BLOCK_FUNC}

            {POST_FUNC}

            {FILM2_FUNC}

            {SPEC_FUNC}

            {FLU_FUNC}

        )",
        fmt::arg("LOC_FUNC",    this->location_encoding->generate_device_function(fn_name + "_loc")),
        fmt::arg("FILM1_FUNC",  this->beam_conditioner1->generate_device_function(fn_name + "_film1")),
        fmt::arg("BLOCK_FUNC",  this->mlp_block->generate_device_function(fn_name + "_block")),
        fmt::arg("POST_FUNC",   this->mlp_post->generate_device_function(fn_name + "_post")),
        fmt::arg("FILM2_FUNC",  this->beam_conditioner2->generate_device_function(fn_name + "_film2")),
        fmt::arg("SPEC_FUNC",   this->mlp_spectrum_decode->generate_device_function(fn_name + "_spec")),
        fmt::arg("FLU_FUNC",    this->flux_projector->generate_device_function(fn_name + "_flu"))
    );

    std::string body = fmt::format(R"(
        ::tcnn::tvec<float, 3> xyz_in{{ input[0], input[1], input[2] }};
        auto loc_enc = {FN_NAME}_loc(xyz_in, params + {LOC_P}, fwd_ctx ? fwd_ctx + WARP_SIZE * {LOC_O} : nullptr);

        ::tcnn::tvec<float, {TWO_D}> film1_in;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) film1_in[i] = (float)loc_enc[i];
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) film1_in[{D} + i] = input[3 + i];
        auto x1 = {FN_NAME}_film1(film1_in, params + {FILM1_P}, fwd_ctx ? fwd_ctx + WARP_SIZE * {FILM1_O} : nullptr);
        auto x2 = {FN_NAME}_block(x1, params + {BLOCK_P}, fwd_ctx ? fwd_ctx + WARP_SIZE * {BLOCK_O} : nullptr);

        // mlp_block + mlp_post = the flat pure-Python `block2` (no concat-skip).
        // mlp_post takes mlp_block's output directly (loc_enc only feeds FiLM1
        // and the terminal `x3 + loc_enc` skip below).
        auto x2_merged = {FN_NAME}_post(x2, params + {POST_P}, fwd_ctx ? fwd_ctx + WARP_SIZE * {POST_O} : nullptr);

        ::tcnn::tvec<float, {TWO_D}> film2_in;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) film2_in[i] = (float)x2_merged[i];
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) film2_in[{D} + i] = input[3 + i];

        auto x3 = {FN_NAME}_film2(film2_in, params + {FILM2_P}, fwd_ctx ? fwd_ctx + WARP_SIZE * {FILM2_O} : nullptr);

        // Additive residual just before the decoders: add the LocationEncoding
        // output (loc_enc) — "xyz after first MLP block" — to the post-FILM2
        // trunk activation. This is the NeRF-style identity skip from the
        // encoder to the decoders, giving the decoders a residual path to
        // raw positional info.
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) x3[i] = (::tcnn::network_precision_t)((float)x3[i] + (float)loc_enc[i]);

        // Decoders are bare Linear layers; no per-channel bias — a single
        // configurable additive `flux_offset` replaces the bias for each flux
        // head, and the spectrum head's bias was absorbed into Softplus+sum-
        // normalise (a uniform shift to all bins is a no-op after sum-norm).
        auto spec  = {FN_NAME}_spec (x3, params + {SPEC_P},   fwd_ctx ? fwd_ctx + WARP_SIZE * {SPEC_O}   : nullptr);
        auto flu   = {FN_NAME}_flu  (x3, params + {FLU_P},    fwd_ctx ? fwd_ctx + WARP_SIZE * {FLU_O}    : nullptr);

        // Flux activation. Two variants compiled into the kernel, applied
        // INDEPENDENTLY to each head:
        //   0 (clamp)     — gradient-conserving hard clamp
        //                   y = clamp(z + flux_offset_*, flux_clamp_min_*, flux_clamp_max_*).
        //                   Identity Jacobian inside; "predict 0 forever" lock-in
        //                   is possible. Per-head ranges allow each channel to
        //                   pick its own tonemap codomain.
        //   1 (softclip)  — y = 0.5*(tanh(z) + 1). Smooth (0,1); flux_offset and
        //                   flux_clamp_* are ignored because softclip already
        //                   lands at 0.5 at z=0 and is fixed to [0, 1].
        // NOTE: SoftClip here is the generic DSP/DL "soft clipping" form, NOT
        // related to the SoftCLIP cross-modal paper.
        float flu_y_s;
        {FLUX_ACTIVATION_FWD}

        float spec_z[32];
        float spec_sp[32];
        float spec_S = 0.0f;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < 32; i++) {{
            const float z = fminf(fmaxf((float)spec[i], -{CLIP}), {CLIP});
            spec_z[i] = z;
            // Numerically stable softplus: log1p(exp(-|z|)) + max(z, 0)
            spec_sp[i] = log1pf(__expf(-fabsf(z))) + fmaxf(z, 0.0f);
            spec_S += spec_sp[i];
        }}
        const float spec_invS = 1.0f / fmaxf(spec_S, 1e-8f);

        // Save spectrum logits (32) + scatter flux y (1) + direct flux y (1)
        // for the backward Jacobian (per-thread, 34 floats). The two flux
        // slots are consumed only by the SoftClip backward; the hard-clamp
        // backward ignores them but the stash cost is negligible.
        if (fwd_ctx) {{
            float* my_act = (float*)(fwd_ctx + WARP_SIZE * {ACT_O}) + ::tcnn::lane_id() * 33;
            TCNN_PRAGMA_UNROLL
            for (int i = 0; i < 32; i++) my_act[i] = spec_z[i];
            my_act[32] = flu_y_s;
        }}

        ::tcnn::tvec<::tcnn::network_precision_t, 33> out;
        out[0] = (::tcnn::network_precision_t)flu_y_s;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < 32; i++) out[1 + i] = (::tcnn::network_precision_t)(spec_sp[i] * spec_invS);
        return out;
    )",
        fmt::arg("FN_NAME",  fn_name),
        fmt::arg("D",        this->d_model),
        fmt::arg("TWO_D",    2u * this->d_model),
        fmt::arg("LOC_O",    loc_off),
        fmt::arg("FILM1_O",  film1_off),
        fmt::arg("BLOCK_O",  block_off),
        fmt::arg("POST_O",   post_off),
        fmt::arg("FILM2_O",  film2_off),
        fmt::arg("SPEC_O",   spec_off),
        fmt::arg("ACT_O",    act_off),
        fmt::arg("FLU_O",    flu_off),
        fmt::arg("LOC_P",    loc_p),
        fmt::arg("FILM1_P",  film1_p),
        fmt::arg("BLOCK_P",  block_p),
        fmt::arg("POST_P",   post_p),
        fmt::arg("FILM2_P",  film2_p),
        fmt::arg("SPEC_P",   spec_p),
        fmt::arg("FLU_P",    flu_p),
        // Spectrum logit clamp: pure divergence guard for the Softplus path.
        // Hardtanh(0,1) on flux already bounds itself, no clamp needed there.
        fmt::arg("CLIP",        "30.0f"),
        fmt::arg("FLUX_ACTIVATION_FWD", (this->flux_activation == 1)
            ? std::string(
                "flu_y_s = 0.5f * (tanhf((float)flu[0]) + 1.0f);")
            : fmt::format(
                "flu_y_s = fminf({:.6f}f, fmaxf({:.6f}f, (float)flu[0]   + {:.6f}f));",
                this->flux_clamp_max,        this->flux_clamp_min,        this->flux_offset))
    );

    return fmt::format("{}{}", preamble, this->generate_device_function_from_body(fn_name, body));
}

std::string BaseRadiationPredictionModel::generate_backward_device_function(const std::string& fn_name, uint32_t n_threads) const {
    // Mirror the forward layout: same per-warp fwd_ctx offsets, same parameter
    // offsets, just walked in reverse. Each sub-module already implements its
    // own JIT-fused backward; we chain them and stitch dL/dx back together.
    const uint32_t loc_ctx    = ::tcnn::next_multiple<uint32_t>(this->location_encoding->device_function_fwd_ctx_bytes(),     NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t film1_ctx  = ::tcnn::next_multiple<uint32_t>(this->beam_conditioner1->device_function_fwd_ctx_bytes(),     NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t block_ctx  = ::tcnn::next_multiple<uint32_t>(this->mlp_block->device_function_fwd_ctx_bytes(),             NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t post_ctx   = ::tcnn::next_multiple<uint32_t>(this->mlp_post->device_function_fwd_ctx_bytes(),               NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t film2_ctx  = ::tcnn::next_multiple<uint32_t>(this->beam_conditioner2->device_function_fwd_ctx_bytes(),     NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t spec_ctx   = ::tcnn::next_multiple<uint32_t>(this->mlp_spectrum_decode->device_function_fwd_ctx_bytes(),   NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t flu_ctx    = ::tcnn::next_multiple<uint32_t>(this->flux_projector->device_function_fwd_ctx_bytes(),        NEXT_MULTIPLE_FOR_TYPE);

    const uint32_t loc_off    = 0;
    const uint32_t film1_off  = loc_off    + loc_ctx;
    const uint32_t block_off  = film1_off  + film1_ctx;
    const uint32_t post_off   = block_off  + block_ctx;
    const uint32_t film2_off  = post_off   + post_ctx;
    const uint32_t spec_off   = film2_off  + film2_ctx;
    const uint32_t flu_off    = spec_off   + spec_ctx;
    // Same activation-ctx slab the forward writes (32 spec logits + 1 flux y per thread).
    const uint32_t act_off    = flu_off    + flu_ctx;

    const uint32_t loc_params    = ::tcnn::next_multiple<uint32_t>(this->location_encoding->n_params(),       NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t film1_params  = ::tcnn::next_multiple<uint32_t>(this->beam_conditioner1->n_params(),       NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t block_params  = ::tcnn::next_multiple<uint32_t>(this->mlp_block->n_params(),               NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t post_params   = ::tcnn::next_multiple<uint32_t>(this->mlp_post->n_params(),                NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t film2_params  = ::tcnn::next_multiple<uint32_t>(this->beam_conditioner2->n_params(),       NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t spec_params   = ::tcnn::next_multiple<uint32_t>(this->mlp_spectrum_decode->n_params(),     NEXT_MULTIPLE_FOR_TYPE);
    const uint32_t flu_params    = ::tcnn::next_multiple<uint32_t>(this->flux_projector->n_params(),          NEXT_MULTIPLE_FOR_TYPE);

    const uint32_t loc_p    = 0;
    const uint32_t film1_p  = loc_p    + loc_params;
    const uint32_t block_p  = film1_p   + film1_params;
    const uint32_t post_p   = block_p   + block_params;
    const uint32_t film2_p  = post_p    + post_params;
    const uint32_t spec_p   = film2_p   + film2_params;
    const uint32_t flu_p    = spec_p    + spec_params;

    std::string preamble = fmt::format(R"(
            {LOC_BWD}

            {FILM1_BWD}

            {BLOCK_BWD}

            {POST_BWD}

            {FILM2_BWD}

            {SPEC_BWD}

            {FLU_BWD}

        )",
        fmt::arg("LOC_BWD",    this->location_encoding->generate_backward_device_function(fn_name + "_loc_bwd",         n_threads)),
        fmt::arg("FILM1_BWD",  this->beam_conditioner1->generate_backward_device_function(fn_name + "_film1_bwd",       n_threads)),
        fmt::arg("BLOCK_BWD",  this->mlp_block->generate_backward_device_function(fn_name + "_block_bwd",               n_threads)),
        fmt::arg("POST_BWD",   this->mlp_post->generate_backward_device_function(fn_name + "_post_bwd",                  n_threads)),
        fmt::arg("FILM2_BWD",  this->beam_conditioner2->generate_backward_device_function(fn_name + "_film2_bwd",       n_threads)),
        fmt::arg("SPEC_BWD",   this->mlp_spectrum_decode->generate_backward_device_function(fn_name + "_spec_bwd",      n_threads)),
        fmt::arg("FLU_BWD",    this->flux_projector->generate_backward_device_function(fn_name + "_flu_bwd",            n_threads))
    );

    std::string body = fmt::format(R"(
        // Spectrum (32) + scatter flux y (1) + direct flux y (1) per thread
        // — same layout as the forward.
        const float* my_act = (const float*)(fwd_ctx + WARP_SIZE * {ACT_O}) + ::tcnn::lane_id() * 33;
        float spec_z[32];
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < 32; i++) spec_z[i] = my_act[i];

        // Per-head flux activation backward.
        //   clamp:    dL/dz = dL/dy (gradient-conserving identity)
        //   softclip: dL/dz = dL/dy * 2 * y * (1 - y), where y was stashed.
        // dL_dy[0] = scatter loss grad,  dL_dy[1] = direct loss grad.
        {FLUX_ACTIVATION_BWD}

        // y_i = softplus(z_i) / S,  S = Σ softplus(z_j).
        // dy_i/dz_k = sigma(z_k)/S * (delta_ik - y_i),  sigma = softplus'.
        // => dL/dz_k = sigma(z_k)/S * (dL/dy_k - Σ_i dL/dy_i * y_i).
        // Spectrum offset is 2 — out[0]=scatter, out[1]=direct, out[2..34) = spectrum.
        float spec_sp[32], spec_sigma[32];
        float spec_S = 0.0f;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < 32; i++) {{
            const float z = spec_z[i];
            spec_sp[i]    = log1pf(__expf(-fabsf(z))) + fmaxf(z, 0.0f);
            spec_sigma[i] = 1.0f / (1.0f + __expf(-z));
            spec_S += spec_sp[i];
        }}
        const float spec_invS = 1.0f / fmaxf(spec_S, 1e-8f);
        float spec_y[32];
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < 32; i++) spec_y[i] = spec_sp[i] * spec_invS;
        float dot = 0.0f;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < 32; i++) dot += (float)dL_dy[1 + i] * spec_y[i];

        // Both flux projectors emit 16 padded channels; only channel 0 carries gradient.
        ::tcnn::tvec<::tcnn::network_precision_t, 16> dL_dflu_s;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < 16; i++) dL_dflu_s[i] = (::tcnn::network_precision_t)0.f;
        dL_dflu_s[0] = (::tcnn::network_precision_t)dL_dz_flu_s;

        ::tcnn::tvec<::tcnn::network_precision_t, 32> dL_dspec;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < 32; i++) {{
            const float g = spec_sigma[i] * spec_invS * ((float)dL_dy[1 + i] - dot);
            dL_dspec[i] = (::tcnn::network_precision_t)g;
        }}

        // Backward through the three decoder heads; all three feed x3. The
        // gradients sum because x3 is the common ancestor of every head.
        ::tcnn::tvec<::tcnn::network_precision_t, {D}> dL_dx3_flu_s;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) dL_dx3_flu_s[i] = (::tcnn::network_precision_t)0.f;
        {FN_NAME}_flu_bwd(dL_dflu_s, params + {FLU_P}, fwd_ctx + WARP_SIZE * {FLU_O}, dL_dparams ? dL_dparams + {FLU_P} : nullptr, &dL_dx3_flu_s);

        ::tcnn::tvec<::tcnn::network_precision_t, {D}> dL_dx3_spec;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) dL_dx3_spec[i] = (::tcnn::network_precision_t)0.f;
        {FN_NAME}_spec_bwd(dL_dspec, params + {SPEC_P}, fwd_ctx + WARP_SIZE * {SPEC_O}, dL_dparams ? dL_dparams + {SPEC_P} : nullptr, &dL_dx3_spec);

        ::tcnn::tvec<::tcnn::network_precision_t, {D}> dL_dx3;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) dL_dx3[i] = dL_dx3_flu_s[i] + dL_dx3_spec[i];

        // FiLM2 backward returns the gradient over its full (feature, condition)
        // input — first D entries are dL/dx2_merged, last D are dL/dbeam_slice.
        // (x2_merged is the concat-projector output, not the raw mlp_block
        // output; the projector backward below routes it correctly.)
        ::tcnn::tvec<float, {TWO_D}> dL_dfilm2_in;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {TWO_D}; i++) dL_dfilm2_in[i] = 0.f;
        {FN_NAME}_film2_bwd(dL_dx3, params + {FILM2_P}, fwd_ctx + WARP_SIZE * {FILM2_O}, dL_dparams ? dL_dparams + {FILM2_P} : nullptr, &dL_dfilm2_in);

        // dL/dx2_merged (half) from FiLM2's first D outputs.
        ::tcnn::tvec<::tcnn::network_precision_t, {D}> dL_dx2_merged;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) dL_dx2_merged[i] = (::tcnn::network_precision_t)dL_dfilm2_in[i];

        // mlp_post backward (no concat-skip): input is d_model (mlp_block's
        // output), so the gradient is dL/dx2 (d_model) directly.
        ::tcnn::tvec<::tcnn::network_precision_t, {D}> dL_dx2;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) dL_dx2[i] = (::tcnn::network_precision_t)0.f;
        {FN_NAME}_post_bwd(dL_dx2_merged, params + {POST_P}, fwd_ctx + WARP_SIZE * {POST_O}, dL_dparams ? dL_dparams + {POST_P} : nullptr, &dL_dx2);

        // mlp_block (half->half): no precision cast needed on its output gradient.
        ::tcnn::tvec<::tcnn::network_precision_t, {D}> dL_dx1;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) dL_dx1[i] = (::tcnn::network_precision_t)0.f;
        {FN_NAME}_block_bwd(dL_dx2, params + {BLOCK_P}, fwd_ctx + WARP_SIZE * {BLOCK_O}, dL_dparams ? dL_dparams + {BLOCK_P} : nullptr, &dL_dx1);

        // Residual skip is now on loc_enc (not x1): x3 = FiLM2(...) + loc_enc.
        // The direct dL_dx3 contribution therefore flows to dL_dlocenc below,
        // not to dL_dx1. (Compared to the previous design with the residual
        // on x1, this gives the encoder a third gradient path from the
        // decoders, which is closer in spirit to the NeRF identity-skip.)

        // FiLM1 backward — same layout as FiLM2.
        ::tcnn::tvec<float, {TWO_D}> dL_dfilm1_in;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {TWO_D}; i++) dL_dfilm1_in[i] = 0.f;
        {FN_NAME}_film1_bwd(dL_dx1, params + {FILM1_P}, fwd_ctx + WARP_SIZE * {FILM1_O}, dL_dparams ? dL_dparams + {FILM1_P} : nullptr, &dL_dfilm1_in);

        // loc_enc feeds TWO downstream consumers (concat-skip removed), so its
        // gradient is the sum of two paths:
        //   (1) dL_dfilm1_in[0..D)  — through FILM1 → mlp_block → mlp_post → FILM2
        //   (2) dL_dx3              — direct additive terminal residual into x3
        ::tcnn::tvec<::tcnn::network_precision_t, {D}> dL_dlocenc;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < {D}; i++) dL_dlocenc[i] = (::tcnn::network_precision_t)(
            (float)dL_dfilm1_in[i] + (float)dL_dx3[i]
        );

        ::tcnn::tvec<float, 3> dL_dxyz;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < 3; i++) dL_dxyz[i] = 0.f;
        {FN_NAME}_loc_bwd(dL_dlocenc, params + {LOC_P}, fwd_ctx + WARP_SIZE * {LOC_O}, dL_dparams ? dL_dparams + {LOC_P} : nullptr, &dL_dxyz);

        if (dL_dx) {{
            TCNN_PRAGMA_UNROLL
            for (int i = 0; i < 3; i++) (*dL_dx)[i] = dL_dxyz[i];
            // The beam slice was fed into both FiLM1 and FiLM2; sum both grads.
            TCNN_PRAGMA_UNROLL
            for (int i = 0; i < {D}; i++) (*dL_dx)[3 + i] = dL_dfilm1_in[{D} + i] + dL_dfilm2_in[{D} + i];
        }}
    )",
        fmt::arg("FN_NAME",  fn_name),
        fmt::arg("D",        this->d_model),
        fmt::arg("TWO_D",    2u * this->d_model),
        fmt::arg("LOC_O",    loc_off),
        fmt::arg("FILM1_O",  film1_off),
        fmt::arg("BLOCK_O",  block_off),
        fmt::arg("POST_O",   post_off),
        fmt::arg("FILM2_O",  film2_off),
        fmt::arg("SPEC_O",   spec_off),
        fmt::arg("ACT_O",    act_off),
        fmt::arg("FLU_O",    flu_off),
        fmt::arg("LOC_P",    loc_p),
        fmt::arg("FILM1_P",  film1_p),
        fmt::arg("BLOCK_P",  block_p),
        fmt::arg("POST_P",   post_p),
        fmt::arg("FILM2_P",  film2_p),
        fmt::arg("SPEC_P",   spec_p),
        fmt::arg("FLU_P",    flu_p),
        fmt::arg("FLUX_ACTIVATION_BWD", (this->flux_activation == 1)
            ? std::string(
                "const float fy_s = my_act[32]; "
                "const float dL_dz_flu_s = (float)dL_dy[0] * 2.0f * fy_s * (1.0f - fy_s);")
            : std::string(
                "const float dL_dz_flu_s = (float)dL_dy[0];"))
    );

    return fmt::format("{}{}", preamble, this->generate_backward_device_function_from_body(fn_name, body));
}
