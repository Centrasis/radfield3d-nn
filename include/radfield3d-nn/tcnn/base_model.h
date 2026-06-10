#pragma once
#include <memory>
#include <vector>
#include <tiny-cuda-nn/common.h>
#include "radfield3d-nn/tcnn/layers/film.h"
#include "radfield3d-nn/tcnn/layers/beam_fusion.h"
#include "radfield3d-nn/tcnn/layers/layer_norm.h"
#include "radfield3d-nn/tcnn/encodings/location_encoding.h"
#include "radfield3d-nn/tcnn/encodings/global_parameters.h"


namespace tcnn {
    template<typename T>
    class Encoding;

    template<typename T, typename B>
    class Network;

    template<typename T>
    class GPUMatrixDynamic;

    template<typename T>
    class GPUMemory;

    class Context;
};


namespace rfnn::tcnn {
    // Main BaseRadiationPredictionModel forward. Input layout per sample:
    //   input[0..3)            xyz position (float)
    //   input[3..3+d_model)    pre-encoded beam (float; produced by BeamEncoder, then cast back from half)
    // Output layout (33 channels) — single flux head + joined spectrum:
    //   output[0]              flux       (per-volume-relative joined flux)
    //   output[1..33)          spectrum (32 bins, joined per-voxel histogram)
    //
    // A single bias-less flux projector + clamp predicts the joined relative
    // flux; the spectrum head emits the joined per-voxel histogram.
    class BaseRadiationPredictionModel : public ::tcnn::DifferentiableObject<float, ::tcnn::network_precision_t, ::tcnn::network_precision_t> {
    public:
        // flux_activation: 0 = hard clamp ([0,1], gradient-conserving) — the
        //                  historic default; suffers from "predict 0 forever"
        //                  lock-in once z is pushed below the lower clamp.
        //                 1 = SoftClip 0.5*(tanh(z)+1); smooth in (0,1),
        //                  gradient nonzero everywhere on R so the lock-in
        //                  cannot occur. NOT related to the SoftCLIP paper.
        // location_encoding_kind selects Fourier-features vs InstantNGP hash-
        // grid for the xyz path inside LocationEncoding. Frequency is the
        // historic default (and appends raw xyz to the last 3 channels of the
        // encoding output); HashGrid is the multi-resolution table.
        // flux_clamp_min / flux_clamp_max + flux_offset apply to the flux
        // head. Softclip (flux_activation == 1) ignores the clamp values — it
        // only produces [0, 1] — and the Python facade rejects non-default
        // clamp ranges paired with softclip.
        BaseRadiationPredictionModel(uint32_t d_model = 64, uint32_t location_encoding_dim = 12, float flux_offset = 0.5f, int flux_activation = 0,
                                     LocationEncodingKind location_encoding_kind = LocationEncodingKind::Frequency,
                                     float flux_clamp_min = 0.0f, float flux_clamp_max = 1.0f,
                                     uint32_t trunk_hidden_layers = 1u,
                                     BeamFusionKind beam_fusion = BeamFusionKind::FiLM);
        virtual ~BaseRadiationPredictionModel();

        // interface for tcnn
        virtual size_t n_params() const override;
        void set_params_impl(::tcnn::network_precision_t* params, ::tcnn::network_precision_t* inference_params, ::tcnn::network_precision_t* gradients) override;
        std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override;
        void initialize_params(::tcnn::pcg32& rnd, float* params_full_precision, float scale = 1) override;

        virtual std::string generate_device_function(const std::string& name) const override;
        virtual std::string generate_backward_device_function(const std::string& name, uint32_t n_threads) const override;

        std::unique_ptr<::tcnn::Context> forward_impl(cudaStream_t stream, const ::tcnn::GPUMatrixDynamic<float>& input, ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override { throw std::runtime_error("Use JIT!"); };
        void backward_impl(cudaStream_t stream, const ::tcnn::Context& ctx, const ::tcnn::GPUMatrixDynamic<float>& input, const ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>& output, const ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>& dL_doutput, ::tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr, bool use_inference_params = false, ::tcnn::GradientMode param_gradients_mode = ::tcnn::GradientMode::Overwrite) override { throw std::runtime_error("Use JIT!"); };

        uint32_t device_function_fwd_ctx_bytes() const override;

        bool device_function_fwd_ctx_aligned_per_element() const override {
            return false;
        }

        uint32_t backward_device_function_shmem_bytes(uint32_t n_threads, ::tcnn::GradientMode param_gradients_mode) const override;

        nlohmann::json hyperparams() const override {
            return {
                {"otype", "BaseRadiationPredictionModel"},
                {"d_model", this->d_model},
                {"flux_offset", this->flux_offset},
                {"flux_activation", this->flux_activation},
                {"flux_clamp_min", this->flux_clamp_min},
                {"flux_clamp_max", this->flux_clamp_max},
                {"location_encoding_kind", static_cast<int>(this->location_encoding_kind)},
                {"trunk_hidden_layers", this->trunk_hidden_layers},
                {"beam_fusion", static_cast<int>(this->beam_fusion_kind)}
            };
        }

        void inference_mixed_precision_impl(
            cudaStream_t stream,
            const ::tcnn::GPUMatrixDynamic<float>& input,
            ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>& output,
            bool use_inference_params = true
        ) override {
            this->forward(stream, input, &output, use_inference_params, false);
        }

        uint32_t input_width() const override {
            return 3u + this->d_model;
        }

        uint32_t padded_output_width() const override {
            // 1 flux head + 32 spectrum bins.
            return 33u;
        }

        uint32_t output_width() const override {
            return 33u;
        }

        uint32_t required_input_alignment() const override {
            return 1u;
        }

        void convert_params_to_jit_layout(cudaStream_t stream, bool use_inference_params) override;
        void convert_params_from_jit_layout(cudaStream_t stream, bool use_inference_params) override;

        uint32_t d_model_dim() const { return d_model; }

        // Element offset within the flat params blob where the two output heads
        // (mlp_spectrum_decode + flux_projector) begin. Everything BEFORE this
        // offset is the shared trunk (location_encoding, FiLM1, mlp_block,
        // mlp_post, FiLM2). DB-MTL uses it to take per-task gradient norms over
        // the trunk only (excluding the task-specific heads), mirroring the
        // pure-Python `output_head_markers` exclusion.
        size_t output_head_param_offset() const;

    private:
        uint32_t d_model;
        size_t total_parameter_count = 0;
        // Static, configurable additive offset applied to the raw flux logit
        // before the [0,1] hard clamp in the fused kernel. Replaces both the
        // previous Sigmoid activation (which saturates) and the previously
        // attempted learned `rfnn::tcnn::Bias` (which is now removed entirely —
        // bias-less tcnn matmul + fixed offset is the new contract). Default
        // 0.5 = codomain midpoint; setting to ~0.58 centers on the DS03
        // joined-flux normalized data mean.
        float flux_offset;
        // 0 = hard clamp, 1 = SoftClip. See ctor doc.
        int flux_activation;
        // Codomain of the hard-clamp activation for the scatter head;
        // ignored by softclip. See ctor doc.
        float flux_clamp_min;
        float flux_clamp_max;
        // Frequency (Fourier+append xyz) or HashGrid (InstantNGP). See
        // LocationEncoding for semantic differences.
        LocationEncodingKind location_encoding_kind;
        // n_hidden_layers of the two trunk MLPs (mlp_block, mlp_post). Default
        // 1 (each = 2 weight matrices). Set to 0 to make the trunk shallow
        // (each MLP = a single input→output Linear) — useful with the HashGrid
        // location encoding, which already supplies high-frequency features so
        // a deep trunk is wasted (and is harder to train in fp16). With 0 the
        // flux path is 4 Linears: mlp_block(1) + mlp_post(1) + flux_projector(2).
        uint32_t trunk_hidden_layers;
        // Which fusion the two beam conditioners use (FiLM affine vs bounded
        // GatedFusion). Stored for hyperparams() round-tripping.
        BeamFusionKind beam_fusion_kind;

        std::unique_ptr<rfnn::tcnn::LocationEncoding> location_encoding;
        std::unique_ptr<rfnn::tcnn::BeamFusionModule> beam_conditioner1;
        // mlp_block — the pre-concat trunk MLP: takes the FiLM1 output
        // (d_model wide), produces d_model. n_hidden_layers=2 (3 weight
        // matrices). Corresponds to NeRF's pre-skip 4-layer block in spirit
        // (we run 3 layers + FiLM modulations rather than 4 raw Linears).
        std::unique_ptr<::tcnn::Network<::tcnn::network_precision_t, ::tcnn::network_precision_t>> mlp_block;
        // mlp_post — the post-concat trunk MLP: takes
        // concat(mlp_block_out, loc_enc) of width 2*d_model and produces
        // d_model. n_hidden_layers=1 (2 weight matrices). This is the
        // proper NeRF-style mid-trunk skip merger — a real MLP that ingests
        // the wider input through a JIT-fused matmul (no separate 0-hidden
        // projector). The next FiLM and additive residual sit between this
        // MLP and the decoder heads.
        std::unique_ptr<::tcnn::Network<::tcnn::network_precision_t, ::tcnn::network_precision_t>> mlp_post;
        std::unique_ptr<rfnn::tcnn::BeamFusionModule> beam_conditioner2;
        std::unique_ptr<::tcnn::Network<::tcnn::network_precision_t, ::tcnn::network_precision_t>> mlp_spectrum_decode;
        // Single bias-less flux projector (CutlassMLP); its output is wrapped
        // by the clamp/offset activation in the JIT body to emit the joined
        // per-volume-relative flux.
        std::unique_ptr<::tcnn::Network<::tcnn::network_precision_t, ::tcnn::network_precision_t>> flux_projector;

        uint32_t NEXT_MULTIPLE_FOR_TYPE;
    };
};
