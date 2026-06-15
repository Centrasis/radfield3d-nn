#pragma once
#include <memory>
#include <vector>
#include <tiny-cuda-nn/common.h>
#include "radfield3d-nn/tcnn/layers/layer_norm.h"
#include "radfield3d-nn/tcnn/encodings/location_encoding.h"
#include "radfield3d-nn/tcnn/encodings/global_parameters.h"


namespace rfnn::tcnn {
    // Distance-less beam-side encoder for SPERFNetCPP (mirrors the pure-Python
    // SPERFNet's beam encoder): takes only (direction, spectrum) — no beam
    // distance — and produces the d_model-wide feature vector that downstream
    // FiLM layers in BaseRadiationPredictionModel consume. Pipeline:
    //   MLP1 → LN2 → MLP2 → ParameterSetEncoding(SH(3) + None(spec))
    //                                                   → final LayerNorm.
    // Identical to PBRFBeamEncoder except for the missing distance one-blob
    // input — for fixed-distance datasets where the beam distance is constant
    // and would only add an uninformative input.
    class SPERFBeamEncoder : public ::tcnn::DifferentiableObject<float, ::tcnn::network_precision_t, ::tcnn::network_precision_t> {
    public:
        SPERFBeamEncoder(uint32_t spectrum_dim, uint32_t d_model);
        virtual ~SPERFBeamEncoder();

        size_t n_params() const override;
        void set_params_impl(::tcnn::network_precision_t* params, ::tcnn::network_precision_t* inference_params, ::tcnn::network_precision_t* gradients) override;
        std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override;
        void initialize_params(::tcnn::pcg32& rnd, float* params_full_precision, float scale = 1) override;

        std::string generate_device_function(const std::string& name) const override;
        std::string generate_backward_device_function(const std::string& name, uint32_t n_threads) const override;

        std::unique_ptr<::tcnn::Context> forward_impl(cudaStream_t /*stream*/, const ::tcnn::GPUMatrixDynamic<float>& /*input*/, ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>* /*output*/ = nullptr, bool /*use_inference_params*/ = false, bool /*prepare_input_gradients*/ = false) override { throw std::runtime_error("Use JIT!"); }
        void backward_impl(cudaStream_t /*stream*/, const ::tcnn::Context& /*ctx*/, const ::tcnn::GPUMatrixDynamic<float>& /*input*/, const ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>& /*output*/, const ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>& /*dL_doutput*/, ::tcnn::GPUMatrixDynamic<float>* /*dL_dinput*/ = nullptr, bool /*use_inference_params*/ = false, ::tcnn::GradientMode /*param_gradients_mode*/ = ::tcnn::GradientMode::Overwrite) override { throw std::runtime_error("Use JIT!"); }

        uint32_t device_function_fwd_ctx_bytes() const override;

        bool device_function_fwd_ctx_aligned_per_element() const override {
            return false;
        }

        uint32_t backward_device_function_shmem_bytes(uint32_t n_threads, ::tcnn::GradientMode param_gradients_mode) const override;

        void inference_mixed_precision_impl(
            cudaStream_t stream,
            const ::tcnn::GPUMatrixDynamic<float>& input,
            ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>& output,
            bool use_inference_params = true
        ) override {
            this->forward(stream, input, &output, use_inference_params, false);
        }

        nlohmann::json hyperparams() const override {
            return {{"otype", "SPERFBeamEncoder"}, {"spectrum_dim", spectrum_dim}, {"d_model", d_model}};
        }

        // Input layout: [direction(3), spectrum(spectrum_dim)] — no distance.
        uint32_t input_width() const override { return 3u + this->spectrum_dim; }
        uint32_t padded_output_width() const override { return this->d_model; }
        uint32_t output_width() const override { return this->d_model; }
        uint32_t required_input_alignment() const override { return 1u; }

        void convert_params_to_jit_layout(cudaStream_t stream, bool use_inference_params) override;
        void convert_params_from_jit_layout(cudaStream_t stream, bool use_inference_params) override;

        uint32_t spectrum_input_dim() const { return spectrum_dim; }
        uint32_t encoded_dim() const { return d_model; }

    protected:
        uint32_t spectrum_dim;
        uint32_t d_model;
        uint32_t spectrum_enc_dim;
        uint32_t spectrum_dim_padded;
        uint32_t NEXT_MULTIPLE_FOR_TYPE;

        // Spectrum sub-MLP: MLP1 → LN → MLP2 (mirrors Python
        // SimpleSpectraEncoder; the previous LN on the raw spectrum was a
        // mistake, see PBRFBeamEncoder comment).
        std::unique_ptr<::tcnn::Network<::tcnn::network_precision_t, ::tcnn::network_precision_t>> spectrum_mlp1;
        std::unique_ptr<rfnn::tcnn::LayerNorm> spectrum_ln2;
        std::unique_ptr<::tcnn::Network<::tcnn::network_precision_t, ::tcnn::network_precision_t>> spectrum_mlp2;

        // Final beam-side encoding: SH(direction) + None(spectrum_encoded) → d_model
        std::unique_ptr<rfnn::tcnn::ParameterSetEncoding> beam_params_encoding;
        // LayerNorm on the d_model-wide ParameterSetEncoding output (mirrors
        // PBRFBeamEncoder; downstream FiLM needs unit-variance input).
        std::unique_ptr<rfnn::tcnn::LayerNorm> beam_params_ln;
    };
}
