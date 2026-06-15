#pragma once
#include <memory>
#include <string>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/object.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network.h>


namespace rfnn::tcnn {
    // Bounded gated feature fusion (an alternative to FiLM).
    //
    // Treats its input as the concatenation [feature, condition] (per sample,
    // along the channel axis), exactly like FiLM. An internal MLP maps
    // `condition` -> (gate_logits, candidate). A hardsigmoid bounds the gate to
    // [0, 1] and the output is the convex blend
    //
    //     out = g * feature + (1 - g) * candidate,   g = hardsigmoid(gate_logit)
    //
    // optionally followed by ReLU/SiLU. Unlike FiLM's unbounded affine
    // (1 + gamma) * feature + beta, the gate is bounded (hardsigmoid is
    // piecewise-linear, so fp16-stable, no exp), letting the condition SELECT
    // or REPLACE feature channels rather than only rescale them. This is the
    // intended pairing with the discontinuous HashGrid location encoding, where
    // a global affine modulation conditions the features poorly.
    //
    // Same input/output widths and internal param layout as FiLM, so it is a
    // drop-in beam conditioner inside BaseRadiationPredictionModel.
    class GatedFusion : public ::tcnn::DifferentiableObject<float, ::tcnn::network_precision_t, ::tcnn::network_precision_t> {
    public:
        GatedFusion(uint32_t feature_channels, uint32_t condition_channels, const std::string& non_linearity = "None");
        virtual ~GatedFusion() {};

        // Differentiable Interface from TCNN
        void set_params_impl(::tcnn::network_precision_t* params, ::tcnn::network_precision_t* inference_params, ::tcnn::network_precision_t* gradients) override;
        size_t n_params() const override;
        std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override;
        void initialize_params(::tcnn::pcg32& rnd, float* params_full_precision, float scale = 1) override;

        std::string generate_device_function(const std::string& name) const override;
        std::string generate_backward_device_function(const std::string& name, uint32_t n_threads) const override;

        std::unique_ptr<::tcnn::Context> forward_impl(cudaStream_t stream, const ::tcnn::GPUMatrixDynamic<float>& input, ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override { throw std::runtime_error("Use JIT!"); };
        void backward_impl(cudaStream_t stream, const ::tcnn::Context& ctx, const ::tcnn::GPUMatrixDynamic<float>& input, const ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>& output, const ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>& dL_doutput, ::tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr, bool use_inference_params = false, ::tcnn::GradientMode param_gradients_mode = ::tcnn::GradientMode::Overwrite) override { throw std::runtime_error("Use JIT!"); };

        uint32_t device_function_fwd_ctx_bytes() const override;

        bool device_function_fwd_ctx_aligned_per_element() const override {
            return false;
        }

        uint32_t backward_device_function_shmem_bytes(uint32_t n_threads, ::tcnn::GradientMode param_gradients_mode) const override;

        nlohmann::json hyperparams() const override;

        void inference_mixed_precision_impl(
            cudaStream_t stream,
            const ::tcnn::GPUMatrixDynamic<float>& input,
            ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>& output,
            bool use_inference_params = true
        ) override {
            this->forward(stream, input, &output, use_inference_params, false);
        }

        uint32_t input_width() const override {
            return this->feature_channels + this->condition_channels;
        }

        uint32_t padded_output_width() const override {
            return this->feature_channels;
        }

        uint32_t output_width() const override {
            return this->feature_channels;
        }

        uint32_t required_input_alignment() const override {
            return this->NEXT_MULTIPLE_FOR_TYPE;
        }

        void convert_params_to_jit_layout(cudaStream_t stream, bool use_inference_params) override;
        void convert_params_from_jit_layout(cudaStream_t stream, bool use_inference_params) override;

    protected:
        uint32_t feature_channels;
        uint32_t condition_channels;
        uint32_t condition_padded;
        uint32_t NEXT_MULTIPLE_FOR_TYPE;
        bool use_relu;
        bool use_silu;
        std::unique_ptr<::tcnn::Network<::tcnn::network_precision_t, ::tcnn::network_precision_t>> mlp_gate_candidate;
    };
};
