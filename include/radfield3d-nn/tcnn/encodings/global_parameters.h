#pragma once
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/object.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/network.h>

namespace rfnn::tcnn {
    class ParameterSetEncoding : public ::tcnn::DifferentiableObject<float, ::tcnn::network_precision_t, ::tcnn::network_precision_t> {     
    public:
        typedef struct ParameterSet {
            std::string encoding;
            unsigned long dimensions;
            unsigned long feature_dimensions;

            ParameterSet(std::string encoding, unsigned long dimensions, unsigned long feature_dimensions = 0)
                : dimensions(dimensions), feature_dimensions(feature_dimensions), encoding(encoding)
            {}
        } ParameterSet;
    using ParameterSetEncodings_t = std::vector<std::pair<ParameterSet, std::unique_ptr<::tcnn::Encoding<::tcnn::network_precision_t>>>>;

    protected:
        ParameterSetEncodings_t parameter_sets;
        std::unique_ptr<::tcnn::Network<::tcnn::network_precision_t, ::tcnn::network_precision_t>> encoding_mlp;
        uint32_t encoded_dims;
        uint32_t NEXT_MULTIPLE_FOR_TYPE;
        uint32_t encoding_mlp_input_width;

    public:
        ParameterSetEncoding(const std::vector<ParameterSet>& parameter_sets, uint32_t encoded_dims);
        virtual ~ParameterSetEncoding() {};

        // Differentiable Interface from TCNN
        void set_params_impl(::tcnn::network_precision_t* params, ::tcnn::network_precision_t* inference_params, ::tcnn::network_precision_t* gradients) override;
        virtual size_t n_params() const override;
        std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override;
        void initialize_params(::tcnn::pcg32& rnd, float* params_full_precision, float scale = 1) override;
        virtual std::string generate_device_function(const std::string& name) const override;
        virtual std::string generate_backward_device_function(const std::string& name, uint32_t n_threads) const override;
        std::unique_ptr<::tcnn::Context> forward_impl(cudaStream_t stream, const ::tcnn::GPUMatrixDynamic<float>& input, ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override { throw std::runtime_error("Use JIT!"); };
        void backward_impl(cudaStream_t stream, const ::tcnn::Context& ctx, const ::tcnn::GPUMatrixDynamic<float>& input, const ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>& output, const ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>& dL_doutput,	::tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr, bool use_inference_params = false, ::tcnn::GradientMode param_gradients_mode = ::tcnn::GradientMode::Overwrite) override { throw std::runtime_error("Use JIT!"); }; 
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

        uint32_t input_width() const override;
        uint32_t padded_output_width() const override;
        uint32_t output_width() const override {
            return this->encoded_dims;
        }
        uint32_t required_input_alignment() const override {
            return this->NEXT_MULTIPLE_FOR_TYPE;
        }

        void convert_params_to_jit_layout(cudaStream_t stream, bool use_inference_params) override;
        void convert_params_from_jit_layout(cudaStream_t stream, bool use_inference_params) override;
    };
};
