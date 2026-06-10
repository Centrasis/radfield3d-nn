#pragma once
#include <memory>
#include <vector>
#include <string>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/object.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/network.h>

#include <tiny-cuda-nn/network_with_input_encoding.h>


namespace tcnn {
    class Context;

    class CudaRtcKernel;
};

namespace rfnn::tcnn {
    // Selects the underlying coordinate encoder. Frequency (Fourier) is the
    // historic default and now ALSO appends the raw xyz into the last 3
    // channels of the d_model-wide output — this mirrors the Python
    // SinusoidalFrequencyEncoding(append_input=True) which feeds the trunk
    // a low-frequency signal even after the per-axis Fourier features.
    // HashGrid uses a multi-resolution hash table (Instant-NGP) and does
    // NOT append xyz — the lookup is already an explicit function of
    // position, so an extra raw copy is redundant.
    enum class LocationEncodingKind : int {
        Frequency = 0,
        HashGrid  = 1,
    };

    class LocationEncoding : public ::tcnn::DifferentiableObject<float, ::tcnn::network_precision_t, ::tcnn::network_precision_t> {
    private:
        struct Context : ::tcnn::Context {
            std::unique_ptr<::tcnn::Context> mlp_ctx;
            std::unique_ptr<::tcnn::Context> mlp_encoding;
            std::shared_ptr<const ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>> encoded_locations;
            std::shared_ptr<const ::tcnn::GPUMatrixDynamic<float>> input_locations;
            std::shared_ptr<::tcnn::GPUMatrix<float>> dy_dx;
        };

    public:
        // For HashGrid, `frequencies` is reinterpreted as `n_levels`
        // (Instant-NGP terminology); other hash-grid hyperparameters use the
        // defaults from the Müller 2022 paper (2 features/level,
        // base_resolution=16, hashmap 2^19, per_level_scale derived from a
        // typical voxel-grid extent).
        LocationEncoding(uint32_t frequencies, uint32_t encoded_dims,
                          LocationEncodingKind kind = LocationEncodingKind::Frequency);
        virtual ~LocationEncoding() {};

        std::shared_ptr<::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>> encode_locations(const ::tcnn::GPUMatrixDynamic<float>& input);


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

        uint32_t backward_device_function_shmem_bytes(uint32_t n_threads, ::tcnn::GradientMode param_gradients_mode) const override {
            if (this->kind == LocationEncodingKind::HashGrid) {
                return this->mlp_encoding_block->backward_device_function_shmem_bytes(n_threads, param_gradients_mode);
            }
            uint32_t b = this->freq_mlp->backward_device_function_shmem_bytes(n_threads, param_gradients_mode);
            b = std::max(b, this->freq_encoding->backward_device_function_shmem_bytes(n_threads, param_gradients_mode));
            return b;
        }

        nlohmann::json hyperparams() const override {
            if (this->kind == LocationEncodingKind::HashGrid) {
                return {
                    {"otype", "LocationEncoding"},
                    {"kind", "HashGrid"},
                    {"mlp_encoding_block", this->mlp_encoding_block->hyperparams()},
                };
            }
            return {
                {"otype", "LocationEncoding"},
                {"kind", "Frequency"},
                {"freq_encoding", this->freq_encoding->hyperparams()},
                {"freq_mlp",      this->freq_mlp->hyperparams()},
                {"freq_real_out_width", this->freq_real_out_width},
                {"padded_loc_enc_dims", this->padded_loc_enc_dims},
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
            return 3;
        }

        uint32_t padded_output_width() const override {
            return (this->kind == LocationEncodingKind::HashGrid)
                ? this->mlp_encoding_block->padded_output_width()
                : this->freq_mlp->padded_output_width();
        }

        uint32_t output_width() const override {
            return this->encoded_dims;
        }

        uint32_t required_input_alignment() const override {
            return 1;
        }

        void convert_params_to_jit_layout(cudaStream_t stream, bool use_inference_params) override {
            if (this->kind == LocationEncodingKind::HashGrid) {
                this->mlp_encoding_block->convert_params_to_jit_layout(stream, use_inference_params);
            } else {
                this->freq_mlp->convert_params_to_jit_layout(stream, use_inference_params);
            }
        }

        void convert_params_from_jit_layout(cudaStream_t stream, bool use_inference_params) override {
            if (this->kind == LocationEncodingKind::HashGrid) {
                this->mlp_encoding_block->convert_params_from_jit_layout(stream, use_inference_params);
            } else {
                this->freq_mlp->convert_params_from_jit_layout(stream, use_inference_params);
            }
        }

    protected:
        bool is_training_mode = true;
        const uint32_t encoded_dims;
        const uint32_t frequencies;
        const LocationEncodingKind kind;

        // Frequency path: split encoding+MLP so we can inject raw xyz into 3
        // of the encoding's zero/one-filled alignment pads — matches Python
        // SinusoidalFrequencyEncoding(append_input=True), where the Fourier
        // features and raw xyz BOTH feed into the next Linear layer (vs. the
        // overwrite-the-MLP-output variant, where xyz bypasses the MLP).
        //   freq_real_out_width = 2 * 3 * F   (real Fourier outputs)
        //   padded_loc_enc_dims = next_multiple(freq_real_out_width + 3, 16)
        //                         — guarantees >= 3 spare slots for xyz.
        //   xyz_inject_offset   = freq_real_out_width
        //                         — the first slot where xyz lands.
        uint32_t padded_loc_enc_dims = 0;
        uint32_t freq_real_out_width = 0;
        uint32_t xyz_inject_offset = 0;

        // HashGrid path keeps the fused NetworkWithInputEncoding (hash table
        // is the encoder's own parameter, no need to intercept between
        // encoder and MLP). Frequency path uses the split pair.
        std::unique_ptr<::tcnn::NetworkWithInputEncoding<::tcnn::network_precision_t>> mlp_encoding_block;
        std::unique_ptr<::tcnn::Encoding<::tcnn::network_precision_t>>                 freq_encoding;
        std::unique_ptr<::tcnn::Network<::tcnn::network_precision_t, ::tcnn::network_precision_t>> freq_mlp;
    };
};