#pragma once
//
// CombinedRadiationModel — a beam encoder + a radiation predictor held together
// as one loadable unit. Unlike the bare tcnn models (which do not own their
// parameters — the Python ModuleBridge does), this class **owns** the weight
// buffers for both sub-models, so a model restored by `ModelFactory::load`
// is immediately usable for C++ inference with no Python/torch involvement.
//
#include <cstdint>
#include <memory>
#include <string>
#include <tiny-cuda-nn/gpu_memory.h>
#include <radfield3d-nn/tcnn/base_model.h>
#include <radfield3d-nn/tcnn/encodings/beam_encoder.h>

namespace rfnn::tcnn {

    // Reconstruct a concrete tcnn sub-model from its `hyperparams()` JSON string
    // (keyed on "otype"). Returns the model and reports its n_params via out-arg.
    // Throws std::runtime_error on an unknown otype.
    std::unique_ptr<::tcnn::DifferentiableObject<float, ::tcnn::network_precision_t, ::tcnn::network_precision_t>>
    build_module_from_hparams(const std::string& hparams_json, size_t& n_params_out);

    class CombinedRadiationModel {
    public:
        using Module = ::tcnn::DifferentiableObject<float, ::tcnn::network_precision_t, ::tcnn::network_precision_t>;

        // Takes ownership of both sub-models. `*_weights_device` are copied into
        // this object's own GPU buffers and wired into the sub-models via
        // set_params(), so the caller's buffers can be released afterwards.
        CombinedRadiationModel(std::unique_ptr<Module> encoder, size_t encoder_n_params,
                               const ::tcnn::network_precision_t* encoder_weights_device,
                               std::unique_ptr<Module> predictor, size_t predictor_n_params,
                               const ::tcnn::network_precision_t* predictor_weights_device);

        Module& encoder()   { return *m_encoder; }
        Module& predictor() { return *m_predictor; }
        const ::tcnn::GPUMemory<::tcnn::network_precision_t>& encoder_weights()   const { return m_encoder_weights; }
        const ::tcnn::GPUMemory<::tcnn::network_precision_t>& predictor_weights() const { return m_predictor_weights; }

        // Full inference: encode the per-field beam, then predict the field for
        // each voxel. `xyz` is (3, n_voxels); `beam` is the encoder's raw input
        // (1 field). Output is (predictor.output_width(), n_voxels).
        void forward(cudaStream_t stream,
                     const ::tcnn::GPUMatrixDynamic<float>& xyz,
                     const ::tcnn::GPUMatrixDynamic<float>& beam,
                     ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>& output);

    private:
        std::unique_ptr<Module> m_encoder;
        std::unique_ptr<Module> m_predictor;
        ::tcnn::GPUMemory<::tcnn::network_precision_t> m_encoder_weights;
        ::tcnn::GPUMemory<::tcnn::network_precision_t> m_predictor_weights;
        ::tcnn::GPUMemory<::tcnn::network_precision_t> m_encoder_grad;    // unused placeholder for set_params
        ::tcnn::GPUMemory<::tcnn::network_precision_t> m_predictor_grad;
    };

}  // namespace rfnn::tcnn
