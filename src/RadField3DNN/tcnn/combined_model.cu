#include <radfield3d-nn/tcnn/combined_model.h>
#include <radfield3d-nn/tcnn/encodings/sperf_beam_encoder.h>
#include <radfield3d-nn/tcnn/encodings/location_encoding.h>
#include <tiny-cuda-nn/common.h>
#include <json/json.hpp>
#include <stdexcept>

namespace rfnn::tcnn {

    using json = nlohmann::json;
    using Module = CombinedRadiationModel::Module;

    // Broadcast the single beam encoding (d_model,1) into rows [3, 3+d_model) of
    // every column of the column-major predictor input (in_w, n_voxels).
    __global__ void broadcast_encoding(uint32_t n_elements, uint32_t d_model, uint32_t in_w,
                                       const ::tcnn::network_precision_t* __restrict__ encoded,
                                       float* __restrict__ pred_in) {
        const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements) return;
        const uint32_t voxel = i / d_model;
        const uint32_t k = i % d_model;
        pred_in[voxel * in_w + 3u + k] = (float)encoded[k];
    }

    std::unique_ptr<Module> build_module_from_hparams(const std::string& hparams_json, size_t& n_params_out) {
        json h = json::parse(hparams_json);
        const std::string otype = h.at("otype").get<std::string>();
        std::unique_ptr<Module> m;

        if (otype == "PBRFBeamEncoder") {
            m.reset(new PBRFBeamEncoder(
                h.at("spectrum_dim").get<uint32_t>(),
                h.at("d_model").get<uint32_t>(),
                h.value("distance_bins", 16u)));
        } else if (otype == "SPERFBeamEncoder") {
            m.reset(new SPERFBeamEncoder(
                h.at("spectrum_dim").get<uint32_t>(),
                h.at("d_model").get<uint32_t>(),
                h.value("distance_bins", 16u)));
        } else if (otype == "BaseRadiationPredictionModel") {
            m.reset(new BaseRadiationPredictionModel(
                h.at("d_model").get<uint32_t>(),
                h.value("location_encoding_dim", 12u),
                h.value("flux_offset", 0.5f),
                h.value("flux_activation", 0),
                static_cast<LocationEncodingKind>(h.value("location_encoding_kind", 0)),
                h.value("flux_clamp_min", 0.0f),
                h.value("flux_clamp_max", 1.0f),
                h.value("trunk_hidden_layers", 1u),
                static_cast<BeamFusionKind>(h.value("beam_fusion", 0))));
        } else {
            throw std::runtime_error("ModelFactory: unknown otype '" + otype + "'");
        }
        n_params_out = m->n_params();
        return m;
    }

    static void wire(Module& mod, size_t n_params,
                     const ::tcnn::network_precision_t* src_device,
                     ::tcnn::GPUMemory<::tcnn::network_precision_t>& weights,
                     ::tcnn::GPUMemory<::tcnn::network_precision_t>& grad) {
        weights.resize(n_params);
        grad.resize(n_params);
        if (n_params > 0 && src_device != nullptr) {
            CUDA_CHECK_THROW(cudaMemcpy(weights.data(), src_device,
                             n_params * sizeof(::tcnn::network_precision_t), cudaMemcpyDeviceToDevice));
        }
        grad.memset(0);
        // params == inference_params (single owned buffer); grad is a sink.
        mod.set_params(weights.data(), weights.data(), grad.data());
    }

    CombinedRadiationModel::CombinedRadiationModel(
        std::unique_ptr<Module> encoder, size_t encoder_n_params,
        const ::tcnn::network_precision_t* encoder_weights_device,
        std::unique_ptr<Module> predictor, size_t predictor_n_params,
        const ::tcnn::network_precision_t* predictor_weights_device)
        : m_encoder(std::move(encoder)), m_predictor(std::move(predictor)) {
        wire(*m_encoder, encoder_n_params, encoder_weights_device, m_encoder_weights, m_encoder_grad);
        wire(*m_predictor, predictor_n_params, predictor_weights_device, m_predictor_weights, m_predictor_grad);
    }

    void CombinedRadiationModel::forward(cudaStream_t stream,
                                         const ::tcnn::GPUMatrixDynamic<float>& xyz,
                                         const ::tcnn::GPUMatrixDynamic<float>& beam,
                                         ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>& output) {
        // 1) encode the beam (one column) → encoded (d_model, 1)
        const uint32_t d_model = m_encoder->output_width();
        const uint32_t n_voxels = xyz.n();
        ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t> encoded(d_model, beam.n(), stream);
        m_encoder->inference_mixed_precision(stream, beam, encoded, true);

        // 2) build the predictor input (3 + d_model, n_voxels): xyz on the first
        //    three rows, the (single) beam encoding broadcast across all voxels.
        const uint32_t in_w = m_predictor->input_width();  // 3 + d_model
        ::tcnn::GPUMatrixDynamic<float> pred_in(in_w, n_voxels, stream);
        // xyz rows
        CUDA_CHECK_THROW(cudaMemcpy2DAsync(
            pred_in.data(), in_w * sizeof(float),
            xyz.data(), xyz.m() * sizeof(float),
            3 * sizeof(float), n_voxels, cudaMemcpyDeviceToDevice, stream));
        // broadcast encoded[:,0] into rows [3, 3+d_model) for every voxel
        ::tcnn::linear_kernel(broadcast_encoding, 0, stream,
                            (size_t)n_voxels * d_model, d_model, in_w, encoded.data(), pred_in.data());

        // 3) predict
        m_predictor->inference_mixed_precision(stream, pred_in, output, true);
    }

}  // namespace rfnn::tcnn
