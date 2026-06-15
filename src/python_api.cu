#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <typeinfo>
#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include "radfield3d-nn-bindings/utils.h"
#include "radfield3d-nn/tcnn/encodings/location_encoding.h"
#include "radfield3d-nn/tcnn/encodings/global_parameters.h"
#include "radfield3d-nn/tcnn/encodings/beam_encoder.h"
#include "radfield3d-nn/tcnn/encodings/sperf_beam_encoder.h"
#include "radfield3d-nn/tcnn/layers/film.h"
#include "radfield3d-nn/tcnn/layers/layer_norm.h"
#include "radfield3d-nn/tcnn/base_model.h"
#include "radfield3d-nn/tcnn/utils/histogram_resample.h"
#include <ATen/cuda/CUDAContext.h>

#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <exception>
#include <cctype>
#include <string>

#include <radfield3d-nn-bindings/bridge.inl>

#ifdef WIN32
typedef Py_ssize_t ssize_t;
#endif

namespace py = pybind11;


using LocationEncodingBridgeImpl    = rfnn::tcnn::autograd::ModuleBridge<rfnn::tcnn::LocationEncoding,    rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::LocationEncoding,    float>>;
using ParameterSetEncodingBridgeImpl = rfnn::tcnn::autograd::ModuleBridge<rfnn::tcnn::ParameterSetEncoding, rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::ParameterSetEncoding, float>>;
using FiLMBridgeImpl                 = rfnn::tcnn::autograd::ModuleBridge<rfnn::tcnn::FiLM,                rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::FiLM,                float>>;
using LayerNormBridgeImpl            = rfnn::tcnn::autograd::ModuleBridge<rfnn::tcnn::LayerNorm,           rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::LayerNorm,           float>>;
using PBRFBeamEncoderBridgeImpl      = rfnn::tcnn::autograd::ModuleBridge<rfnn::tcnn::PBRFBeamEncoder,     rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::PBRFBeamEncoder,     float>>;
using SPERFBeamEncoderBridgeImpl     = rfnn::tcnn::autograd::ModuleBridge<rfnn::tcnn::SPERFBeamEncoder,    rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::SPERFBeamEncoder,    float>>;
using MainModelBridgeImpl            = rfnn::tcnn::autograd::ModuleBridge<rfnn::tcnn::BaseRadiationPredictionModel, rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::BaseRadiationPredictionModel, float>>;

TORCH_MODULE(LocationEncodingBridge);
TORCH_MODULE(ParameterSetEncodingBridge);
TORCH_MODULE(FiLMBridge);
TORCH_MODULE(LayerNormBridge);
TORCH_MODULE(PBRFBeamEncoderBridge);
TORCH_MODULE(SPERFBeamEncoderBridge);
TORCH_MODULE(MainModelBridge);


// Python facade for PBRFBeamEncoder: takes the three raw beam tensors and
// concatenates them into the single input tensor the underlying bridge
// expects. Keeps the C++ DifferentiableObject contract (single input tensor)
// while giving Python a clean three-argument forward.
class PBRFBeamEncoderPyImpl : public torch::nn::Module {
    std::shared_ptr<PBRFBeamEncoderBridgeImpl> bridge;

    public:
    PBRFBeamEncoderPyImpl(uint32_t spectrum_dim, uint32_t d_model, uint32_t distance_bins) {
        this->bridge = std::make_shared<PBRFBeamEncoderBridgeImpl>(spectrum_dim, d_model, distance_bins);
        // Expose the bridge's weights as a parameter of *this* module rather than
        // registering the bridge as a submodule: torch.nn.cpp.ModuleWrapper yields
        // raw C++ children out of _modules and then tries to recurse into them with
        // Python kwargs (memo, remove_duplicate, ...) that the C++ binding doesn't
        // accept, which breaks named_parameters() / named_modules() on the outer
        // Python module. The tensor object is shared, so optimizer updates flow
        // straight back into the bridge's storage.
        this->register_parameter("weights", this->bridge->torch_weights());
    }

    torch::Tensor forward(torch::Tensor direction, torch::Tensor distance, torch::Tensor spectrum) {
        TORCH_CHECK(direction.dim() == 2 && direction.size(1) == 3, "direction must be [B,3]");
        TORCH_CHECK(distance.dim()  == 2 && distance.size(1)  == 1, "distance must be [B,1]");
        TORCH_CHECK(spectrum.dim()  == 2,                            "spectrum must be 2-D");
        TORCH_CHECK(direction.size(0) == distance.size(0) && distance.size(0) == spectrum.size(0),
                    "direction/distance/spectrum batch sizes must match");

        torch::Tensor concat = torch::cat({direction, distance, spectrum}, /*dim=*/1).contiguous();
        return this->bridge->forward<torch::Tensor>(concat);
    }
};


// Python facade for SPERFBeamEncoder: distance-less variant for fixed-distance
// datasets. Concatenates (direction, spectrum) into the single input tensor the
// bridge expects.
class SPERFBeamEncoderPyImpl : public torch::nn::Module {
    std::shared_ptr<SPERFBeamEncoderBridgeImpl> bridge;

    public:
    SPERFBeamEncoderPyImpl(uint32_t spectrum_dim, uint32_t d_model) {
        this->bridge = std::make_shared<SPERFBeamEncoderBridgeImpl>(spectrum_dim, d_model);
        this->register_parameter("weights", this->bridge->torch_weights());
    }

    torch::Tensor forward(torch::Tensor direction, torch::Tensor spectrum) {
        TORCH_CHECK(direction.dim() == 2 && direction.size(1) == 3, "direction must be [B,3]");
        TORCH_CHECK(spectrum.dim()  == 2,                            "spectrum must be 2-D");
        TORCH_CHECK(direction.size(0) == spectrum.size(0),
                    "direction/spectrum batch sizes must match");

        torch::Tensor concat = torch::cat({direction, spectrum}, /*dim=*/1).contiguous();
        return this->bridge->forward<torch::Tensor>(concat);
    }
};


// Python facade for BaseRadiationPredictionModel: concatenates (xyz, beam_encoded)
// into the single input tensor the underlying bridge expects and splits the
// 33-channel output into (flux, spectrum). The model itself has no
// knowledge of how `beam_encoded` was produced.
class BaseRadiationPredictionModelPyImpl : public torch::nn::Module {
    std::shared_ptr<MainModelBridgeImpl> bridge;

    public:
    // Convert a user-facing string ("frequency" / "hashgrid") into the
    // C++ LocationEncodingKind enum. Case-insensitive; common aliases
    // accepted ("fourier" == "frequency", "hash" / "hash_grid" == "hashgrid").
    // Invalid strings raise std::invalid_argument which pybind11 translates to
    // a Python InvalidArgument (registered in PYBIND11_MODULE).
    static rfnn::tcnn::LocationEncodingKind parse_kind(const std::string& s) {
        std::string lower; lower.reserve(s.size());
        for (char c : s) lower.push_back((char)std::tolower((unsigned char)c));
        if (lower == "frequency" || lower == "fourier" || lower == "sinusoidal") {
            return rfnn::tcnn::LocationEncodingKind::Frequency;
        }
        if (lower == "hashgrid" || lower == "hash" || lower == "hash_grid") {
            return rfnn::tcnn::LocationEncodingKind::HashGrid;
        }
        throw std::invalid_argument(
            "location_encoding_kind must be one of {frequency, hashgrid}, got: " + s);
    }

    BaseRadiationPredictionModelPyImpl(uint32_t d_model, uint32_t location_encoding_dim, float flux_offset, int flux_activation,
                                       const std::string& location_encoding_kind,
                                       float flux_clamp_min, float flux_clamp_max,
                                       uint32_t trunk_hidden_layers,
                                       const std::string& beam_fusion) {
        const rfnn::tcnn::LocationEncodingKind kind = parse_kind(location_encoding_kind);
        const rfnn::tcnn::BeamFusionKind fusion = rfnn::tcnn::parse_beam_fusion_kind(beam_fusion);
        if (flux_clamp_min >= flux_clamp_max) {
            throw std::invalid_argument(
                "flux_clamp_min must be strictly less than flux_clamp_max.");
        }
        if (flux_activation == 1 && (flux_clamp_min != 0.0f || flux_clamp_max != 1.0f)) {
            throw std::invalid_argument(
                "softclip flux_activation only produces [0, 1]; non-default flux_clamp_min/max "
                "is incompatible. Use flux_activation='clamp' for a configurable range.");
        }
        this->bridge = std::make_shared<MainModelBridgeImpl>(d_model, location_encoding_dim, flux_offset, flux_activation, kind,
                                                              flux_clamp_min, flux_clamp_max, trunk_hidden_layers, fusion);
        // See PBRFBeamEncoderPyImpl above for rationale (register_parameter, not register_module).
        this->register_parameter("weights", this->bridge->torch_weights());
    }

    py::tuple forward(torch::Tensor xyz, torch::Tensor beam_encoded) {
        TORCH_CHECK(xyz.dim() == 2 && xyz.size(1) == 3, "xyz must be [B,3]");
        TORCH_CHECK(beam_encoded.dim() == 2, "beam_encoded must be 2-D");
        TORCH_CHECK(xyz.size(0) == beam_encoded.size(0), "xyz and beam_encoded batches must match");

        torch::Tensor beam_as_float = beam_encoded.to(torch::kFloat32);
        torch::Tensor concat = torch::cat({xyz, beam_as_float}, /*dim=*/1).contiguous();

        // Output layout (33 channels, single-head):
        //   [0]    = flux (joined per-volume-relative flux),
        //   [1..33) = joined spectrum (32 bins).
        torch::Tensor out = this->bridge->forward<torch::Tensor>(concat);
        torch::Tensor flux     = out.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)});
        torch::Tensor spectrum = out.index({torch::indexing::Slice(), torch::indexing::Slice(1, 33)});
        return py::make_tuple(flux, spectrum);
    }

    // View tensors into the flat `weights` blob, split at the head boundary:
    //   trunk_weights() = weights[:off]  (shared trunk: enc, FiLM1, mlp_block,
    //                                     mlp_post, FiLM2),
    //   head_weights()  = weights[off:]  (the spectrum + flux output heads).
    // Both SHARE storage with `weights` (no copy), so reads reflect live values
    // and `.grad` for these slices is `weights.grad[:off] / [off:]`. DB-MTL uses
    // `trunk_weights().numel()` to restrict its per-task gradient norm to the
    // shared trunk (excluding the heads that live in the same fused blob).
    torch::Tensor trunk_weights() {
        const int64_t off = static_cast<int64_t>(this->bridge->output_head_param_offset());
        return this->bridge->torch_weights().narrow(0, 0, off);
    }
    torch::Tensor head_weights() {
        torch::Tensor w = this->bridge->torch_weights();
        const int64_t off = static_cast<int64_t>(this->bridge->output_head_param_offset());
        return w.narrow(0, off, w.numel() - off);
    }
};

TORCH_MODULE(PBRFBeamEncoderPy);
TORCH_MODULE(SPERFBeamEncoderPy);
TORCH_MODULE(BaseRadiationPredictionModelPy);


PYBIND11_MODULE(radfield3dnn, m) {
    py::register_exception<std::invalid_argument>(m, "InvalidArgument");
    py::register_exception<std::out_of_range>(m, "OutOfRange");
    // NOTE: do NOT register translators for std::exception or std::runtime_error
    // here. pybind11 signals end-of-iteration by throwing pybind11::stop_iteration,
    // which derives from std::runtime_error (and therefore std::exception); a
    // module-local translator for either base class swallows it and turns it into
    // a generic Python exception, which breaks named_parameters() /
    // named_modules() and every other iterator coming out of the C++ Module
    // dicts. std::out_of_range and std::invalid_argument inherit from
    // std::logic_error and are safe to translate.

    // CUDA histogram resampler — torch glue around the torch-free CUDA kernel
    // (rfnn::tcnn::utils::resample_histogram_bilinear, src/RadField3DNN/utils/
    // histogram_resample.cu), the C++/CUDA port of
    // radfield3dnn.utils.mean_sampling.resample_histogram_bilinear so the fused
    // cpp encoders can resample the raw beam spectrum during pure-C++ inference.
    m.def("resample_histogram_bilinear",
          [](const torch::Tensor& histogram, int64_t target_bins) -> torch::Tensor {
              TORCH_CHECK(histogram.dim() == 2, "histogram must be 2-D (N, source_bins)");
              TORCH_CHECK(histogram.is_cuda(), "histogram must be a CUDA tensor");
              TORCH_CHECK(target_bins > 0, "target_bins must be positive");
              auto h = histogram.contiguous().to(torch::kFloat32);
              auto out = torch::empty({h.size(0), target_bins}, h.options());
              rfnn::tcnn::utils::resample_histogram_bilinear(
                  at::cuda::getCurrentCUDAStream(),
                  h.data_ptr<float>(), out.data_ptr<float>(),
                  static_cast<uint32_t>(h.size(0)), static_cast<uint32_t>(h.size(1)),
                  static_cast<uint32_t>(target_bins));
              return out.to(histogram.dtype());
          },
          py::arg("histogram"), py::arg("target_bins"));

    // Standalone LocationEncoding binding stays on the two-argument form
    // (defaults to Frequency); the kind selector is exposed through the
    // BaseRadiationPredictionModel facade below, which is the only path the
    // training pipeline actually uses.
    torch::python::bind_module<LocationEncodingBridgeImpl>(m, "LocationEncoding")
        .def(py::init<uint32_t, uint32_t>())
        .def("forward", &LocationEncodingBridgeImpl::forward<torch::Tensor>);

    torch::python::bind_module<FiLMBridgeImpl>(m, "FiLM")
        .def(py::init<uint32_t, uint32_t, const std::string&>(), py::arg("feature_channels"), py::arg("condition_channels"), py::arg("non_linearity") = "ReLU")
        .def("forward", &FiLMBridgeImpl::forward<torch::Tensor>);

    torch::python::bind_module<LayerNormBridgeImpl>(m, "LayerNorm")
        .def(py::init<uint32_t, float>(), py::arg("channels"), py::arg("eps") = 1e-5f)
        .def("forward", &LayerNormBridgeImpl::forward<torch::Tensor>);

    torch::python::bind_module<PBRFBeamEncoderPyImpl>(m, "PBRFBeamEncoder")
        .def(py::init<uint32_t, uint32_t, uint32_t>(), py::arg("spectrum_dim"), py::arg("d_model"), py::arg("distance_bins") = 16)
        .def("forward", &PBRFBeamEncoderPyImpl::forward, py::arg("direction"), py::arg("distance"), py::arg("spectrum"));

    torch::python::bind_module<SPERFBeamEncoderPyImpl>(m, "SPERFBeamEncoder")
        .def(py::init<uint32_t, uint32_t>(), py::arg("spectrum_dim"), py::arg("d_model"))
        .def("forward", &SPERFBeamEncoderPyImpl::forward, py::arg("direction"), py::arg("spectrum"));

    torch::python::bind_module<BaseRadiationPredictionModelPyImpl>(m, "BaseRadiationPredictionModel")
        .def(py::init<uint32_t, uint32_t, float, int, const std::string&, float, float, uint32_t, const std::string&>(),
             py::arg("d_model"), py::arg("location_encoding_dim") = 12,
             py::arg("flux_offset") = 0.5f, py::arg("flux_activation") = 0,
             py::arg("location_encoding_kind") = std::string("frequency"),
             py::arg("flux_clamp_min") = 0.0f, py::arg("flux_clamp_max") = 1.0f,
             py::arg("trunk_hidden_layers") = 1u,
             py::arg("beam_fusion") = std::string("film"))
        .def("forward", &BaseRadiationPredictionModelPyImpl::forward, py::arg("xyz"), py::arg("beam_encoded"))
        .def("trunk_weights", &BaseRadiationPredictionModelPyImpl::trunk_weights)
        .def("head_weights", &BaseRadiationPredictionModelPyImpl::head_weights);
}
