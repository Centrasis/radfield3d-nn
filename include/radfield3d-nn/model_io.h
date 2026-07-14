#pragma once
//
// Model store/load factories.
//
// Two factories, kept apart by namespace:
//
//   * rfnn::io::V1::ModelStore — the main store (the V1 RF3M format). Binds exported ONNX graphs
//                                to the RadFiled3D field geometry it predicts plus the model's
//                                *validity domain* (parameter ranges + the physical meaning of
//                                the normalised inputs/outputs, in metric units) and the test
//                                metrics. Saves a single self-contained "RF3M" artifact, and
//                                loads one STRAIGHT to a runnable predictor (the parse + build is
//                                one step — load() returns the VoxelFieldPredictor /
//                                VolumeFieldPredictor, carrying the package metadata on it). No
//                                tiny-cuda-nn / libtorch / CUDA dependency — lives in the deploy
//                                lib (libRadField3DNNDeploy).
//
//   * rfnn::tcnn::ModelFactory — the serialiser for the fused tcnn models
//                                (encoder+predictor raw network_precision_t weights). Only built
//                                with RFNN_WITH_TCNN; pulls in tiny-cuda-nn.
//
// ── RF3M deployment container (little-endian) ────────────────────────────────
//   [4]   magic "RF3M"
//   [u32] backend  (ModelBackend; == 2 ONNX. 1 was tiny-cuda-nn, never shipped, rejected on load)
//   [u32] version  (schema version WITHIN that backend; == 1)
//   --- ModelInterface: WHAT the model consumes/produces (model_interface.h). Declared here when
//       the model is STORED and used at LOAD to pick the executable predictor class and to check
//       the domain below actually describes every resolution the flags promise. ---
//   [u64] interface id ((inputs << 32) | outputs)
//   [u8]  resolution_aware   [i32] region_state_dims   [f32] region_width_frame
//   [u32 dataset_name_len][dataset_name bytes]
//   [u32 software_version_len][software_version bytes]
//   [u32 physics_len][physics bytes]
//   --- ModelDomain (the model's fixed I/O domain, metric units). Spatial field geometry is
//       deliberately NOT stored: the predicted resolution is chosen at inference and may vary
//       across a dataset, so it is not a property of the model. ---
//   [i32]     spectrum_bins              # output spectrum histogram bins
//   [f32]     spectrum_max_energy_ev     # bin i spans [i, i+1)·max/bins eV
//   [f32 x3]  field_dimensions_m         # the metric box [0,1]^3 positions map into
//   [i32]     angular_phi_segments   [i32] angular_theta_segments   # AngularFlux resolution
//   [u8]      collimation                # CollimationType; fixes the BeamCollimation input width
//   [u32 n_beam_params]   then n_beam_params × a beam-parameter descriptor:
//             [u32 name_len][name][u8 range_type][u32 payload_len][payload]
//   --- metrics ---
//   [u32 n_metrics]   then n_metrics × ([u32 key_len][key bytes][f32 value])
//   --- payload (named ONNX graphs that compose the model) ---
//   [u32 n_graphs]   then n_graphs × ([u32 name_len][name][u64 onnx_len][onnx bytes])
//             A model may compose several graphs around a shared "trunk" (which consumes the
//             beam parameters): e.g. "beam_encoder", later "geometry_encoder". A monolithic
//             model is a single graph (conventionally "trunk").
//
// There is exactly ONE writer: model_io.cpp. The Python producer
// (radfield3dnn/deploy/model_packager.py) gathers the metadata and calls straight into it through
// the `rfnn_deploy` bindings — it does not serialise anything itself.
//
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <radfield3d-nn/model_domain.h>
#include <radfield3d-nn/model_binding.h>   // radfield3dnn::Backend — the EP a caller asks for
#include <radfield3d-nn/model_interface.h>   // rfnn::io::{ParameterRange,BeamParameter,ModelDomain,ModelProvenance}

namespace radfield3dnn {
class VolumeFieldPredictor;   // field_predictors.h (fwd-decl; base predictor, defined in the deploy lib)
struct ExecutionOptions;      // field_predictors.h (fwd-decl; EP/fp16 selection for the load overloads)
}  // namespace radfield3dnn

namespace rfnn {
namespace io {
namespace V1 {

// A model is one or more named ONNX graphs. They compose around a shared "trunk" (which consumes
// the beam-parameter vector); encoders/decoders feed into or out of it. Names are free-form; the
// constants below are the conventional ones a deployment looks for.
using NamedGraphs = std::map<std::string, std::vector<uint8_t>>;
inline constexpr const char* kTrunkGraph           = "trunk";             // consumes beam parameters
inline constexpr const char* kBeamEncoderGraph     = "beam_encoder";      // beam vector → latent
inline constexpr const char* kGeometryEncoderGraph = "geometry_encoder";  // (future) geometry → latent

// Which runtime the payload targets. The first header word has always discriminated this; it is
// now named for what it does. TCNN was never shipped and is rejected on load.
enum class ModelBackend : uint32_t { TCNN = 1, ONNX = 2 };

class ModelStore {
public:
    static constexpr char     kMagic[4] = {'R', 'F', '3', 'M'};
    static constexpr ModelBackend kBackend = ModelBackend::ONNX;
    static constexpr uint32_t     kVersion = 1;   // schema version within the ONNX backend

    // Build the RF3M container in memory (the single source of the byte layout). `graphs` is the
    // named set of ONNX graphs composing the model (at least a "trunk").
    // `interface` is the model's DECLARATION of what it consumes/produces. It is validated against
    // `domain` here, so a package can never promise an output whose resolution nobody recorded.
    static std::vector<uint8_t> save_to_memory(const NamedGraphs& graphs,
                                               const radfield3dnn::ModelInterface& interface,
                                               const ModelDomain& domain,
                                               const ModelProvenance& provenance,
                                               const std::map<std::string, float>& metrics);

    // Same, written straight to `path`.
    static void save(const std::string& path,
                     const NamedGraphs& graphs,
                     const radfield3dnn::ModelInterface& interface,
                     const ModelDomain& domain,
                     const ModelProvenance& provenance,
                     const std::map<std::string, float>& metrics);

    // Parse a package AND build its runnable predictor in one step (no intermediate handle, never
    // touching disk for the graphs). The DECLARED interface decides the predictor: ModelInput::Position
    // set -> VoxelFieldPredictor (wired with the "beam_encoder" graph if present); clear ->
    // VolumeFieldPredictor (trunk only). The trunk graph is cross-checked against the declaration. The package's domain (parameter ranges applied),
    // provenance, metrics and graph names are carried ON the returned predictor (see its
    // domain()/provenance()/metrics()/graph_names()). Dynamic type is VoxelFieldPredictor for
    // per-voxel models; the static return type is the base.
    static std::unique_ptr<radfield3dnn::VolumeFieldPredictor>
        load_from_memory(const void* bytes, size_t n, bool use_cuda = true);
    static std::unique_ptr<radfield3dnn::VolumeFieldPredictor>
        load(const std::string& path, bool use_cuda = true);

    // Same, with full execution-provider control (TensorRT/CUDA/CPU, fp16, engine cache). Use these to
    // expose the TensorRT fp16 knob — fp16 kernels are faster but shift results a few % vs fp32, so it is a
    // speed/accuracy trade the caller should make explicitly (the bool overloads above keep ExecutionOptions'
    // defaults: GPU on, TensorRT on, fp16 on).
    static std::unique_ptr<radfield3dnn::VolumeFieldPredictor>
        load_from_memory(const void* bytes, size_t n, const radfield3dnn::ExecutionOptions& exec);
    static std::unique_ptr<radfield3dnn::VolumeFieldPredictor>
        load(const std::string& path, const radfield3dnn::ExecutionOptions& exec);

    // Request an execution provider by name/enum — the same vocabulary Python's
    // load_rf3m(backend=...) uses, because it IS this function. A backend fixes which memory the
    // model can bind with no copy, so ask for the one your buffers already live on.
    static std::unique_ptr<radfield3dnn::VolumeFieldPredictor>
        load(const std::string& path, radfield3dnn::Backend backend);
    static std::unique_ptr<radfield3dnn::VolumeFieldPredictor>
        load_from_memory(const void* bytes, size_t n, radfield3dnn::Backend backend);

    // The package metadata that lives in the RF3M header, ahead of the ONNX graphs.
    struct PackageMetadata {
        ModelProvenance              provenance;
        ModelDomain                  domain;
        std::map<std::string, float> metrics;
        radfield3dnn::ModelInterface interface;   // what the model consumes/produces
    };

    // Read ONLY the metadata header (interface + provenance + domain + metrics) — WITHOUT loading the ONNX graphs
    // or building a runnable predictor (no ONNX Runtime session). The graphs are serialised last, so
    // this stops before them. Use this for UI / metadata display; it is cheap and must never touch ORT.
    // (Predictor *type* — voxel vs volume — is NOT here; it needs the trunk graph, i.e. a real load.)
    static PackageMetadata read_metadata_from_memory(const void* bytes, size_t n);
    static PackageMetadata read_metadata(const std::string& path);

    // Read the raw named ONNX graphs (name -> protobuf bytes) WITHOUT building a predictor / touching
    // ONNX Runtime. This is the read counterpart to save_to_memory, so tools that re-pack a package
    // (e.g. fp16 conversion) never have to re-implement the RF3M byte layout in Python.
    static NamedGraphs read_graphs_from_memory(const void* bytes, size_t n);
    static NamedGraphs read_graphs(const std::string& path);
};

}  // namespace V1
}  // namespace io
}  // namespace rfnn

#ifdef RFNN_WITH_TCNN
#include <radfield3d-nn/tcnn/combined_model.h>

namespace rfnn {
namespace tcnn {

    // Raw-weight serialiser for the fused tiny-cuda-nn (encoder, predictor) pair. Library types
    // are fully qualified `::tcnn::…` because this namespace shadows the tiny-cuda-nn `::tcnn`.
    class ModelFactory {
    public:
        static constexpr char kMagic[6] = {'R', 'F', 'N', 'N', 'M', '\0'};
        static constexpr uint8_t kVersion = 1;

        static void save(const std::string& path,
                         const ::tcnn::network_precision_t* encoder_weights_device,
                         const std::string& encoder_type, const std::string& encoder_hparams_json,
                         size_t encoder_n_params,
                         const ::tcnn::network_precision_t* predictor_weights_device,
                         const std::string& predictor_type, const std::string& predictor_hparams_json,
                         size_t predictor_n_params);

        static std::unique_ptr<CombinedRadiationModel> load(const std::string& path);
    };

}  // namespace tcnn
}  // namespace rfnn
#endif  // RFNN_WITH_TCNN
