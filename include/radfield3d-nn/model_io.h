#pragma once
//
// Model store/load factories.
//
// Two factories, kept apart by namespace:
//
//   * rfnn::io::V1::ModelFactory — the main factory (the V1 RF3M format). Binds exported ONNX graphs
//                                to the RadFiled3D field geometry it predicts plus the model's
//                                *validity domain* (parameter ranges + the physical meaning of
//                                the normalised inputs/outputs, in metric units) and the test
//                                metrics. Saves/loads a single self-contained "RF3M" artifact,
//                                to disk OR to/from memory. No tiny-cuda-nn / libtorch / CUDA
//                                dependency — it lives in the deploy lib (libRadField3DNNDeploy).
//
//   * rfnn::tcnn::ModelFactory — the legacy serialiser for the fused tcnn models
//                                (encoder+predictor raw network_precision_t weights). Only built
//                                with RFNN_WITH_TCNN; pulls in tiny-cuda-nn.
//
// ── RF3M deployment container (little-endian) ────────────────────────────────
//   [4]   magic "RF3M"
//   [u32] version (== 2)
//   [u32 dataset_name_len][dataset_name bytes]
//   [u32 software_version_len][software_version bytes]
//   [u32 physics_len][physics bytes]
//   --- ModelDomain (the model's fixed I/O domain, metric units). Spatial field geometry is
//       deliberately NOT stored: the predicted resolution is chosen at inference and may vary
//       across a dataset, so it is not a property of the model. ---
//   [i32]     spectrum_bins              # output spectrum histogram bins
//   [f32]     spectrum_max_energy_ev     # bin i spans [i, i+1)·max/bins eV
//   [u32 n_beam_params]   then n_beam_params × a beam-parameter descriptor:
//             [u32 name_len][name][i32 count][f32 range_min][f32 range_max][u32 unit_len][unit]
//   --- metrics ---
//   [u32 n_metrics]   then n_metrics × ([u32 key_len][key bytes][f32 value])
//   --- payload (named ONNX graphs that compose the model) ---
//   [u32 n_graphs]   then n_graphs × ([u32 name_len][name][u64 onnx_len][onnx bytes])
//             A model may compose several graphs around a shared "trunk" (which consumes the
//             beam parameters): e.g. "beam_encoder", later "geometry_encoder". A monolithic
//             model is a single graph (conventionally "trunk").
//
// The Python producer (radfield3dnn/deploy/model_packager.py) writes the *same* layout; this
// header / model_io.cpp is the authoritative format definition.
//
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace radfield3dnn {
class VolumeFieldPredictor;   // field_predictors.h (fwd-decl; base predictor, defined in the deploy lib)
}  // namespace radfield3dnn

namespace rfnn {
namespace io {

// Valid range + physical unit of one beam parameter (a segment of the model's input vector).
struct ParameterRange {
    float       min  = 0.f;
    float       max  = 0.f;
    std::string unit;        // e.g. "m", "deg", "eV", "" (dimensionless, e.g. a unit direction)
};

// One entry of the model's beam-parameter input vector: its name, how many scalar slots of the
// input vector it occupies, and the metric range/unit those slots are valid over. The ordered
// list describes the whole beam-parameter vector passed to volume prediction.
struct BeamParameter {
    std::string    name;     // e.g. "direction", "distance", "spectrum", "opening_angle"
    int            count = 0;  // number of input-vector slots this parameter spans
    ParameterRange range;
};

// The model's fixed I/O domain, in metric units. A model generalises over a *range* of beam
// parameters, so we store that range and how the normalised inputs/outputs map to physical units
// rather than any single simulation's tube settings. The spatial field geometry (resolution /
// voxel size) is NOT part of the model — it is chosen at inference and may vary across a dataset.
struct ModelDomain {
    int                        spectrum_bins = 0;            // output spectrum histogram bins …
    float                      spectrum_max_energy_ev = 0.f; // … bin i spans [i, i+1)·max/bins eV
    std::vector<BeamParameter> beam_parameters;              // ordered model input-vector layout
};

// Lightweight provenance — what the model was trained on. No per-simulation tube metadata.
struct ModelProvenance {
    std::string dataset_name;
    std::string software_version;
    std::string physics;
};

namespace V1 {

// A model is one or more named ONNX graphs. They compose around a shared "trunk" (which consumes
// the beam-parameter vector); encoders/decoders feed into or out of it. Names are free-form; the
// constants below are the conventional ones a deployment looks for.
using NamedGraphs = std::map<std::string, std::vector<uint8_t>>;
inline constexpr const char* kTrunkGraph           = "trunk";             // consumes beam parameters
inline constexpr const char* kBeamEncoderGraph     = "beam_encoder";      // beam vector → latent
inline constexpr const char* kGeometryEncoderGraph = "geometry_encoder";  // (future) geometry → latent

struct LoadedModel {
    NamedGraphs                  graphs;       // name → ONNX bytes ("trunk", "beam_encoder", …)
    ModelDomain                  domain;
    ModelProvenance              provenance;
    std::map<std::string, float> metrics;

    bool has(const std::string& name) const { return graphs.find(name) != graphs.end(); }

    // Construct the runnable predictor from the embedded graphs (never touching disk). The factory
    // decides by the "trunk" graph type: a per-voxel trunk -> VoxelFieldPredictor (wired with the
    // "beam_encoder" graph if present); a field-wise trunk -> VolumeFieldPredictor (trunk only).
    std::unique_ptr<radfield3dnn::VolumeFieldPredictor> build(bool use_cuda = true) const;
};

class ModelFactory {
public:
    static constexpr char     kMagic[4] = {'R', 'F', '3', 'M'};
    static constexpr uint32_t kVersion  = 2;

    // Build the RF3M container in memory (the single source of the byte layout). `graphs` is the
    // named set of ONNX graphs composing the model (at least a "trunk").
    static std::vector<uint8_t> save_to_memory(const NamedGraphs& graphs,
                                               const ModelDomain& domain,
                                               const ModelProvenance& provenance,
                                               const std::map<std::string, float>& metrics);

    // Same, written straight to `path`.
    static void save(const std::string& path,
                     const NamedGraphs& graphs,
                     const ModelDomain& domain,
                     const ModelProvenance& provenance,
                     const std::map<std::string, float>& metrics);

    // Parse a package: the named ONNX graphs (kept as bytes; build runnable models on demand via
    // LoadedModel::build), the validity domain, provenance and metrics. Never touches disk for
    // the graphs.
    static LoadedModel load_from_memory(const void* bytes, size_t n);
    static LoadedModel load(const std::string& path);
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
