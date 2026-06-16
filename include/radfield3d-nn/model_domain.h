#pragma once
//
// Model I/O domain descriptors — the pure-data types that describe WHAT a deployed model is:
// the metric range/unit of each beam parameter it generalises over, the output spectrum layout,
// and lightweight training provenance. No I/O, no ONNX, no CUDA.
//
// Shared by the package format (rfnn::io::V1::ModelStore, model_io.h) which writes/reads them,
// and by the runtime predictor (radfield3dnn::VolumeFieldPredictor, field_predictors.h) which
// carries them once loaded — so this header sits below both and neither has to include the other.
//
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace rfnn {
namespace io {

// What kind of value a beam parameter's range carries. Stored per entry so the format is
// self-describing and forward-compatible: a reader reads the type + the entry's byte length and
// either deserialises the payload it understands or skips those bytes — so existing metadata can
// gain new range kinds without breaking older readers (they just iterate the names).
enum class ParameterRangeType : uint8_t {
    MinMax   = 0,   // a [min, max] interval (+ unit): direction, distance, opening angle, …
    Spectrum = 1,   // a histogram range [min, max] with a fixed bin_width (+ unit): the tube spectrum
    Map      = 2,   // a nested name -> range map (children)
};

// One beam parameter's range — a tagged variant. Which members are meaningful depends on `type`:
//   MinMax   -> min, max, unit
//   Spectrum -> min, max, bin_width, unit   (bins = round((max - min) / bin_width))
//   Map      -> children (ordered name -> range)
struct ParameterRange {
    ParameterRangeType type      = ParameterRangeType::MinMax;
    float         min       = 0.f;
    float         max       = 0.f;
    float         bin_width = 0.f;   // Spectrum only
    std::string   unit;              // MinMax / Spectrum
    std::vector<std::pair<std::string, ParameterRange>> children;   // Map only
};

// One entry of the model's beam-parameter input layout: a name and its typed range. The ordered
// list describes the whole beam-parameter vector the model generalises over.
struct BeamParameter {
    std::string name;     // e.g. "direction", "distance", "spectrum", "opening_angle"
    ParameterRange   range;
};

// The model's fixed I/O domain, in metric units. A model generalises over a *range* of beam
// parameters, so we store that range and how the normalised inputs/outputs map to physical units
// rather than any single simulation's tube settings. The voxel *resolution* is NOT part of the model
// (chosen at inference, varies across a dataset), but the physical *field dimensions* (the metric box
// the normalised [0,1]^3 positions map into, from the training dataset) ARE — they fix what metric
// location a relative location inside the neural field corresponds to.
struct ModelDomain {
    int                        spectrum_bins = 0;            // output spectrum histogram bins …
    float                      spectrum_max_energy_ev = 0.f; // … bin i spans [i, i+1)·max/bins eV
    std::array<float, 3>       field_dimensions_m{ {0.f, 0.f, 0.f} }; // metric field box (m); 0 = unknown
    std::vector<BeamParameter> beam_parameters;              // ordered model input-vector layout
};

// Lightweight provenance — what the model was trained on. No per-simulation tube metadata.
struct ModelProvenance {
    std::string dataset_name;
    std::string software_version;
    std::string physics;
};

}  // namespace io
}  // namespace rfnn
