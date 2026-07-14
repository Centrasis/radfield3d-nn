#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>

#include <radfield3d-nn/model_domain.h>

namespace radfield3dnn {

// The deployment interface of a stored model: WHICH quantities it consumes and produces. It never
// names a network — two architectures with the same I/O are interchangeable to a consumer, and a
// network that changes its I/O gets a different id instead of silently breaking existing ones.
//
// The flags are bit positions and are ABI: APPEND-ONLY. Never reorder, never reuse a retired bit.
// They encode PRESENCE only. Every resolution (spectrum bins, angular segments, collimation kind)
// lives in the model's ModelDomain (the dynamic metadata header) — validate() checks that the domain
// actually carries what the declared flags require, on both store and load.
//
// This header is the single definition; Python receives the flags through the pybind bindings.

enum class ModelInput : std::uint32_t {
    None                = 0u,
    // per-query (trunk) inputs
    Position            = 1u << 0,   // xyz. SET => per-voxel model (VoxelPredictor);
                                     //      CLEAR => whole-volume model (VolumePredictor).
    QueryDirection      = 1u << 1,   // view/query direction at the sampled point
    // per-field (beam / global encoder) inputs
    TubeSpectrum        = 1u << 2,   // tube spectrum histogram (bins from ModelDomain)
    BeamDirection       = 1u << 3,   // unit vec3
    SourceDistance      = 1u << 4,   // scalar (the 1D normalised origin)
    SourceOrigin3D      = 1u << 5,   // vec3 origin (alternative to SourceDistance)
    BeamCollimation     = 1u << 6,   // collimation parameters; the KIND (rect/cone/ellipsoid) and
                                     // hence the tensor width come from ModelDomain::collimation
    PatientTranslation3D = 1u << 7,  // vec3, metres
    PatientRotation3D   = 1u << 8,   // vec3, radians
    GeometryMap         = 1u << 9,   // dense voxelized density map
    AnodeAngle          = 1u << 10,  // degrees (heel effect)
};

enum class ModelOutput : std::uint32_t {
    None             = 0u,
    Flux             = 1u << 0,
    Spectrum         = 1u << 1,   // needs ModelDomain::spectrum_bins
    AngularFlux      = 1u << 2,   // needs ModelDomain::angular_phi_segments / _theta_segments
    Error            = 1u << 3,
    AirKerma         = 1u << 4,   // the model emits air-kerma directly (NOT flux x spectrum)
    SeparateChannels = 1u << 5,   // scatter and direct beam are emitted separately, not joined
};

// Highest bit currently defined. Everything above is RESERVED and rejected, so a package written by
// a newer producer cannot be silently half-understood by an older consumer.
inline constexpr std::uint32_t MODEL_INPUT_KNOWN_MASK  = (1u << 11) - 1u;
inline constexpr std::uint32_t MODEL_OUTPUT_KNOWN_MASK = (1u << 6) - 1u;

inline constexpr ModelInput operator|(ModelInput a, ModelInput b) {
    return static_cast<ModelInput>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}
inline constexpr ModelOutput operator|(ModelOutput a, ModelOutput b) {
    return static_cast<ModelOutput>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}
inline constexpr bool has(ModelInput set, ModelInput flag) {
    return (static_cast<std::uint32_t>(set) & static_cast<std::uint32_t>(flag)) != 0u;
}
inline constexpr bool has(ModelOutput set, ModelOutput flag) {
    return (static_cast<std::uint32_t>(set) & static_cast<std::uint32_t>(flag)) != 0u;
}

class UnsupportedInterfaceError : public std::runtime_error {
public:
    explicit UnsupportedInterfaceError(const std::string& what) : std::runtime_error(what) {}
};

// Packed (inputs << 32) | outputs. Exact equality selects the executable predictor class; the
// individual bits answer capability questions ("does this model take a translation?").
using InterfaceId = std::uint64_t;

struct ModelInterface {
    ModelInput  inputs  = ModelInput::None;
    ModelOutput outputs = ModelOutput::None;

    // How the model must be DRIVEN (not what it emits) — the location encoder may need configuring
    // for the grid being queried, so the resolution can change per inference (LOD).
    bool  resolution_aware   = false;
    int   region_state_dims  = 0;      // width of the trunk's region_state input (0 = absent)
    float region_width_frame = 1.0f;   // region_width = frame / N for an N^3 query grid

    constexpr InterfaceId id() const { return make_id(inputs, outputs); }
    constexpr bool takes(ModelInput f) const { return has(inputs, f); }
    constexpr bool gives(ModelOutput f) const { return has(outputs, f); }

    // A model that is queried at explicit positions is per-voxel; one that is not emits the whole
    // volume in one shot. The predictor class follows from this bit alone.
    constexpr bool is_voxelwise() const { return takes(ModelInput::Position); }

    static constexpr InterfaceId make_id(ModelInput in, ModelOutput out) {
        return (static_cast<InterfaceId>(static_cast<std::uint32_t>(in)) << 32)
             | static_cast<InterfaceId>(static_cast<std::uint32_t>(out));
    }

    // Reject reserved bits, then check that the domain carries every resolution the declared flags
    // depend on. Called when a model is STORED and again when it is LOADED, so a package can never
    // promise an output whose shape nobody recorded.
    void validate(const rfnn::io::ModelDomain& domain) const {
        const auto in_bits  = static_cast<std::uint32_t>(inputs);
        const auto out_bits = static_cast<std::uint32_t>(outputs);
        if (in_bits & ~MODEL_INPUT_KNOWN_MASK)
            throw UnsupportedInterfaceError("model interface declares reserved input bits (written "
                                            "by a newer producer): 0x" + std::to_string(in_bits));
        if (out_bits & ~MODEL_OUTPUT_KNOWN_MASK)
            throw UnsupportedInterfaceError("model interface declares reserved output bits (written "
                                            "by a newer producer): 0x" + std::to_string(out_bits));
        if (out_bits == 0u)
            throw UnsupportedInterfaceError("model interface declares no outputs");

        if (gives(ModelOutput::Spectrum) && domain.spectrum_bins <= 0)
            throw UnsupportedInterfaceError("interface promises Spectrum but the domain records no "
                                            "spectrum_bins");
        if (gives(ModelOutput::AngularFlux)
            && (domain.angular_phi_segments <= 0 || domain.angular_theta_segments <= 0))
            throw UnsupportedInterfaceError("interface promises AngularFlux but the domain records no "
                                            "angular resolution");
        if (takes(ModelInput::TubeSpectrum) && domain.spectrum_bins <= 0)
            throw UnsupportedInterfaceError("interface consumes TubeSpectrum but the domain records no "
                                            "spectrum bins");
        if (takes(ModelInput::BeamCollimation)
            && domain.collimation == rfnn::io::CollimationType::None)
            throw UnsupportedInterfaceError("interface consumes BeamCollimation but the domain records "
                                            "no collimation kind");
        if (takes(ModelInput::SourceDistance) && takes(ModelInput::SourceOrigin3D))
            throw UnsupportedInterfaceError("interface consumes both SourceDistance and SourceOrigin3D; "
                                            "they are alternatives");
        if (resolution_aware && region_state_dims <= 0)
            throw UnsupportedInterfaceError("interface is resolution_aware but declares no "
                                            "region_state_dims");
    }
};

}  // namespace radfield3dnn
