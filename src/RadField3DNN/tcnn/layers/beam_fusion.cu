#include "radfield3d-nn/tcnn/layers/beam_fusion.h"
#include "radfield3d-nn/tcnn/layers/film.h"
#include "radfield3d-nn/tcnn/layers/gated_fusion.h"
#include <stdexcept>

namespace rfnn::tcnn {

BeamFusionKind parse_beam_fusion_kind(const std::string& kind) {
    if (kind == "film" || kind == "FiLM" || kind == "affine") return BeamFusionKind::FiLM;
    if (kind == "gated" || kind == "GatedFusion" || kind == "gated_fusion") return BeamFusionKind::GatedFusion;
    throw std::runtime_error("Unknown beam_fusion kind '" + kind + "'. Use 'film' or 'gated'.");
}

std::unique_ptr<BeamFusionModule> create_beam_fusion(
    BeamFusionKind kind,
    uint32_t feature_channels,
    uint32_t condition_channels,
    const std::string& non_linearity)
{
    switch (kind) {
        case BeamFusionKind::FiLM:
            return std::make_unique<FiLM>(feature_channels, condition_channels, non_linearity);
        case BeamFusionKind::GatedFusion:
            return std::make_unique<GatedFusion>(feature_channels, condition_channels, non_linearity);
        default:
            throw std::runtime_error("create_beam_fusion: unhandled BeamFusionKind");
    }
}

} // namespace rfnn::tcnn
