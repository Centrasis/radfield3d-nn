#pragma once
#include <memory>
#include <string>
#include <tiny-cuda-nn/object.h>
#include <tiny-cuda-nn/common.h>

namespace rfnn::tcnn {
    // Common type of a beam-encoding fusion layer: it consumes the concat
    // [feature, condition] and emits the conditioned feature, as a JIT-fused
    // DifferentiableObject. Both FiLM and GatedFusion satisfy this contract, so
    // BaseRadiationPredictionModel can hold either one via this base type.
    using BeamFusionModule = ::tcnn::DifferentiableObject<float, ::tcnn::network_precision_t, ::tcnn::network_precision_t>;

    enum class BeamFusionKind {
        FiLM = 0,         // affine: (1 + gamma) * feature + beta
        GatedFusion = 1   // bounded gate: hardsigmoid(g) * feature + (1 - g) * candidate
    };

    BeamFusionKind parse_beam_fusion_kind(const std::string& kind);

    // Build a beam fusion layer of the requested kind. feature/condition widths
    // and the optional non-linearity are forwarded to the concrete layer.
    std::unique_ptr<BeamFusionModule> create_beam_fusion(
        BeamFusionKind kind,
        uint32_t feature_channels,
        uint32_t condition_channels,
        const std::string& non_linearity);
};
