#pragma once
#include <string>
#include <cstdint>

namespace rfnn::tcnn {
    // blocks/: helpers that emit a multi-layer MLP block, as opposed to the
    // single-layer modules in layers/.
    //
    // Pick the tcnn MLP backend for an inner MLP block.
    //
    // FullyFusedMLP is the fast warp-pipelined kernel but is only valid when the
    // hidden width fits the fused tiles (<= 128) and there is at least one
    // hidden layer (FullyFusedMLP throws on n_hidden_layers == 0). In every MLP
    // we build, the hidden width (n_neurons) equals d_model / feature_channels,
    // which is bounded by the input width, so keying the choice off the input
    // and output widths is sufficient: use FullyFusedMLP iff
    //   in_features <= 128 AND out_features <= 128 AND n_hidden_layers >= 1,
    // otherwise fall back to CutlassMLP (which handles wide/narrow dims and the
    // zero-hidden-layer single-Linear case).
    inline std::string select_mlp_otype(uint32_t in_features, uint32_t out_features, uint32_t n_hidden_layers) {
        const bool fully_fused = (in_features <= 128u) && (out_features <= 128u) && (n_hidden_layers >= 1u);
        return fully_fused ? std::string("FullyFusedMLP") : std::string("CutlassMLP");
    }
}
