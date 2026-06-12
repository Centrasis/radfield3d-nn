#include "radfield3d-nn/model_io.h"
#include "radfield3d-nn/field_predictors.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

// rfnn::ModelFactory — the deployment-side ONNX package factory. See model_io.h for the
// authoritative RF3M container layout (this file is the format's reference implementation).
// load() both parses the container AND assembles the runnable predictor; it does so only through
// the predictor's PUBLIC declared API (ctors / set_parameter_range / is_voxelwise), so this TU
// pulls in no ONNX Runtime headers and stores only the model's fixed I/O domain (no spatial field
// geometry — the predicted resolution is supplied at inference) and no RadFiled3D types.

namespace rfnn {
namespace io {
namespace V1 {
namespace {

template <class T> void put(std::ostream& os, const T& v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <class T> T get(std::istream& is) {
    T v{}; is.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!is) throw std::runtime_error("model_io: truncated RF3M package");
    return v;
}
void put_str(std::ostream& os, const std::string& s) {
    put<uint32_t>(os, static_cast<uint32_t>(s.size()));
    os.write(s.data(), static_cast<std::streamsize>(s.size()));
}
std::string get_str(std::istream& is) {
    const uint32_t n = get<uint32_t>(is);
    std::string s(n, '\0');
    is.read(s.data(), n);
    if (!is) throw std::runtime_error("model_io: truncated string field");
    return s;
}

void put_domain(std::ostream& os, const ModelDomain& d) {
    put<int32_t>(os, d.spectrum_bins);
    put<float>(os, d.spectrum_max_energy_ev);
    put<uint32_t>(os, static_cast<uint32_t>(d.beam_parameters.size()));
    for (const auto& p : d.beam_parameters) {
        put_str(os, p.name);
        put<int32_t>(os, p.count);
        put<float>(os, p.range.min);
        put<float>(os, p.range.max);
        put_str(os, p.range.unit);
    }
}
ModelDomain get_domain(std::istream& is) {
    ModelDomain d;
    d.spectrum_bins = get<int32_t>(is);
    d.spectrum_max_energy_ev = get<float>(is);
    const uint32_t n = get<uint32_t>(is);
    d.beam_parameters.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        BeamParameter p;
        p.name       = get_str(is);
        p.count      = get<int32_t>(is);
        p.range.min  = get<float>(is);
        p.range.max  = get<float>(is);
        p.range.unit = get_str(is);
        d.beam_parameters.push_back(std::move(p));
    }
    return d;
}

}  // namespace

std::vector<uint8_t> ModelFactory::save_to_memory(const NamedGraphs& graphs,
                                                  const ModelDomain& domain,
                                                  const ModelProvenance& provenance,
                                                  const std::map<std::string, float>& metrics) {
    std::ostringstream os(std::ios::binary);
    os.write(kMagic, 4);
    put<uint32_t>(os, kVersion);
    put_str(os, provenance.dataset_name);
    put_str(os, provenance.software_version);
    put_str(os, provenance.physics);
    put_domain(os, domain);
    put<uint32_t>(os, static_cast<uint32_t>(metrics.size()));
    for (const auto& [k, v] : metrics) { put_str(os, k); put<float>(os, v); }
    put<uint32_t>(os, static_cast<uint32_t>(graphs.size()));
    for (const auto& [name, onnx] : graphs) {
        put_str(os, name);
        put<uint64_t>(os, static_cast<uint64_t>(onnx.size()));
        os.write(reinterpret_cast<const char*>(onnx.data()), static_cast<std::streamsize>(onnx.size()));
    }
    const std::string s = os.str();
    return std::vector<uint8_t>(s.begin(), s.end());
}

void ModelFactory::save(const std::string& path,
                        const NamedGraphs& graphs,
                        const ModelDomain& domain,
                        const ModelProvenance& provenance,
                        const std::map<std::string, float>& metrics) {
    const std::vector<uint8_t> bytes = save_to_memory(graphs, domain, provenance, metrics);
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) throw std::runtime_error("model_io: cannot open '" + path + "' for writing");
    os.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!os) throw std::runtime_error("model_io: write failed for '" + path + "'");
}

std::unique_ptr<radfield3dnn::VolumeFieldPredictor>
ModelFactory::load_from_memory(const void* bytes, size_t n, bool use_cuda) {
    const std::string buf(static_cast<const char*>(bytes), n);
    std::istringstream is(buf, std::ios::binary);

    char magic[4]; is.read(magic, 4);
    if (!is || std::memcmp(magic, kMagic, 4) != 0)
        throw std::runtime_error("model_io: bad magic (not an RF3M package)");
    const uint32_t version = get<uint32_t>(is);
    if (version != kVersion)
        throw std::runtime_error("model_io: unsupported RF3M version " + std::to_string(version));

    // ── parse the container ──────────────────────────────────────────────────────────────────
    rfnn::io::ModelProvenance provenance;
    provenance.dataset_name     = get_str(is);
    provenance.software_version = get_str(is);
    provenance.physics          = get_str(is);
    const rfnn::io::ModelDomain domain = get_domain(is);

    std::map<std::string, float> metrics;
    const uint32_t n_metrics = get<uint32_t>(is);
    for (uint32_t i = 0; i < n_metrics; ++i) {
        const std::string k = get_str(is);
        metrics[k] = get<float>(is);
    }

    NamedGraphs graphs;
    const uint32_t n_graphs = get<uint32_t>(is);
    for (uint32_t i = 0; i < n_graphs; ++i) {
        std::string name = get_str(is);
        const uint64_t len = get<uint64_t>(is);
        std::vector<uint8_t> g(static_cast<size_t>(len));
        is.read(reinterpret_cast<char*>(g.data()), static_cast<std::streamsize>(len));
        if (!is) throw std::runtime_error("model_io: truncated graph payload");
        graphs.emplace(std::move(name), std::move(g));
    }

    // ── build the runnable predictor from the embedded graphs (the former LoadedModel::build) ──
    auto trunk_it = graphs.find(kTrunkGraph);
    if (trunk_it == graphs.end())
        throw std::runtime_error("model_io: package has no '" + std::string(kTrunkGraph) + "' graph");
    const std::vector<uint8_t>& trunk = trunk_it->second;

    // The ModelDomain carries the [min,max] metric range of each beam parameter; register them on
    // whichever predictor consumes the beam parameters (the beam encoder for a two-graph model,
    // else the trunk) so its metric inputs are clipped+normalised to [0,1] before encoding,
    // matching training. Constructing a predictor only touches its declared public API — no ORT
    // header is pulled into this TU.
    auto apply_ranges = [&domain](radfield3dnn::VolumeFieldPredictor& p) {
        for (const auto& bp : domain.beam_parameters)
            p.set_parameter_range(bp.name, bp.range.min, bp.range.max);
    };

    // The graph names the predictor was composed from (carried as metadata for introspection).
    std::vector<std::string> graph_names;
    graph_names.reserve(graphs.size());
    for (const auto& kv : graphs) graph_names.push_back(kv.first);

    // Build the trunk once (the expensive step — TRT engine build). A field-wise trunk stays a
    // VolumeFieldPredictor; a per-voxel trunk is move-adopted (no re-load) into a
    // VoxelFieldPredictor, wired with the "beam_encoder" graph if the package carries one.
    auto trunk_pred = std::make_unique<radfield3dnn::VolumeFieldPredictor>(
        trunk.data(), trunk.size(), use_cuda);
    apply_ranges(*trunk_pred);  // single-graph models bind the beam params on the trunk

    std::unique_ptr<radfield3dnn::VolumeFieldPredictor> predictor;
    if (!trunk_pred->is_voxelwise()) {
        predictor = std::move(trunk_pred);
    } else {
        std::shared_ptr<radfield3dnn::VolumeFieldPredictor> encoder;
        auto eit = graphs.find(kBeamEncoderGraph);
        if (eit != graphs.end()) {
            encoder = std::make_shared<radfield3dnn::VolumeFieldPredictor>(
                eit->second.data(), eit->second.size(), use_cuda);
            apply_ranges(*encoder);  // two-graph models bind the beam params on the encoder
        }
        predictor = std::make_unique<radfield3dnn::VoxelFieldPredictor>(
            std::move(*trunk_pred), std::move(encoder));
    }

    predictor->set_package_metadata(domain, std::move(provenance), std::move(metrics),
                                    std::move(graph_names));
    return predictor;
}

std::unique_ptr<radfield3dnn::VolumeFieldPredictor>
ModelFactory::load(const std::string& path, bool use_cuda) {
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is) throw std::runtime_error("model_io: cannot open '" + path + "'");
    const std::streamsize n = is.tellg();
    is.seekg(0);
    std::vector<char> buf(static_cast<size_t>(n));
    is.read(buf.data(), n);
    if (!is) throw std::runtime_error("model_io: failed reading '" + path + "'");
    return load_from_memory(buf.data(), buf.size(), use_cuda);
}

}  // namespace V1
}  // namespace io
}  // namespace rfnn
