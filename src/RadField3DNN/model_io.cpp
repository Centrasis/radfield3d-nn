#include "radfield3d-nn/model_io.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

// rfnn::ModelFactory — the deployment-side ONNX package factory. See model_io.h for the
// authoritative RF3M container layout (this file is the format's reference implementation).
// The package stores only the model's fixed I/O domain (no spatial field geometry — the
// predicted resolution is supplied at inference), so this TU has no RadFiled3D dependency.

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

LoadedModel ModelFactory::load_from_memory(const void* bytes, size_t n) {
    const std::string buf(static_cast<const char*>(bytes), n);
    std::istringstream is(buf, std::ios::binary);

    char magic[4]; is.read(magic, 4);
    if (!is || std::memcmp(magic, kMagic, 4) != 0)
        throw std::runtime_error("model_io: bad magic (not an RF3M package)");
    const uint32_t version = get<uint32_t>(is);
    if (version != kVersion)
        throw std::runtime_error("model_io: unsupported RF3M version " + std::to_string(version));

    LoadedModel out;
    out.provenance.dataset_name     = get_str(is);
    out.provenance.software_version = get_str(is);
    out.provenance.physics          = get_str(is);
    out.domain = get_domain(is);

    const uint32_t n_metrics = get<uint32_t>(is);
    for (uint32_t i = 0; i < n_metrics; ++i) {
        const std::string k = get_str(is);
        out.metrics[k] = get<float>(is);
    }

    const uint32_t n_graphs = get<uint32_t>(is);
    for (uint32_t i = 0; i < n_graphs; ++i) {
        std::string name = get_str(is);
        const uint64_t len = get<uint64_t>(is);
        std::vector<uint8_t> g(static_cast<size_t>(len));
        is.read(reinterpret_cast<char*>(g.data()), static_cast<std::streamsize>(len));
        if (!is) throw std::runtime_error("model_io: truncated graph payload");
        out.graphs.emplace(std::move(name), std::move(g));
    }
    return out;
}

LoadedModel ModelFactory::load(const std::string& path) {
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is) throw std::runtime_error("model_io: cannot open '" + path + "'");
    const std::streamsize n = is.tellg();
    is.seekg(0);
    std::vector<char> buf(static_cast<size_t>(n));
    is.read(buf.data(), n);
    if (!is) throw std::runtime_error("model_io: failed reading '" + path + "'");
    return load_from_memory(buf.data(), buf.size());
}

// LoadedModel::build() is defined in field_predictors.cpp (it needs the predictor classes + ORT);
// this TU stays free of any ONNX Runtime dependency.

}  // namespace V1
}  // namespace io
}  // namespace rfnn
