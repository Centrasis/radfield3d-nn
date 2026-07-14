// RF3M model-package round-trip: save a package (embedded ONNX + I/O domain + provenance + test
// metrics) via rfnn::io::V1::ModelStore, load it back, and verify the domain, provenance, metrics,
// and that the reconstructed predictor produces the same prediction as the same ONNX loaded
// directly from disk.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <map>
#include <memory>
#include <vector>

#include "radfield3d-nn/model_io.h"
#include "radfield3d-nn/field_predictors.h"

#ifndef RFNN_TEST_DATA_DIR
#define RFNN_TEST_DATA_DIR "."
#endif

using namespace rfnn::io;       // ParameterRange, BeamParameter, ModelDomain, ModelProvenance
using namespace rfnn::io::V1;    // NamedGraphs, ModelStore, k*Graph
using radfield3dnn::BeamParameters;
using radfield3dnn::PredictorType;
using radfield3dnn::VoxelFieldPredictor;

namespace {
std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto n = f.tellg(); f.seekg(0);
    std::vector<char> b(static_cast<size_t>(n));
    f.read(b.data(), n);
    return b;
}
const std::string kOnnx = std::string(RFNN_TEST_DATA_DIR) + "/tiny_voxel_mlp.onnx";
}  // namespace

TEST(ModelIo, RoundTripDomainMetricsAndModel) {
    auto onnx = read_file(kOnnx);
    ASSERT_FALSE(onnx.empty()) << "missing fixture: " << kOnnx;

    ModelDomain domain;
    domain.spectrum_bins = 32;
    domain.spectrum_max_energy_ev = 150000.f;
    const auto minmax = [](float lo, float hi, const char* unit) {
        ParameterRange r;
        r.type = ParameterRangeType::MinMax;
        r.min = lo; r.max = hi; r.unit = unit;
        return r;
    };
    ParameterRange spectrum_range;   // a histogram range: bins = round((max - min) / bin_width)
    spectrum_range.type = ParameterRangeType::Spectrum;
    spectrum_range.min = 0.f;
    spectrum_range.max = 150000.f;
    spectrum_range.bin_width = 150000.f / 32.f;
    spectrum_range.unit = "eV";

    domain.beam_parameters = {
        {"direction",     minmax(-1.f, 1.f, "")},
        {"distance",      minmax(0.2f, 1.5f, "m")},
        {"opening_angle", minmax(5.f, 40.f, "deg")},
        {"spectrum",      spectrum_range},
    };

    ModelProvenance prov;
    prov.dataset_name = "DS03";
    prov.software_version = "RadFiled3D 1.3.3";
    prov.physics = "G4EmStandardPhysics_option4";

    const std::map<std::string, float> metrics = {
        {"test/airkerma_accuracy_scatter", 0.84f},
        {"test/airkerma_accuracy_top90",   0.71f},
    };

    // The declared I/O. Position => this package is a per-voxel model, so load() must hand back a
    // VoxelFieldPredictor. The domain above carries the spectrum bins the Spectrum output needs.
    radfield3dnn::ModelInterface iface;
    iface.inputs  = radfield3dnn::ModelInput::Position;
    iface.outputs = radfield3dnn::ModelOutput::Flux | radfield3dnn::ModelOutput::Spectrum;

    NamedGraphs graphs;
    graphs[kTrunkGraph] = std::vector<uint8_t>(onnx.begin(), onnx.end());

    const std::string pkg = std::string(RFNN_TEST_DATA_DIR) + "/roundtrip.rf3m";
    ModelStore::save(pkg, graphs, iface, domain, prov, metrics);

    // load() parses the container AND builds the runnable predictor in one step, carrying the
    // package metadata on the returned predictor.
    std::unique_ptr<radfield3dnn::VolumeFieldPredictor> model =
        ModelStore::load(pkg, /*use_cuda=*/false);
    ASSERT_TRUE(model != nullptr);

    // Provenance.
    EXPECT_EQ(model->provenance().dataset_name, prov.dataset_name);
    EXPECT_EQ(model->provenance().software_version, prov.software_version);
    EXPECT_EQ(model->provenance().physics, prov.physics);

    // Metrics.
    ASSERT_EQ(model->metrics().size(), metrics.size());
    for (const auto& [k, v] : metrics)
        EXPECT_NEAR(model->metrics().at(k), v, 1e-5f);

    // The interface survives the round-trip, and it is what selected the predictor class.
    EXPECT_EQ(model->interface().id(), iface.id());
    EXPECT_TRUE(model->interface().takes(radfield3dnn::ModelInput::Position));
    EXPECT_TRUE(model->interface().gives(radfield3dnn::ModelOutput::Spectrum));

    // Domain (metric units; no spatial geometry stored).
    EXPECT_EQ(model->domain().spectrum_bins, domain.spectrum_bins);
    EXPECT_NEAR(model->domain().spectrum_max_energy_ev, domain.spectrum_max_energy_ev, 1e-3f);

    // Beam-parameter descriptor list (name + typed range).
    ASSERT_EQ(model->domain().beam_parameters.size(), 4u);
    const auto& dist = model->domain().beam_parameters[1];
    EXPECT_EQ(dist.name, "distance");
    EXPECT_EQ(dist.range.type, ParameterRangeType::MinMax);
    EXPECT_NEAR(dist.range.min, 0.2f, 1e-6f);
    EXPECT_NEAR(dist.range.max, 1.5f, 1e-6f);
    EXPECT_EQ(dist.range.unit, "m");

    const auto& spec = model->domain().beam_parameters[3];
    EXPECT_EQ(spec.name, "spectrum");
    EXPECT_EQ(spec.range.type, ParameterRangeType::Spectrum);
    EXPECT_NEAR(spec.range.bin_width, 150000.f / 32.f, 1e-3f);  // -> 32 bins

    // The package built into a VoxelFieldPredictor (the trunk graph is per-voxel) whose prediction
    // matches the same ONNX loaded directly. The trunk graph name is carried in graph_names().
    const auto& names = model->graph_names();
    ASSERT_NE(std::find(names.begin(), names.end(), kTrunkGraph), names.end());
    ASSERT_EQ(model->type(), PredictorType::VoxelField);
    auto* voxel = static_cast<VoxelFieldPredictor*>(model.get());

    VoxelFieldPredictor direct(onnx.data(), onnx.size(), /*beam_encoder=*/nullptr, /*use_cuda=*/false);
    BeamParameters beam;
    beam.spectrum = {1.f};  // unused by this position-only model, but a valid beam
    std::vector<std::array<float, 3>> pts = {{0.1f, 0.2f, 0.3f}, {0.5f, 0.5f, 0.5f}};

    auto a = direct.predict_voxelwise(pts, direct.encode_beam(beam));
    auto b = voxel->predict_voxelwise(pts, voxel->encode_beam(beam));
    ASSERT_EQ(a.flux.size(), b.flux.size());
    for (size_t i = 0; i < a.flux.size(); ++i) EXPECT_FLOAT_EQ(a.flux[i], b.flux[i]);
    ASSERT_EQ(a.spectrum.size(), b.spectrum.size());
    for (size_t i = 0; i < a.spectrum.size(); ++i) EXPECT_FLOAT_EQ(a.spectrum[i], b.spectrum[i]);
}
