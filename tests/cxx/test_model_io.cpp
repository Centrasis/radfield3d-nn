// RF3M model-package round-trip test: save an RF3M package (embedded ONNX + I/O domain +
// provenance + test metrics) via rfnn::ModelFactory, load it back, and verify the domain,
// provenance, metrics, and that the reconstructed (from-memory) TrainedModel produces the same
// prediction as the same ONNX loaded directly from disk.

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
using namespace rfnn::io::V1;    // NamedGraphs, ModelFactory, k*Graph
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
    domain.beam_parameters = {
        {"direction",     3, {-1.f, 1.f, ""}},
        {"distance",      1, {0.2f, 1.5f, "m"}},
        {"opening_angle", 1, {5.f, 40.f, "deg"}},
        {"spectrum",     32, {0.f, 150000.f, "eV"}},
    };

    ModelProvenance prov;
    prov.dataset_name = "DS03";
    prov.software_version = "RadFiled3D 1.3.3";
    prov.physics = "G4EmStandardPhysics_option4";

    const std::map<std::string, float> metrics = {
        {"test/airkerma_accuracy_scatter", 0.84f},
        {"test/airkerma_accuracy_top90",   0.71f},
    };

    NamedGraphs graphs;
    graphs[kTrunkGraph] = std::vector<uint8_t>(onnx.begin(), onnx.end());

    const std::string pkg = std::string(RFNN_TEST_DATA_DIR) + "/roundtrip.rf3m";
    ModelFactory::save(pkg, graphs, domain, prov, metrics);

    // load() parses the container AND builds the runnable predictor in one step, carrying the
    // package metadata on the returned predictor.
    std::unique_ptr<radfield3dnn::VolumeFieldPredictor> model =
        ModelFactory::load(pkg, /*use_cuda=*/false);
    ASSERT_TRUE(model != nullptr);

    // Provenance.
    EXPECT_EQ(model->provenance().dataset_name, prov.dataset_name);
    EXPECT_EQ(model->provenance().software_version, prov.software_version);
    EXPECT_EQ(model->provenance().physics, prov.physics);

    // Metrics.
    ASSERT_EQ(model->metrics().size(), metrics.size());
    for (const auto& [k, v] : metrics)
        EXPECT_NEAR(model->metrics().at(k), v, 1e-5f);

    // Domain (metric units; no spatial geometry stored).
    EXPECT_EQ(model->domain().spectrum_bins, domain.spectrum_bins);
    EXPECT_NEAR(model->domain().spectrum_max_energy_ev, domain.spectrum_max_energy_ev, 1e-3f);

    // Beam-parameter descriptor list (name, slot count, range, unit).
    ASSERT_EQ(model->domain().beam_parameters.size(), 4u);
    const auto& dist = model->domain().beam_parameters[1];
    EXPECT_EQ(dist.name, "distance");
    EXPECT_EQ(dist.count, 1);
    EXPECT_NEAR(dist.range.min, 0.2f, 1e-6f);
    EXPECT_NEAR(dist.range.max, 1.5f, 1e-6f);
    EXPECT_EQ(dist.range.unit, "m");
    EXPECT_EQ(model->domain().beam_parameters[3].name, "spectrum");
    EXPECT_EQ(model->domain().beam_parameters[3].count, 32);

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
