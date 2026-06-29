// Device-inference correctness: load an RF3M model, run the whole-field prediction BOTH ways — host
// (predict_volume) and device-only (predict_to_device, ONNX outputs bound to CUDA device memory) — copy
// the device flux back, and confirm they match. This validates the zero-copy device path (step 3/4); the
// CUDA->Vulkan-texture half is validated separately by RadField.TestCudaVulkanInterop.
//
// Build (UE clang+libc++ for ABI match with the staged deploy lib) + run: see the inline command in the
// chat / Scripts. Usage: test_cuda_device_inference <model.rf3m> [dim]

#include "radfield3d-nn/model_io.h"
#include "radfield3d-nn/field_predictors.h"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace radfield3dnn;

int main(int argc, char** argv)
{
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.rf3m> [dim]\n", argv[0]); return 2; }
    const int dim = (argc >= 3) ? std::atoi(argv[2]) : 48;

    std::unique_ptr<VolumeFieldPredictor> pred;
    try { pred = rfnn::io::V1::ModelStore::load(argv[1], /*use_cuda*/ true); }
    catch (const std::exception& e) { std::fprintf(stderr, "load failed: %s\n", e.what()); return 2; }
    if (!pred) { std::fprintf(stderr, "load returned null\n"); return 2; }

    std::printf("predictor type: %s\n",
                pred->type() == PredictorType::VolumeField ? "VolumeField" : "VoxelField (tiled)");

    const std::array<int, 3> dims{ dim, dim, dim };
    BeamParameters beam;   // defaults: direction (0,0,-1), origin (0.5,0.5,0.5)
    // Fill the tube-spectrum input the model expects (uniform, normalized) — else ORT rejects the shape.
    // The beam-encoder graph's spectrum length is the ground truth (the domain metadata can disagree), so
    // allow an explicit override via argv[3]; default to the domain layout. (dump_model_inputs shows it.)
    int sbins = pred->input_spectrum_layout().bins;
    if (argc >= 4) sbins = std::atoi(argv[3]);
    if (sbins > 0) beam.spectrum.assign((size_t)sbins, 1.0f / (float)sbins);
    std::printf("input spectrum bins: %d\n", sbins);

    FieldPrediction host;
    try { host = pred->predict_volume(beam, dims); }
    catch (const std::exception& e) { std::fprintf(stderr, "predict_volume failed: %s\n", e.what()); return 1; }

    DeviceFieldOutputs* dev = nullptr;
    try { dev = pred->predict_to_device(beam, dims, /*device_id*/ 0); }
    catch (const std::exception& e) { std::fprintf(stderr, "predict_to_device threw: %s\n", e.what()); return 1; }
    if (!dev) { std::fprintf(stderr, "predict_to_device returned null (no GPU EP?)\n"); return 1; }

    if (device_outputs_is_fp16(dev)) {
        std::printf("model outputs fp16 — this float32 comparison test skips it (mechanism is identical). SKIP\n");
        release_device_outputs(dev);
        return 0;
    }

    const size_t n = device_outputs_voxel_count(dev);
    std::vector<float> dev_flux(n, 0.f);
    const cudaError_t e = cudaMemcpy(dev_flux.data(), device_outputs_flux(dev),
                                     n * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) { std::fprintf(stderr, "cudaMemcpy D2H: %s\n", cudaGetErrorString(e)); release_device_outputs(dev); return 1; }

    double max_abs = 0.0, host_sum = 0.0, dev_sum = 0.0;
    size_t dev_nonzero = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double)dev_flux[i] - (double)host.flux[i]);
        if (d > max_abs) max_abs = d;
        host_sum += host.flux[i]; dev_sum += dev_flux[i];
        if (dev_flux[i] != 0.f) ++dev_nonzero;
    }
    std::printf("voxels=%zu  host_sum=%.6g  dev_sum=%.6g  dev_nonzero=%zu  max|dev-host|=%.6g\n",
                n, host_sum, dev_sum, dev_nonzero, max_abs);

    const bool pass = (dev_nonzero > 0) && (max_abs <= 1e-3);
    std::printf("%s\n", pass ? "PASS: device-bound ONNX flux matches the host path"
                             : "FAIL: device flux differs from host (or all zero)");
    release_device_outputs(dev);

    // ── Timing the DEVICE path (what the renderer uses): warm-up, then N runs. The provider (TensorRT vs
    //    CUDA EP) is whatever registered for this session — run with/without the TRT libs on the path to
    //    compare. ORT Run() is synchronous, so each predict_to_device returns with results ready. ──
    const int iters = 30;
    DeviceFieldOutputs* warm = pred->predict_to_device(beam, dims, 0);
    if (warm) release_device_outputs(warm);
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        DeviceFieldOutputs* d = pred->predict_to_device(beam, dims, 0);
        if (d) release_device_outputs(d);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    std::printf("DEVICE-PATH TIMING: %.3f ms/inference @ %d^3 (%d iters)\n", ms, dim, iters);
    return pass ? 0 : 1;
}
