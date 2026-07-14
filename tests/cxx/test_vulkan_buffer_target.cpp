// rfnn::vk::VulkanBufferTarget — registering an EXISTING externally-owned allocation as a direct
// prediction target, through the public API only.
//
// No Vulkan SDK is required: the test fabricates the external allocation with the CUDA driver API
// (cuMemCreate + cuMemExportToShareableHandle), which produces the SAME kind of opaque POSIX FD a
// Vulkan vkGetMemoryFdKHR export yields — so register_buffer() exercises the identical import path.
// Verification reads the memory back through a SECOND export of the same allocation, proving the
// aliasing works and the target's own mapping stays hidden.
//
// Skips cleanly when no CUDA device (or no driver) is present.

#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <vector>

#include "radfield3d-nn/field_predictors.h"
#include "radfield3d-nn/model_io.h"
#include "radfield3d-nn/vk/cuda_vulkan_export.h"
#include "radfield3d-nn/vk/vulkan_buffer_target.h"

#ifndef RFNN_TEST_MODEL_PKG
#define RFNN_TEST_MODEL_PKG \
    "/mnt/data/models_nn/logs/pbrf-roifull-floor1e8/pbrf-roifull-floor1e8-ds03/models/PBRFNet.rf3m"
#endif

using namespace radfield3dnn;

namespace {

// An externally-owned device allocation exported as opaque FDs — the stand-in for a Vulkan buffer.
struct ExternalAllocation {
    CUmemGenericAllocationHandle handle{};
    size_t size = 0;
    bool ok = false;

    bool create(int device, size_t min_bytes) {
        if (cuInit(0) != CUDA_SUCCESS) return false;
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        size_t gran = 0;
        if (cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS)
            return false;
        size = ((min_bytes + gran - 1) / gran) * gran;
        ok = cuMemCreate(&handle, size, &prop, 0) == CUDA_SUCCESS;
        return ok;
    }
    int export_fd() const {
        int fd = -1;
        if (cuMemExportToShareableHandle(&fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0)
            != CUDA_SUCCESS)
            return -1;
        return fd;
    }
    ~ExternalAllocation() {
        if (ok) cuMemRelease(handle);
    }
};

// Read the allocation back through its own import — the observer Vulkan would be.
std::vector<float> read_back(int fd, size_t alloc_size, size_t bytes) {
    cudaExternalMemoryHandleDesc mem_desc{};
    mem_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    mem_desc.handle.fd = fd;
    mem_desc.size = alloc_size;
    cudaExternalMemory_t ext{};
    if (cudaImportExternalMemory(&ext, &mem_desc) != cudaSuccess) return {};
    cudaExternalMemoryBufferDesc buf_desc{};
    buf_desc.size = bytes;
    void* dptr = nullptr;
    std::vector<float> host(bytes / sizeof(float));
    if (cudaExternalMemoryGetMappedBuffer(&dptr, ext, &buf_desc) == cudaSuccess) {
        cudaMemcpy(host.data(), dptr, bytes, cudaMemcpyDeviceToHost);
        cudaFree(dptr);
    } else {
        host.clear();
    }
    cudaDestroyExternalMemory(ext);
    return host;
}

bool has_cuda() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

}  // namespace

TEST(VulkanBufferTargetTest, PredictorWritesTheExternalAllocationDirectly) {
    if (!has_cuda()) GTEST_SKIP() << "no CUDA device";

    // The device UUID is what a renderer would hand over from its VkPhysicalDevice.
    cudaDeviceProp prop{};
    ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
    const auto* uuid = reinterpret_cast<const uint8_t*>(prop.uuid.bytes);
    ASSERT_EQ(rfnn::cuda_vk::cuda_device_for_uuid(uuid), 0);

    constexpr int N = 16;
    const size_t flux_bytes = size_t(N) * N * N * sizeof(float);

    ExternalAllocation alloc;
    if (!alloc.create(0, flux_bytes)) GTEST_SKIP() << "driver VMM export unsupported";

    // Fill with a sentinel through one export, so an unwritten buffer cannot pass the test.
    {
        int fd = alloc.export_fd();
        ASSERT_GE(fd, 0);
        cudaExternalMemoryHandleDesc d{};
        d.type = cudaExternalMemoryHandleTypeOpaqueFd;
        d.handle.fd = fd;
        d.size = alloc.size;
        cudaExternalMemory_t ext{};
        ASSERT_EQ(cudaImportExternalMemory(&ext, &d), cudaSuccess);
        cudaExternalMemoryBufferDesc b{};
        b.size = flux_bytes;
        void* p = nullptr;
        ASSERT_EQ(cudaExternalMemoryGetMappedBuffer(&p, ext, &b), cudaSuccess);
        std::vector<float> sentinel(flux_bytes / 4, -7.0f);
        cudaMemcpy(p, sentinel.data(), flux_bytes, cudaMemcpyHostToDevice);
        cudaFree(p);
        cudaDestroyExternalMemory(ext);
    }

    // ── the public flow a renderer uses ──────────────────────────────────────────────────────────
    const int fd = alloc.export_fd();
    ASSERT_GE(fd, 0);
    auto target = rfnn::vk::VulkanBufferTarget::register_buffer(uuid, fd, alloc.size,
                                                                /*mem_offset=*/0, flux_bytes);
    ASSERT_TRUE(static_cast<bool>(target));
    EXPECT_EQ(target.capacity_bytes(), flux_bytes);

    ExecutionOptions exec;
    exec.use_gpu = true;
    exec.use_tensorrt = false;
    exec.device_id = target.device_index();
    auto pred = rfnn::io::V1::ModelStore::load(RFNN_TEST_MODEL_PKG, exec);
    if (!pred->uses_gpu()) GTEST_SKIP() << "CUDA EP unavailable";

    pred->set_voxel_grid({N, N, N});
    std::vector<float> spectrum(pred->input_spectrum_bins(), 1.0f);
    std::vector<float> direction{0.f, 0.f, -1.f};
    std::vector<float> distance{0.f};
    auto host = [](std::vector<float>& v) {
        return MemoryRef{v.data(), v.size() * sizeof(float), DeviceKind::Cpu, ElementType::F32, 0};
    };
    // Host inputs on a CUDA session: an explicit convert (staged + uploaded per run).
    pred->bind_global_parameter(ModelInput::TubeSpectrum, host(spectrum), /*convert=*/true);
    pred->bind_global_parameter(ModelInput::BeamDirection, host(direction), /*convert=*/true);
    pred->bind_global_parameter(ModelInput::SourceDistance, host(distance), /*convert=*/true);

    target.bind_output(*pred, ModelOutput::Flux);
    pred->predict_volume();

    // ── verify through an independent import (the renderer's view of the same memory) ───────────
    const int verify_fd = alloc.export_fd();
    ASSERT_GE(verify_fd, 0);
    const std::vector<float> got = read_back(verify_fd, alloc.size, flux_bytes);
    ASSERT_EQ(got.size(), flux_bytes / 4);

    BeamParameters beam;
    beam.direction = {0.f, 0.f, -1.f};
    beam.origin = {0.5f, 0.5f, 0.5f};
    beam.spectrum = spectrum;
    const FieldPrediction ref = pred->predict_volume(beam, {N, N, N});

    double max_diff = 0.0;
    for (size_t i = 0; i < got.size(); ++i)
        max_diff = std::max(max_diff, double(std::abs(got[i] - ref.flux[i])));
    EXPECT_LT(max_diff, 1e-4) << "external buffer does not hold the prediction";

    // The sentinel must be gone (the model actually wrote, not the fill).
    size_t sentinels = 0;
    for (float v : got) sentinels += (v == -7.0f);
    EXPECT_EQ(sentinels, 0u);
}

TEST(VulkanBufferTargetTest, EmptyTargetRefusesToBind) {
    rfnn::vk::VulkanBufferTarget empty;
    EXPECT_FALSE(static_cast<bool>(empty));
    EXPECT_EQ(empty.device_index(), -1);
    if (!has_cuda()) GTEST_SKIP() << "no CUDA device (predictor needed for the bind attempt)";
    auto pred = rfnn::io::V1::ModelStore::load(RFNN_TEST_MODEL_PKG, false);
    EXPECT_THROW(empty.bind_output(*pred, ModelOutput::Flux), BindingError);
}
